import os
import logging
import pandas as pd
import glob
from ehrs import register_ehr, EHR

logger = logging.getLogger(__name__)


@register_ehr("mimiciv")
class MIMICIV(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        cache_dir = os.path.expanduser("~/.cache/ehr")
        physionet_file_path = "mimiciv/2.0/"

        self.data_dir = self.get_physionet_dataset(physionet_file_path, cache_dir)
        self.ccs_path = self.get_ccs(cache_dir)
        self.gem_path = self.get_gem(cache_dir)

        self.ext = ".csv.gz"
        if (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*" + self.ext))) != 21
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*" + self.ext))) != 8
        ):
            self.ext = ".csv"
            if (
                len(glob.glob(os.path.join(self.data_dir, "hosp", "*" + self.ext)))
                != 21
                or len(glob.glob(os.path.join(self.data_dir, "icu", "*" + self.ext)))
                != 8
            ):
                raise AssertionError(
                    "Provided data directory is not correct. Please check if --data is correct. "
                    "--data: {}".format(self.data_dir)
                )
        self.icustays = "icu/icustays" + self.ext
        self.patients = "hosp/patients" + self.ext
        self.admissions = "hosp/admissions" + self.ext
        self.diagnoses = "hosp/diagnoses_icd" + self.ext

        self.features = [
            {
                "fname": "hosp/labevents" + self.ext,
                "type": "lab",
                "timestamp": "charttime",
                "exclude": ["labevent_id", "storetime", "subject_id", "specimen_id"],
                "code": ["itemid"],
                "desc": ["hosp/d_labitems" + self.ext],
                "desc_key": ["label"],
            },
            {
                "fname": "hosp/prescriptions" + self.ext,
                "type": "med",
                "timestamp": "starttime",
                "exclude": [
                    "gsn",
                    "ndc",
                    "subject_id",
                    "pharmacy_id",
                    "poe_id",
                    "poe_seq",
                    "formulary_drug_cd",
                    "stoptime",
                ],
            },
            {
                "fname": "icu/inputevents" + self.ext,
                "type": "inf",
                "timestamp": "charttime",
                "exclude": [
                    "endtime",
                    "storetime" "orderid",
                    "linkorderid",
                    "subject_id",
                    "continueinnextdept",
                    "statusdescription",
                ],
                "code": ["itemid"],
                "desc": ["icu/d_items" + self.ext],
                "desc_key": ["label"],
            },
        ]

        self.icustay_key = "stay_id"
        self.icustay_start_key = "intime"
        self.icustay_end_key = "outtime"
        self.second_key = "hadm_id"

    def build_cohort(self):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patients))
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustays))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admissions))

        icustays = icustays[icustays["first_careunit"] == icustays["last_careunit"]]
        icustays = icustays[icustays["los"] >= (self.obs_size + self.gap_size) / 24]
        icustays[self.icustay_start_key] = pd.to_datetime(
            icustays[self.icustay_start_key]
        )
        icustays[self.icustay_end_key] = pd.to_datetime(icustays[self.icustay_end_key])

        patients_with_icustays = icustays.merge(patients, on="subject_id", how="left")
        patients_with_icustays["age"] = (
            patients_with_icustays[self.icustay_start_key].dt.year
            - patients_with_icustays["anchor_year"]
            + patients_with_icustays["anchor_age"]
        )

        patients_with_icustays = patients_with_icustays[
            (self.min_age <= patients_with_icustays["age"])
            & (patients_with_icustays["age"] <= self.max_age)
        ]

        patients_with_icustays = patients_with_icustays.merge(
            admissions[
                [self.second_key, "discharge_location", "deathtime", "dischtime"]
            ],
            how="left",
            on=self.second_key,
        )

        patients_with_icustays = self.readmission_label(patients_with_icustays)

        patients_with_icustays["deathtime"] = pd.to_datetime(
            patients_with_icustays["deathtime"]
        )
        patients_with_icustays["dischtime"] = pd.to_datetime(
            patients_with_icustays["dischtime"]
        )

        self.cohort = patients_with_icustays
        logger.info(
            "cohort has been built successfully. loaded {} cohorts.".format(
                len(self.cohort)
            )
        )
        patients_with_icustays.to_pickle(os.path.join(self.dest, "mimiciv.cohorts"))

        return patients_with_icustays

    def prepare_tasks(self):
        # readmission prediction
        labeled_cohort = self.cohort.copy()

        # los prediction
        labeled_cohort["los_3day"] = (self.cohort["los"] > 3).astype(int)
        labeled_cohort["los_7day"] = (self.cohort["los"] > 7).astype(int)

        # mortality prediction

        labeled_cohort["mortality"] = (
            (
                (
                    labeled_cohort[self.icustay_start_key]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.gap_size, unit="h")
                )
                <= labeled_cohort["deathtime"]
            )
            & (
                labeled_cohort["deathtime"]
                <= (
                    labeled_cohort[self.icustay_start_key]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.pred_size, unit="h")
                )
            )
        ).astype(int)

        labeled_cohort["in_icu_mortality"] = (
            labeled_cohort["deathtime"] > labeled_cohort[self.icustay_start_key]
        ) & (
            labeled_cohort["deathtime"] <= labeled_cohort[self.icustay_end_key]
        ).astype(
            int
        )

        # if an icustay is DEAD/EXPIRED, but not in_icu_mortality, then it is in_hospital_mortality
        labeled_cohort["in_hospital_mortality"] = 0
        labeled_cohort.loc[
            (labeled_cohort["discharge_location"] == "DIED")
            & (labeled_cohort["in_icu_mortality"] == 0),
            "in_hospital_mortality",
        ] = 1
        # define new class whose discharge location was 'DEAD/EXPIRED'
        labeled_cohort.loc[
            labeled_cohort["in_icu_mortality"] == 1, "discharge_location"
        ] = "IN_ICU_MORTALITY"
        labeled_cohort.loc[
            labeled_cohort["in_hospital_mortality"] == 1, "discharge_location"
        ] = "IN_HOSPITAL_MORTALITY"

        # define final acuity prediction task
        logger.info("Fincal Acuity Categories")
        logger.info(
            labeled_cohort["discharge_location"].astype("category").cat.categories
        )
        labeled_cohort["final_acuity"] = (
            labeled_cohort["discharge_location"].astype("category").cat.codes
        )

        # define imminent discharge prediction task
        is_discharged = (
            (
                (
                    labeled_cohort[self.icustay_start_key]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.gap_size, unit="h")
                )
                <= labeled_cohort["dischtime"]
            )
            & (
                labeled_cohort["dischtime"]
                <= (
                    labeled_cohort[self.icustay_start_key]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.pred_size, unit="h")
                )
            )
        ).astype(bool)
        labeled_cohort.loc[is_discharged, "imminent_discharge"] = labeled_cohort[
            is_discharged
        ]["discharge_location"]
        labeled_cohort.loc[~is_discharged, "imminent_discharge"] = "No Discharge"
        labeled_cohort.loc[
            (labeled_cohort["imminent_discharge"] == "IN_HOSPITAL_MORTALITY")
            | (labeled_cohort["imminent_discharge"] == "IN_ICU_MORTALITY"),
            "imminent_discharge",
        ] = "Death"

        logger.info("Immminent Discharge Categories")
        logger.info(
            labeled_cohort["imminent_discharge"].astype("category").cat.categories
        )
        labeled_cohort["imminent_discharge"] = (
            labeled_cohort["imminent_discharge"].astype("category").cat.codes
        )

        # drop unnecessary columns
        labeled_cohort = labeled_cohort.drop(
            columns=[
                self.icustay_start_key,
                "in_icu_mortality",
                "dischtime",
                "discharge_location",
                "in_hospital_mortality",
            ]
        )

        # define diagnosis prediction task
        dx = pd.read_csv(os.path.join(self.data_dir, self.diagnoses))

        dx = self.icd10toicd9(dx)

        diagnoses_with_cohort = dx[
            dx[self.second_key].isin(labeled_cohort[self.second_key])
        ]
        diagnoses_with_cohort = (
            diagnoses_with_cohort.groupby(self.second_key)["icd_code_converted"]
            .apply(list)
            .to_frame()
        )
        labeled_cohort = labeled_cohort.join(
            diagnoses_with_cohort, on=self.second_key, how="left"
        )

        ccs_dx = pd.read_csv(self.ccs_path)
        ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
        ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
        lvl1 = {
            x: y for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()
        }

        # Some of patients(21) does not have dx codes
        labeled_cohort.dropna(subset=["icd_code_converted"], inplace=True)

        labeled_cohort["diagnosis"] = labeled_cohort["icd_code_converted"].map(
            lambda dxs: list(set([lvl1[dx] for dx in dxs if dx in lvl1]))
        )

        labeled_cohort = labeled_cohort.drop(columns=["icd_code_converted"])
        labeled_cohort.dropna(subset=["diagnosis"], inplace=True)

        self.labeled_cohort = labeled_cohort
        logger.info("Done preparing tasks given the cohort sets")

        return labeled_cohort

    def encode(self):
        encoded_cohort = self.labeled_cohort.rename(
            columns={self.second_key: "id"}, inplace=False
        )
        # todo resume here

    # def run_pipeline(self):
    #     ...

    def icd10toicd9(self, dx):
        gem = pd.read_csv(self.gem_path)
        dx_icd_10 = dx[dx["icd_version"] == 10]["icd_code"]

        unique_elem_no_map = set(dx_icd_10) - set(gem["icd10cm"])

        map_cms = dict(zip(gem["icd10cm"], gem["icd9cm"]))
        map_manual = dict.fromkeys(unique_elem_no_map, "NaN")

        for code_10 in map_manual:
            for i in range(len(code_10), 0, -1):
                tgt_10 = code_10[:i]
                if tgt_10 in gem["icd10cm"]:
                    tgt_9 = (
                        gem[gem["icd10cm"].str.contains(tgt_10)]["icd9cm"]
                        .mode()
                        .iloc[0]
                    )
                    map_manual[code_10] = tgt_9
                    break

        def icd_convert(icd_version, icd_code):
            if icd_version == 9:
                return icd_code

            elif icd_code in map_cms:
                return map_cms[icd_code]

            elif icd_code in map_manual:
                return map_manual[icd_code]
            else:
                print("WRONG CODE: ", icd_code)

        dx["icd_code_converted"] = dx.apply(
            lambda x: icd_convert(x["icd_version"], x["icd_code"]), axis=1
        )
        return dx
