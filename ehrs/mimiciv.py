import os
import logging
import pandas as pd
import numpy as np
import glob
from ehrs import register_ehr, EHR

logger = logging.getLogger(__name__)

@register_ehr("mimiciv")
class MIMICIV(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "mimiciv"

        if self.data_dir is None:
            self.data_dir = os.path.join(self.cache_dir, self.ehr_name)

            if not os.path.exists(self.data_dir):
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )
                self.download_ehr_from_url(
                    url="https://physionet.org/files/mimiciv/2.0/",
                    dest=self.data_dir
                )

        logger.info("Data directory is set to {}".format(self.data_dir))

        if self.ccs_path is None:
            self.ccs_path = os.path.join(self.cache_dir, "ccs_multi_dx_tool_2015.csv")

            if not os.path.exists(self.ccs_path):
                logger.info(
                    "`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet."
                )
                self.download_ccs_from_url(self.cache_dir)

        if self.gem_path is None:
            self.gem_path = os.path.join(self.cache_dir, "icd10cmtoicd9gem.csv")

            if not os.path.exists(self.gem_path):
                logger.info(
                    "`icd10cmtoicd9gem.csv` is not found so try to download from the internet."
                )
                self.download_icdgem_from_url(self.cache_dir)

        if self.ext is None:
            self.ext = self.infer_data_extension()

        self._icustay_fname = "icu/icustays" + self.ext
        self._patient_fname = "hosp/patients" + self.ext
        self._admission_fname = "hosp/admissions" + self.ext
        self._diagnosis_fname = "hosp/diagnoses_icd" + self.ext

        self.tables = [
            {
                "fname": "hosp/labevents" + self.ext,
                "timestamp": "charttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "labevent_id",
                    "storetime",
                    "subject_id",
                    "specimen_id"
                ],
                "code": ["itemid"],
                "desc": ["hosp/d_labitems" + self.ext],
                "desc_key": ["label"],
            },
            {
                "fname": "hosp/prescriptions" + self.ext,
                "timestamp": "starttime",
                "timeoffsetunit": "abs",
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
                "timestamp": "starttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "endtime",
                    "storetime",
                    "orderid",
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

        if cfg.use_more_tables:
            self.tables+=[
                {
                    "fname": "icu/chartevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "storetime",
                        "subject_id",
                    ],
                    "code": ["itemid"],
                    "desc": ["icu/d_items" + self.ext],
                    "desc_key": ["label"],
                },
                {
                    "fname": "icu/outputevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "storetime",
                        "subject_id",
                    ],
                    "code": ["itemid"],
                    "desc": ["icu/d_items" + self.ext],
                    "desc_key": ["label"],
                },
                {
                    "fname": "hosp/microbiologyevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "chartdate",
                        "storetime",
                        "storedate",
                        "subject_id",
                        "microevent_id",
                        "micro_specimen_id",
                        "spec_itemid",
                        "test_itemid",
                        "org_itemid",
                        "ab_itemid"
                    ],
                },
                {
                    "fname": "icu/procedureevents" + self.ext,
                    "timestamp": "starttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "storetime",
                        "endtime",
                        "subject_id",
                        "orderid",
                        "linkorderid",
                        "continueinnextdept",
                        "statusdescription"
                    ],
                    "code": ["itemid"],
                    "desc": ["icu/d_items" + self.ext],
                    "desc_key": ["label"],
                },
            ]

        self._icustay_key = "stay_id"
        self._hadm_key = "hadm_id"
        self._patient_key = "subject_id"

        self._determine_first_icu = "INTIME"

    def build_cohorts(self, cached=False):
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustay_fname))

        icustays = self.make_compatible(icustays)
        self.icustays = icustays

        cohorts = super().build_cohorts(icustays, cached=cached)

        return cohorts

    def prepare_tasks(self, cohorts, cached=False):
        if cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled.dx")
            if labeled_cohorts is not None:
                self.labeled_cohorts = labeled_cohorts
                return labeled_cohorts

        labeled_cohorts = super().prepare_tasks(cohorts, cached)

        logger.info(
            "Start labeling cohorts for diagnosis prediction."
        )

        # define diagnosis prediction task
        diagnoses = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))

        diagnoses = self.icd10toicd9(diagnoses)

        diagnoses_with_cohorts = diagnoses[
            diagnoses[self.hadm_key].isin(labeled_cohorts[self.hadm_key])
        ]
        diagnoses_with_cohorts = (
            diagnoses_with_cohorts.groupby(self.hadm_key)["icd_code_converted"]
            .apply(list)
            .to_frame()
        )
        labeled_cohorts = labeled_cohorts.join(
            diagnoses_with_cohorts, on=self.hadm_key, how="left"
        )

        ccs_dx = pd.read_csv(self.ccs_path)
        ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
        ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1].astype(int) - 1
        lvl1 = {
            x: y for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()
        }

        # Some of patients(21) does not have dx codes
        labeled_cohorts.dropna(subset=["icd_code_converted"], inplace=True)

        labeled_cohorts["diagnosis"] = labeled_cohorts["icd_code_converted"].map(
            lambda dxs: list(set([lvl1[dx] for dx in dxs if dx in lvl1]))
        )

        labeled_cohorts = labeled_cohorts.drop(columns=["icd_code_converted"])
        labeled_cohorts.dropna(subset=["diagnosis"], inplace=True)

        self.labeled_cohorts = labeled_cohorts
        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled.dx")

        logger.info("Done preparing diagnosis prediction for the given cohorts")

        return labeled_cohorts

    def make_compatible(self, icustays):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patient_fname))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))

        # prepare icustays according to the appropriate format
        icustays = icustays.rename(columns={
            "los": "LOS",
            "intime": "INTIME",
            "outtime": "OUTTIME",
        })
        admissions = admissions.rename(columns={
            "dischtime": "DISCHTIME",
        })

        icustays = icustays[icustays["first_careunit"] == icustays["last_careunit"]]
        icustays.loc[:, "INTIME"] = pd.to_datetime(
            icustays["INTIME"], infer_datetime_format=True
        )
        icustays.loc[:, "OUTTIME"] = pd.to_datetime(
            icustays["OUTTIME"], infer_datetime_format=True
        )

        icustays = icustays.merge(patients, on="subject_id", how="left")
        icustays["AGE"] = (
            icustays["INTIME"].dt.year
            - icustays["anchor_year"]
            + icustays["anchor_age"]
        )

        icustays = icustays.merge(
            admissions[
                [self.hadm_key, "discharge_location", "deathtime", "DISCHTIME"]
            ],
            how="left",
            on=self.hadm_key,
        )

        icustays["discharge_location"].replace("DIED", "Death", inplace=True)
        icustays["DISCHTIME"] = pd.to_datetime(
            icustays["DISCHTIME"], infer_datetime_format=True
        )

        icustays["IN_ICU_MORTALITY"] = (
            (icustays["INTIME"] < icustays["DISCHTIME"])
            & (icustays["DISCHTIME"] <= icustays["OUTTIME"])
            & (icustays["discharge_location"] == "Death")
        )
        icustays.rename(columns={"discharge_location": "HOS_DISCHARGE_LOCATION"}, inplace=True)

        icustays["DISCHTIME"] = (icustays["DISCHTIME"] - icustays["INTIME"]).dt.total_seconds() // 60
        icustays["OUTTIME"] = (icustays["OUTTIME"] - icustays["INTIME"]).dt.total_seconds() // 60

        return icustays

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
                logger.warn("WRONG CODE: " + icd_code)

        dx["icd_code_converted"] = dx.apply(
            lambda x: icd_convert(x["icd_version"], x["icd_code"]), axis=1
        )
        return dx
    
    
    def infer_data_extension(self) -> str:
        if (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv.gz"))) == 21
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv.gz"))) == 8
        ):
            ext = ".csv.gz"
        elif (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv")))==21
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv")))==8
        ):
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext