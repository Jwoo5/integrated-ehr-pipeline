import os
import sys
import logging

from datetime import datetime
import numpy as np
import pandas as pd

from ehrs import register_ehr, EHR
from utils.utils import get_physionet_dataset, get_ccs, get_gem

logger = logging.getLogger(__name__)


@register_ehr("mimiciv")
class MIMICIV(EHR):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.data_dir = cfg.data
        self.ccs_path = cfg.ccs
        self.gem_path = cfg.gem

        cache_dir = os.path.expanduser("~/.cache/ehr")
        physionet_file_path = "mimiciv/2.0/"

        self.data_dir = get_physionet_dataset(cfg, physionet_file_path, cache_dir)
        self.ccs_path = get_ccs(cfg, cache_dir)
        self.gem_path = get_gem(cfg, cache_dir)

        postfix = "" if cfg.data_uncompressed else ".gz"

        self.icustays = f"icu/icustays.csv{postfix}"
        self.patients = f"hosp/patients.csv{postfix}"
        self.admissions = f"hosp/admissions.csv{postfix}"
        self.diagnoses = f"hosp/diagnoses_icd.csv{postfix}"

        self.features = [
            {
                "fname": f"hosp/labevents.csv{postfix}",
                "type": "lab",
                "timestamp": "charttime",
            },
            {
                "fname": f"hosp/prescriptions.csv{postfix}",
                "type": "med",
                "timestamp": "starttime",
            },
            {
                "fname": f"icu/inputevents.csv{postfix}",
                "type": "inf",
                "timestamp": "charttime",
            },
        ]

        self.max_event_size = (
            cfg.max_event_size if cfg.max_event_size is not None else sys.maxsize
        )
        self.min_event_size = (
            cfg.min_event_size if cfg.min_event_size is not None else 0
        )
        assert self.min_event_size <= self.max_event_size, (
            self.min_event_size,
            self.max_event_size,
        )

        self.max_age = cfg.max_age if cfg.max_age is not None else sys.maxsize
        self.min_age = cfg.min_age if cfg.min_age is not None else 0
        assert self.min_age <= self.max_age, (self.min_age, self.max_age)

        self.obs_size = cfg.obs_size
        self.gap_size = cfg.gap_size
        self.pred_size = cfg.pred_size

    def build_cohort(self):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patients))
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustays))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admissions))

        icustays = icustays[icustays["first_careunit"] == icustays["last_careunit"]]
        icustays = icustays[icustays["los"] >= (self.obs_size + self.gap_size) / 24]
        icustays["intime"] = pd.to_datetime(icustays["intime"])
        icustays["outtime"] = pd.to_datetime(icustays["outtime"])

        patients_with_icustays = icustays.merge(patients, on="subject_id", how="left")
        patients_with_icustays["age"] = (
            patients_with_icustays["intime"].dt.year
            - patients_with_icustays["anchor_year"]
            + patients_with_icustays["anchor_age"]
        )

        patients_with_icustays["readmission"] = 1
        # the last icustay for each HADM_ID means that they have no icu readmission
        patients_with_icustays.loc[
            patients_with_icustays.groupby("hadm_id")["intime"].idxmax(), "readmission"
        ] = 0

        cohort = patients_with_icustays.merge(
            admissions[["hadm_id", "discharge_location", "deathtime", "dischtime"]],
            how="left",
            on="hadm_id",
        )
        cohort["deathtime"] = pd.to_datetime(cohort["deathtime"])
        cohort["dischtime"] = pd.to_datetime(cohort["dischtime"])

        self.cohort = cohort
        logger.info(
            "cohort has been built successfully. loaded {} cohorts.".format(
                len(self.cohort)
            )
        )

        return cohort

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
                    labeled_cohort["intime"]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.gap_size, unit="h")
                )
                <= labeled_cohort["deathtime"]
            )
            & (
                labeled_cohort["deathtime"]
                <= (
                    labeled_cohort["intime"]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.pred_size, unit="h")
                )
            )
        ).astype(int)

        labeled_cohort["in_icu_mortality"] = (
            labeled_cohort["deathtime"] > labeled_cohort["intime"]
        ) & (labeled_cohort["deathtime"] <= labeled_cohort["outtime"]).astype(int)

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
                    labeled_cohort["intime"]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.gap_size, unit="h")
                )
                <= labeled_cohort["dischtime"]
            )
            & (
                labeled_cohort["dischtime"]
                <= (
                    labeled_cohort["intime"]
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
                "in_icu_mortality",
                "intime",
                "dischtime",
                "discharge_location",
                "in_hospital_mortality",
            ]
        )

        # define diagnosis prediction task
        dx = pd.read_csv(os.path.join(self.data_dir, self.diagnoses))

        dx = self.icd10toicd9(dx)

        diagnoses_with_cohort = dx[dx["hadm_id"].isin(labeled_cohort["hadm_id"])]
        diagnoses_with_cohort = (
            diagnoses_with_cohort.groupby("hadm_id")["icd_code_converted"]
            .apply(list)
            .to_frame()
        )
        labeled_cohort = labeled_cohort.join(
            diagnoses_with_cohort, on="hadm_id", how="left"
        )

        ccs_dx = pd.read_csv(self.ccs_path)
        ccs_dx["'ICD-9-CM CODE'"] = (
            ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.replace(" ", "")
        )
        ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
        lvl1 = {
            x: y for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()
        }

        dx1_list = []
        # Some of patients(21) does not have dx codes
        labeled_cohort.dropna(subset=["icd_code_converted"], inplace=True)

        labeled_cohort["diagnosis"] = labeled_cohort["icd_code_converted"].map(
            lambda dxs: list(set([lvl1[dx] for dx in dxs if dx in lvl1]))
        )

        # XXX what does this line do?
        labeled_cohort = labeled_cohort.drop(columns=["icd_code_converted"])
        labeled_cohort.dropna(subset=["diagnosis"], inplace=True)

        self.labeled_cohort = labeled_cohort
        logger.info("Done preparing tasks given the cohort sets")

        return labeled_cohort

    def encode(self):
        encoded_cohort = self.labeled_cohort.rename(
            columns={"hadm_id": "id"}, inplace=False
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
