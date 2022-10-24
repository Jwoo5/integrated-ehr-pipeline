import os
import logging
import pandas as pd
import glob
from ehrs import register_ehr, EHR
from utils.utils import get_physionet_dataset, get_ccs, get_gem

logger = logging.getLogger(__name__)


@register_ehr("eicu")
class eICU(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        cache_dir = os.path.expanduser("~/.cache/ehr")
        physionet_file_path = "eicu-crd/2.0"

        self.data_dir = get_physionet_dataset(cfg, physionet_file_path, cache_dir)
        self.ccs_path = get_ccs(cfg, cache_dir)
        self.gem_path = get_gem(cfg, cache_dir)

        self.ext = ".csv"
        # self.ext = ".csv.gz"
        # if len(glob.glob(os.path.join(self.data_dir, "*" + self.ext))) != 33:
        #     self.ext = ".csv"
        #     if len(glob.glob(os.path.join(self.data_dir, "*" + self.ext))) != 33:
        #         raise AssertionError(
        #             "Provided data directory is not correct. Please check if --data is correct. "
        #             "--data: {}".format(self.data_dir)
        #         )
        self.icustays = f"patient" + self.ext
        self.diagnoses = f"diagnosis" + self.ext

        self.features = [
            {
                "fname": "lab" + self.ext,
                "type": "lab",
                "timestamp": "labresultoffset",
                "exclude": ["labid", "labresultrevisedoffset"],
            },
            {
                "fname": "medication" + self.ext,
                "type": "med",
                "timestamp": "drugstartoffset",
                "exclude": [
                    "drugorderoffset",
                    "drugstopoffset",
                    "medicationid",
                    "gtc",
                    "drughiclseqno",
                    "drugordercancelled",
                ],
            },
            {
                "fname": "infusiondrug" + self.ext,
                "type": "inf",
                "timestamp": "infusionoffset",
                "exclude": ["infusiondrugid"],
            },
        ]

        self.icustay_key = "patientunitstayid"
        self.icustay_end_key = "unitdischargeoffset"
        self.second_key = "patienthealthsystemstayid"

    def build_cohort(self):
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustays))
        icustays.loc[:, "los"] = icustays["unitdischargeoffset"] / 60 / 24
        icustays = icustays[icustays["los"] >= (self.obs_size + self.gap_size) / 24]

        icustays.dropna(subset=["age"], inplace=True)
        icustays["age"] = icustays["age"].replace("> 89", 89).astype(int)
        icustays = icustays[
            (self.min_age <= icustays["age"]) & (icustays["age"] <= self.max_age)
        ]

        icustays = self.readmission_label(icustays)

        self.cohort = icustays
        logger.info(
            "cohort has been built successfully. loaded {} cohorts.".format(
                len(self.cohort)
            )
        )
        icustays.to_pickle(os.path.join(self.dest, "mimiciii.cohorts"))

        return icustays

    def prepare_tasks(self):
        labeled_cohort = self.cohort.copy()

        labeled_cohort["los_3day"] = (self.cohort["los"] > 3).astype(int)
        labeled_cohort["los_7day"] = (self.cohort["los"] > 7).astype(int)

        labeled_cohort["mortality"] = (
            (labeled_cohort["unitdischargestatus"] == "Expired")
            & (
                labeled_cohort["unitdischargeoffset"]
                >= (self.obs_size + self.gap_size) * 60
            )
            & (
                labeled_cohort["unitdischargeoffset"]
                <= (self.obs_size + self.pred_size) * 60
            )
        ).astype(int)

        labeled_cohort["in_icu_mortality"] = (
            labeled_cohort["unitdischargestatus"] == "Expired"
        ).astype(int)
        labeled_cohort["in_hospital_mortality"] = (
            (labeled_cohort["unitdischargestatus"] != "Expired")
            & (labeled_cohort["hospitaldischargestatus"] == "Expired")
        ).astype(int)

        labeled_cohort["final_acuity"] = labeled_cohort["hospitaldischargelocation"]
        labeled_cohort.loc[
            labeled_cohort["in_icu_mortality"] == 1, "final_acuity"
        ] = "IN_ICU_MORTALITY"
        labeled_cohort.loc[
            labeled_cohort["in_hospital_mortality"] == 1, "final_acuity"
        ] = "IN_HOSPITAL_MORTALITY"

        logger.info("Fincal Acuity Categories")
        logger.info(labeled_cohort["final_acuity"].astype("category").cat.categories)
        labeled_cohort["final_acuity"] = (
            labeled_cohort["final_acuity"].astype("category").cat.codes
        )

        labeled_cohort["imminent_discharge"] = labeled_cohort[
            "hospitaldischargelocation"
        ]

        is_discharged = (
            (
                labeled_cohort["hospitaldischargeoffset"]
                >= (self.obs_size + self.gap_size) * 60
            )
            & (
                labeled_cohort["hospitaldischargeoffset"]
                <= (self.obs_size + self.pred_size) * 60
            )
        ).astype(bool)

        labeled_cohort.loc[is_discharged, "imminent_discharge"] = labeled_cohort[
            is_discharged
        ]["hospitaldischargelocation"]
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
                "in_hospital_mortality",
            ]
        )

        str2cat = self.make_dx_mapping(labeled_cohort)
        dx = pd.read_csv(os.path.join(self.data_dir, self.diagnoses))
        labeled_cohort = labeled_cohort.merge(
            dx[[self.icustay_key, "diagnosisstring"]], on=self.icustay_key, how="left"
        )
        # label should be 0-base
        labeled_cohort["diagnosis"] = labeled_cohort["diagnosisstring"].map(
            lambda x: int(str2cat.get(x, 0)) - 1
        )
        labeled_cohort = labeled_cohort[labeled_cohort["diagnosis"] != -1]
        labeled_cohort.drop(columns=["diagnosisstring"], inplace=True)

        return labeled_cohort

    def make_dx_mapping(self, cohort):
        diagnosis = pd.read_csv(os.path.join(self.data_dir, self.diagnoses))
        ccs_dx = pd.read_csv(self.ccs_path)
        gem = pd.read_csv(self.gem_path)

        diagnosis = diagnosis[
            diagnosis[self.icustay_key].isin(cohort[self.icustay_key])
        ][["diagnosisstring", "icd9code"]]

        # 1 to 1 matching btw str and code
        # STR: diagnosisstring, CODE: icd9/10 code, CAT:category
        # 1 str -> multiple code -> one cat

        # 1. make str -> code dictonary
        str2code = diagnosis.dropna(subset=["icd9code"])
        str2code = str2code.groupby("diagnosisstring").first().reset_index()
        str2code["icd9code"] = str2code["icd9code"].str.split(",")
        str2code = str2code.explode("icd9code")
        str2code["icd9code"] = str2code["icd9code"].str.replace(".", "")
        # str2code = dict(zip(notnull_dx["diagnosisstring"], notnull_dx["icd9code"]))
        # 이거 하면 dxstring duplicated 자동 제거됨 ->x

        ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
        ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
        icd2cat = dict(zip(ccs_dx["'ICD-9-CM CODE'"], ccs_dx["'CCS LVL 1'"]))

        # 2. if code is not icd9, convert it to icd9
        str2code_icd10 = str2code[str2code["icd9code"].isin(icd2cat.keys())]

        map_cms = dict(zip(gem["icd10cm"], gem["icd9cm"]))
        map_manual = dict.fromkeys(
            set(str2code_icd10["icd9code"]) - set(gem["icd10cm"]), "NaN"
        )

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
        icd102icd9 = {**map_cms, **map_manual}

        # 3. Convert Available Strings to category
        str2cat = {}
        for _, row in str2code.iterrows():
            k, v = row
            if v in icd2cat.keys():
                cat = icd2cat[v]
                if k in str2cat.keys() and str2cat[k] != cat:
                    print(f"Warning: {k} has multiple categories{cat, str2cat[k]}")
                str2cat[k] = icd2cat[v]
            elif v in icd102icd9.keys():
                cat = icd2cat[icd102icd9[v]]
                if k in str2cat.keys() and str2cat[k] != cat:
                    print(f"Warning: {k} has multiple categories{cat, str2cat[k]}")
                str2cat[k] = icd2cat[icd102icd9[v]]
        print(len(str2cat))

        # 4. If no available category by mapping(~25%), use diagnosisstring hierarchy
        unmatched_dxs = set(diagnosis["diagnosisstring"]) - set(str2cat.keys())
        for dx in unmatched_dxs:
            for i in range(len(dx.split("|")), 0, -1):
                _dx = "|".join(dx.split("|")[:i])
                if _dx in str2cat.keys():
                    str2cat[dx] = str2cat[_dx]
                    break

        print(len(str2cat))

        return str2cat
