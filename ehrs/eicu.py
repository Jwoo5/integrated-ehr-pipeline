import os
import logging
import treelib
from collections import Counter
import pandas as pd
import glob

from ehrs import register_ehr, EHR

logger = logging.getLogger(__name__)

@register_ehr("eicu")
class eICU(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "eicu"

        if self.data_dir is None:
            self.data_dir = os.path.join(self.cache_dir, self.ehr_name)

            if not os.path.exists(self.data_dir):
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )
                self.download_ehr_from_url(
                    url="https://physionet.org/files/eicu-crd/2.0/",
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

        self._icustay_fname = "patient" + self.ext
        self._diagnosis_fname = "diagnosis" + self.ext

        self.tables = [
            {
                "fname": "lab" + self.ext,
                "timestamp": "labresultoffset",
                "timeoffsetunit": "min",
                "exclude": ["labid", "labresultrevisedoffset"],
            },
            {
                "fname": "medication" + self.ext,
                "timestamp": "drugstartoffset",
                "timeoffsetunit": "min",
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
                "fname": "infusionDrug" + self.ext,
                "timestamp": "infusionoffset",
                "timeoffsetunit": "min",
                "exclude": ["infusiondrugid"],
            },
        ]
        if cfg.use_more_tables:
            self.tables += [
            {
                "fname": "nurseCharting" + self.ext,
                "timestamp": "nursingchartoffset",
                "timeoffsetunit": "min",
                "exclude": ["nursingchartentryoffset", "nursingchartid"],
            },
                {
                "fname": "nurseCare" + self.ext,
                "timestamp": "nursecareoffset",
                "timeoffsetunit": "min",
                "exclude": ["nursecareentryoffset", "nursecareid"],
            },
            {
                "fname": "intakeOutput" + self.ext,
                "timestamp": "intakeoutputoffset",
                "timeoffsetunit": "min",
                "exclude": ["intakeoutputentryoffset", "intakeoutputid"],
            },
            {
                "fname": "microLab" + self.ext,
                "timestamp": "culturetakenoffset",
                "timeoffsetunit": "min",
                "exclude": ["microlabid"],
            },
            {
                "fname": "nurseAssessment" + self.ext,
                "timestamp": "nurseassessoffset",
                "timeoffsetunit": "min",
                "exclude": ["nurseassessentryoffset", "nurseassessid"],
            },
            {
                "fname": "treatment" + self.ext,
                "timestamp": "treatmentoffset",
                "timeoffsetunit": "min",
                "exclude": ["treatmentid", "activeupondischarge"],
            },
            {
                "fname": "vitalAperiodic" + self.ext,
                "timestamp": "observationoffset",
                "timeoffsetunit": "min",
                "exclude": ["vitalaperiodicid"],
            },
            {
                "fname": "vitalPeriodic" + self.ext,
                "timestamp": "observationoffset",
                "timeoffsetunit": "min",
                "exclude": ["vitalperiodicid"],
            },
        ]

        self._icustay_key = "patientunitstayid"
        self._hadm_key = "patienthealthsystemstayid"
        self._patient_key = "uniquepid"

        self._determine_first_icu = "unitvisitnumber"

    def build_cohorts(self, cached=False):
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustay_fname))

        icustays = self.make_compatible(icustays)
        self.icustays = icustays

        cohorts = super().build_cohorts(icustays, cached=cached)

        return cohorts

    def prepare_tasks(self, cohorts, spark, cached=False):
        if cohorts is None and cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled")
            if labeled_cohorts is not None:
                return labeled_cohorts

        labeled_cohorts = super().prepare_tasks(cohorts, spark, cached)

        if self.diagnosis:
            logger.info(
                "Start labeling cohorts for diagnosis prediction."
            )

            str2cat = self.make_dx_mapping()
            dx = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))
            dx = dx.merge(cohorts[[self.icustay_key, self.hadm_key]], on=self.icustay_key)
            dx["diagnosis"] = dx["diagnosisstring"].map(lambda x: str2cat.get(x, -1))
            # Ignore Rare Class(14)
            dx = dx[(dx["diagnosis"] != -1) & (dx["diagnosis"] != 14)]
            dx.loc[dx['diagnosis']>=14, "diagnosis"] -= 1
            dx = (
                dx.groupby(self.hadm_key)['diagnosis']
                .agg(lambda x: list(set(x)))
                .to_frame()
            )
            labeled_cohorts = labeled_cohorts.merge(dx, on=self.hadm_key, how="left")

        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        logger.info("Done preparing diagnosis prediction for the given cohorts")

        return labeled_cohorts

    def make_compatible(self, icustays):
        icustays.loc[:, "LOS"] = icustays["unitdischargeoffset"] / 60 / 24
        icustays.dropna(subset=["age"], inplace=True)
        icustays["AGE"] = icustays["age"].replace("> 89", 300).astype(int)

        # hacks for compatibility with other ehrs
        icustays["INTIME"] = 0
        icustays.rename(columns={"unitdischargeoffset": "OUTTIME"}, inplace=True)
        # DEATHTIME
        # icustays["DEATHTIME"] = np.nan
        # is_discharged_in_icu = icustays["unitdischargestatus"] == "Expired"
        # icustays.loc[is_discharged_in_icu, "DEATHTIME"] = (
        #     icustays.loc[is_discharged_in_icu, "OUTTIME"]
        # )
        # is_discharged_in_hos = (
        #     (icustays["unitdischargestatus"] != "Expired")
        #     & (icustays["hospitaldischargestatus"] == "Expired")
        # )
        # icustays.loc[is_discharged_in_hos, "DEATHTIME"] = (
        #     icustays.loc[is_discharged_in_hos, "OUTTIME"]
        # ) + 1

        icustays.rename(columns={"hospitaldischargeoffset": "DISCHTIME"}, inplace=True)

        icustays["IN_ICU_MORTALITY"] = icustays["unitdischargestatus"] == "Expired"

        icustays.rename(columns={
            "hospitaldischargelocation": "HOS_DISCHARGE_LOCATION"
        }, inplace=True)

        return icustays

    def make_dx_mapping(self):
        diagnosis = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))
        ccs_dx = pd.read_csv(self.ccs_path)
        gem = pd.read_csv(self.gem_path)

        diagnosis = diagnosis[["diagnosisstring", "icd9code"]]

        # 1 to 1 matching btw str and code
        # STR: diagnosisstring, CODE: icd9/10 code, CAT:category
        # 1 str -> multiple code -> one cat

        # 1. make str -> code dictonary
        str2code = diagnosis.dropna(subset=["icd9code"])
        str2code = str2code.groupby("diagnosisstring").first().reset_index()
        str2code["icd9code"] = str2code["icd9code"].str.split(",")
        str2code = str2code.explode("icd9code")
        str2code["icd9code"] = str2code["icd9code"].str.replace(".", "", regex=False)
        # str2code = dict(zip(notnull_dx["diagnosisstring"], notnull_dx["icd9code"]))
        # 이거 하면 dxstring duplicated 자동 제거됨 ->x

        ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
        ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1].astype(int) - 1
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
                    logger.warning(f"{k} has multiple categories{cat, str2cat[k]}")
                str2cat[k] = icd2cat[v]
            elif v in icd102icd9.keys():
                cat = icd2cat[icd102icd9[v]]
                if k in str2cat.keys() and str2cat[k] != cat:
                    logger.warning(f"{k} has multiple categories{cat, str2cat[k]}")
                str2cat[k] = icd2cat[icd102icd9[v]]

        # 4. If no available category by mapping(~25%), use diagnosisstring hierarchy

        # Make tree structure
        tree = treelib.Tree()
        tree.create_node("root", "root")
        for dx, cat in str2cat.items():
            dx = dx.split("|")
            if not tree.contains(dx[0]):
                tree.create_node(-1, dx[0], parent="root")
            for i in range(2, len(dx)):
                if not tree.contains("|".join(dx[:i])):
                    tree.create_node(-1, "|".join(dx[:i]), parent="|".join(dx[: i - 1]))
            if not tree.contains("|".join(dx)):
                tree.create_node(cat, "|".join(dx), parent="|".join(dx[:-1]))

        # Update non-leaf nodes with majority vote
        nid_list = list(tree.expand_tree(mode=treelib.Tree.DEPTH))
        nid_list.reverse()
        for nid in nid_list:
            if tree.get_node(nid).is_leaf():
                continue
            elif tree.get_node(nid).tag == -1:
                tree.get_node(nid).tag = Counter(
                    [child.tag for child in tree.children(nid)]
                ).most_common(1)[0][0]

        # Evaluate dxs without category
        unmatched_dxs = set(diagnosis["diagnosisstring"]) - set(str2cat.keys())
        for dx in unmatched_dxs:
            dx = dx.split("|")
            # Do not go to root level(can add noise)
            for i in range(len(dx) - 1, 1, -1):
                if tree.contains("|".join(dx[:i])):
                    str2cat["|".join(dx)] = tree.get_node("|".join(dx[:i])).tag
                    break

        return str2cat

    def infer_data_extension(self) -> str:
        if (len(glob.glob(os.path.join(self.data_dir, "*.csv.gz"))) == 31):
            ext = ".csv.gz"
        elif (len(glob.glob(os.path.join(self.data_dir, "*.csv"))) == 31):
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext