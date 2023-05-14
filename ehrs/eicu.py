import glob
import logging
import os
from collections import Counter

import pandas as pd
import pyspark.sql.functions as F
import treelib

from ehrs import EHR, register_ehr

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
                    url="https://physionet.org/files/eicu-crd/2.0/", dest=self.data_dir
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

        if self.creatinine or self.bilirubin or self.platelets or self.wbc or self.hb or self.bicarbonate or self.sodium or self.antibiotics:
            self.task_itemids = {
                "creatinine": {
                    "fname": "lab" + self.ext,
                    "timestamp": "labresultoffset",
                    "timeoffsetunit": "min",
                    "exclude": [
                        "labtypeid",
                        "labresulttext",
                        "labmeasurenamesystem",
                        "labmeasurenameinterface",
                        "labresultrevisedoffset",
                    ],
                    "code": ["labname"],
                    "value": ["labresult"],
                    "itemid": ["creatinine"],
                },
                "bilirubin": {
                    "fname": "lab" + self.ext,
                    "timestamp": "labresultoffset",
                    "timeoffsetunit": "min",
                    "exclude": [
                        "labtypeid",
                        "labresulttext",
                        "labmeasurenamesystem",
                        "labmeasurenameinterface",
                        "labresultrevisedoffset",
                    ],
                    "code": ["labname"],
                    "value": ["labresult"],
                    "itemid": ["total bilirubin"],
                },
                "platelets": {
                    "fname": "lab" + self.ext,
                    "timestamp": "labresultoffset",
                    "timeoffsetunit": "min",
                    "exclude": [
                        "labtypeid",
                        "labresulttext",
                        "labmeasurenamesystem",
                        "labmeasurenameinterface",
                        "labresultrevisedoffset",
                    ],
                    "code": ["labname"],
                    "value": ["labresult"],
                    "itemid": ["platelets x 1000"],
                },
                "wbc": {
                    "fname": "lab" + self.ext,
                    "timestamp": "labresultoffset",
                    "timeoffsetunit": "min",
                    "exclude": [
                        "labtypeid",
                        "labresulttext",
                        "labmeasurenamesystem",
                        "labmeasurenameinterface",
                        "labresultrevisedoffset",
                    ],
                    "code": ["labname"],
                    "value": ["labresult"],
                    "itemid": ["WBC x 1000"],
                },
                "dialysis": {
                    "fname": "intakeOutput" + self.ext,
                    "timestamp": "intakeoutputoffset",
                    "timeoffsetunit": "min",
                    "exclude": [
                        "intakeoutputid",
                        "intaketotal",
                        "outputtotal",
                        "nettotal",
                        "intakeoutputentryoffset",
                    ],
                    "code": ["dialysistotal"],
                    "value": [],
                    "itemid": [],
                },
                "hb": {
                    "fname": "lab" + self.ext,
                    "timestamp": "labresultoffset",
                    "timeoffsetunit": "min",
                    "exclude": [
                        "labtypeid",
                        "labresulttext",
                        "labmeasurenamesystem",
                        "labmeasurenameinterface",
                        "labresultrevisedoffset",
                    ],
                    "code": ["labname"],
                    "value": ["labresult"],
                    "itemid": ["Hgb"],
                },
                "bicarbonate": {
                    "fname": "lab" + self.ext,
                    "timestamp": "labresultoffset",
                    "timeoffsetunit": "min",
                    "exclude": [
                        "labtypeid",
                        "labresulttext",
                        "labmeasurenamesystem",
                        "labmeasurenameinterface",
                        "labresultrevisedoffset",
                    ],
                    "code": ["labname"],
                    "value": ["labresult"],
                    "itemid": ["bicarbonate"],
                },
                "sodium": {
                    "fname": "lab" + self.ext,
                    "timestamp": "labresultoffset",
                    "timeoffsetunit": "min",
                    "exclude": [
                        "labtypeid",
                        "labresulttext",
                        "labmeasurenamesystem",
                        "labmeasurenameinterface",
                        "labresultrevisedoffset",
                    ],
                    "code": ["labname"],
                    "value": ["labresult"],
                    "itemid": ["sodium"],
                },
                "antibiotics": {
                    "fname": "medication" + self.ext,
                    "timestamp": "drugstartoffset",
                    "timeoffsetunit": "min",
                    "exclude": [
                        "medicationid",
                        "drugorderoffset",
                        "drugivadmixture",
                        "drugordercancelled",
                        "drughiclseqno",
                        "dosage",
                        "routeadmin",
                        "frequency",
                        "loadingdose",
                        "prn",
                        "drugstopoffset",
                        "gtc",
                    ],
                    "code": ["drugname"],
                    "itemid": [
                        'ancef',
                        'azithromycin',
                        'bacitracin',
                        'cefazolin',
                        'cefepime',
                        'ceftriaxone',
                        'cipro',
                        'ciprofloxacin',
                        'clindamycin',
                        'flagyl',
                        'levaquin',
                        'levofloxacin',
                        'maxipime',
                        'meropenem',
                        'merrem',
                        'metronidazole',
                        'mupirocin',
                        'nafcillin',
                        'nystatin',
                        'ofloxacin',
                        'piperacillin',
                        'piperacillin-tazobactam',
                        'rocephin',
                        'tazobactam',
                        'vancocin',
                        'vancomycin',
                        'zosyn',
                        ]
                },
            }

        self.disch_map_dict = {
            "Home": "Home",
            "IN_ICU_MORTALITY": "IN_ICU_MORTALITY",
            "Nursing Home": "Other",
            "Other": "Other",
            "Other External": "Other",
            "Other Hospital": "Other",
            "Rehabilitation": "Rehabilitation",
            "Skilled Nursing Facility": "Skilled Nursing Facility",
            "Death": "Death",
        }

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
            logger.info("Start labeling cohorts for diagnosis prediction.")

            str2cat = self.make_dx_mapping()
            dx = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))
            dx = dx.merge(
                cohorts[[self.icustay_key, self.hadm_key]], on=self.icustay_key
            )
            dx["diagnosis"] = dx["diagnosisstring"].map(lambda x: str2cat.get(x, -1))
            # Ignore Rare Class(14)
            dx = dx[(dx["diagnosis"] != -1) & (dx["diagnosis"] != 14)]
            dx.loc[dx["diagnosis"] >= 14, "diagnosis"] -= 1
            dx = (
                dx.groupby(self.hadm_key)["diagnosis"]
                .agg(lambda x: list(set(x)))
                .to_frame()
            )

            labeled_cohorts = labeled_cohorts.merge(dx, on=self.hadm_key, how="left")
            labeled_cohorts["diagnosis"] = labeled_cohorts["diagnosis"].apply(
                lambda x: [] if type(x) != list else x
            )
            # NaN case in diagnosis -> []

        logger.info("Done preparing diagnosis prediction for the given cohorts")

        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        if self.bilirubin or self.platelets or self.creatinine or self.wbc or self.hb or self.bicarbonate or self.sodium or self.antibiotics:
            logger.info("Start labeling cohorts for clinical task prediction.")

            labeled_cohorts = spark.createDataFrame(labeled_cohorts)

            if self.bilirubin:
                labeled_cohorts = self.clinical_task(
                    labeled_cohorts, "bilirubin", spark
                )

            if self.platelets:
                labeled_cohorts = self.clinical_task(
                    labeled_cohorts, "platelets", spark
                )

            if self.creatinine:
                labeled_cohorts = self.clinical_task(
                    labeled_cohorts, "creatinine", spark
                )

            if self.wbc:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "wbc", spark)

            if self.hb:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "hb", spark)

            if self.bicarbonate:
                labeled_cohorts = self.clinical_task(
                    labeled_cohorts, "bicarbonate", spark
                )
            
            if self.sodium:
                labeled_cohorts = self.clinical_task(
                    labeled_cohorts, "sodium", spark
                )
            
            if self.antibiotics:
                labeled_cohorts = self.clinical_task(
                    labeled_cohorts, "antibiotics", spark
                )

            # self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled.clinical_tasks")

            logger.info("Done preparing clinical task prediction for the given cohorts")

        if not isinstance(labeled_cohorts, pd.DataFrame):
            labeled_cohorts = labeled_cohorts.toPandas()

        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")
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
        icustays["hospitaldischargelocation"] = icustays[
            "hospitaldischargelocation"
        ].map(self.disch_map_dict)
        icustays.rename(
            columns={"hospitaldischargelocation": "HOS_DISCHARGE_LOCATION"},
            inplace=True,
        )

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

    def clinical_task(self, cohorts, task, spark):
        fname = self.task_itemids[task]["fname"]
        timestamp = self.task_itemids[task]["timestamp"]
        timeoffsetunit = self.task_itemids[task]["timeoffsetunit"]
        excludes = self.task_itemids[task]["exclude"]
        code = self.task_itemids[task]["code"][0]
        if "value" in self.task_itemids[task].keys():
            value = self.task_itemids[task]["value"][0]
        itemid = self.task_itemids[task]["itemid"]

        table = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
        table = table.drop(*excludes)
        
        if "value" in self.task_itemids[task].keys():
            table = table.filter(F.col(code).isin(itemid)).filter(F.col(value).isNotNull())
        else:
            # Have to match regex
            table = (
                        table.dropna(subset=[code])
                        .withColumn(code, (F.regexp_extract(F.lower(code), '('+"|".join(itemid)+')', 0)))
                        .filter(f"{code} != ''")
                    )

        merge = cohorts.join(table, on=self.icustay_key, how="inner")

        if task == "creatinine":
            patient = spark.read.csv(
                os.path.join(self.data_dir, self._icustay_fname), header=True
            )
            patient = patient.select(
                *[self.patient_key, self.icustay_key, self._hadm_key]
            )  # icuunit intime
            multi_hosp = (
                patient.groupBy(self.patient_key)
                .agg(F.count(self._hadm_key).alias("count"))
                .filter(F.col("count") > 1)
                .select(self.patient_key)
            )
            # multiple hosp

            dialysis_tables = self.task_itemids["dialysis"][
                "fname"
            ]  # Only treatment for dialysis
            dialysis_code = self.task_itemids["dialysis"]["code"][0]
            excludes = self.task_itemids["dialysis"]["exclude"]

            io = spark.read.csv(
                os.path.join(self.data_dir, dialysis_tables), header=True
            )
            io = io.drop(*excludes)

            io_dialysis = io.filter(F.col(dialysis_code) != 0)
            io_dialysis = io_dialysis.join(patient, on=self.icustay_key, how="left")

            dialysis_multihosp = io_dialysis.join(
                multi_hosp, on=self.patient_key, how="leftsemi"
            ).select(self.patient_key)

            io_dialysis = io_dialysis.drop(self.patient_key)

            def dialysis_time(table, timecolumn):
                return table.withColumn("_DIALYSIS_TIME", F.col(timecolumn)).select(
                    self.icustay_key, "_DIALYSIS_TIME"
                )

            io_dialysis = dialysis_time(
                io_dialysis, self.task_itemids["dialysis"]["timestamp"]
            )
            io_dialysis = io_dialysis.groupBy(self.icustay_key).agg(
                F.min("_DIALYSIS_TIME").alias("_DIALYSIS_TIME")
            )
            io_dialysis = io_dialysis.select([self.icustay_key, "_DIALYSIS_TIME"])
            merge = merge.join(io_dialysis, on=self.icustay_key, how="left")
            merge = merge.filter(
                F.isnull("_DIALYSIS_TIME")
                | (F.col("_DIALYSIS_TIME") > F.col(timestamp))
            )
            merge = merge.drop("_DIALYSIS_TIME")

        # For Creatinine task, eliminate icus if patient went through dialysis treatment before (obs_size + pred_size) timestamp

        # Cohort with events within (obs_size + gap_size) - (obs_size + pred_size)
        if self.rolling_from_last:
            merge = merge.filter(
                F.col(timestamp)
                <= F.col("OUTTIME") + self.pred_size * 60 - self.gap_size * 60
            ).filter(F.col(timestamp) >= F.col("OUTTIME"))

        else:
            merge = merge.filter(
                ((self.obs_size + self.gap_size) * 60) <= F.col(timestamp)
            ).filter(((self.obs_size + self.pred_size) * 60) >= F.col(timestamp))

        # Average value of events
        if "value" in self.task_itemids[task].keys():
            value_agg = merge.groupBy(self.icustay_key).agg(
                F.mean(value).alias("avg_value")
            )  # TODO: mean/min/max?
        else:
            value_agg = merge.groupBy(self.icustay_key).agg(
                    F.count(code).alias("event_count")
                ).fillna(0, subset=["event_count"])
            value_agg = (cohorts.select(self.icustay_key)
                         .join(value_agg.select(self.icustay_key, "event_count"), on=self.icustay_key, how="left")
                         .fillna(0, subset=["event_count"]))
        # Labeling
        if task == "bilirubin":
            value_agg = value_agg.withColumn(
                task,
                F.when(value_agg.avg_value < 1.2, 0)
                .when((value_agg.avg_value >= 1.2) & (value_agg.avg_value < 2.0), 1)
                .when((value_agg.avg_value >= 2.0) & (value_agg.avg_value < 6.0), 2)
                .when((value_agg.avg_value >= 6.0) & (value_agg.avg_value < 12.0), 3)
                .when(value_agg.avg_value >= 12.0, 4),
            )
        elif task == "platelets":
            value_agg = value_agg.withColumn(
                task,
                F.when(value_agg.avg_value >= 150, 0)
                .when((value_agg.avg_value >= 100) & (value_agg.avg_value < 150), 1)
                .when((value_agg.avg_value >= 50) & (value_agg.avg_value < 100), 2)
                .when((value_agg.avg_value >= 20) & (value_agg.avg_value < 50), 3)
                .when(value_agg.avg_value < 20, 4),
            )

        elif task == "creatinine":
            value_agg = value_agg.join(
                patient.select([self.patient_key, self.icustay_key]),
                on=self.icustay_key,
                how="left",
            )
            value_agg = value_agg.withColumn(
                task,
                F.when(value_agg.avg_value < 1.2, 0)
                .when((value_agg.avg_value >= 1.2) & (value_agg.avg_value < 2.0), 1)
                .when((value_agg.avg_value >= 2.0) & (value_agg.avg_value < 3.5), 2)
                .when((value_agg.avg_value >= 3.5) & (value_agg.avg_value < 5), 3)
                .when(value_agg.avg_value >= 5, 4),
            )
            value_agg = value_agg.join(
                dialysis_multihosp, on=self.patient_key, how="leftanti"
            )

            value_agg = value_agg.drop(self.patient_key)

        elif task == "wbc":
            value_agg = value_agg.withColumn(
                task,
                F.when(value_agg.avg_value < 4, 0)
                .when((value_agg.avg_value >= 4) & (value_agg.avg_value <= 12), 1)
                .when((value_agg.avg_value > 12), 2),
            )

        elif task == 'hb':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.avg_value < 8, 0).when(
                    (value_agg.avg_value >= 8) & (value_agg.avg_value < 10), 1).when(
                        (value_agg.avg_value >= 10) & (value_agg.avg_value < 12), 2).when(
                            (value_agg.avg_value >= 12), 3)
                )

        elif task == 'bicarbonate':
            value_agg = value_agg.withColumn(task,
                F.when((value_agg.avg_value < 22), 0).when(
                        (value_agg.avg_value >= 22) & (value_agg.avg_value < 29), 1).when(
                            (value_agg.avg_value >= 29), 2)
            )

        elif task == 'sodium':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.avg_value < 135, 0).when(
                    (value_agg.avg_value >= 135) & (value_agg.avg_value < 145), 1).when(
                        (value_agg.avg_value >= 145), 2)
            )

        elif task == 'antibiotics':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.event_count < 1, 0).when(
                    (value_agg.event_count >= 1), 1)
            )

        cohorts = cohorts.join(
            value_agg.select(self.icustay_key, task), on=self.icustay_key, how="left"
        )

        return cohorts

    def infer_data_extension(self) -> str:
        if len(glob.glob(os.path.join(self.data_dir, "*.csv.gz"))) == 31:
            ext = ".csv.gz"
        elif len(glob.glob(os.path.join(self.data_dir, "*.csv"))) == 31:
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext
