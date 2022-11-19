import os
import logging
import pandas as pd
import numpy as np
import glob
from ehrs import register_ehr, EHR
import pyspark.sql.functions as F
import pyspark.sql.types as T

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

        if self.creatinine or self.bilirubin or self.platelets:
            self.task_itemids = {
                "creatinine": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": ["labevent_id", "subject_id", "specimen_id", "storetime", "value", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "priority", "comments"],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [50912]
                },
                "bilirubin": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": ["labevent_id", "subject_id", "specimen_id", "storetime", "value", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "priority", "comments"],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [50885]
                },
                "platelets": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": ["labevent_id", "subject_id", "specimen_id", "storetime", "value", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "priority", "comments"],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [51265]
                },
                "dialysis": {
                    "ce": [226118, 227357, 225725, 226499, 224154, 225810, 227639, 225183, 227438, 224191, 225806, 225807, \
                        228004, 228005, 228006, 224144, 224145, 224149, 224150, 224151, 224152, 224153, 224404, 224406, 226457, 225959, \
                        224135, 224139, 224146, 225323, 225740, 225776, 225951, 225952, 225953, 225954, 225956, 225958, 225961, 225963, \
                        225965, 225976, 225977, 227124, 227290, 227638, 227640, 227753], 
                    "oe": [40386],
                    "ie": [227536, 227525],
                    "pe": [225441, 225802, 225803, 225805, 224270, 225809, 225955, 225436]
                }
            }

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

    def prepare_tasks(self, cohorts, spark, cached=False):
        if cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled.dx")
            if labeled_cohorts is not None:
                self.labeled_cohorts = labeled_cohorts
                return labeled_cohorts

        labeled_cohorts = super().prepare_tasks(cohorts, spark, cached)

        if self.diagnosis:
            logger.info(
                "Start labeling cohorts for diagnosis prediction."
            )

            # define diagnosis prediction task
            diagnoses = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))

            diagnoses = self.icd10toicd9(diagnoses)

            ccs_dx = pd.read_csv(self.ccs_path)
            ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
            ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
            lvl1 = {
                x: int(y)-1 for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()
            }

            diagnoses['diagnosis'] = diagnoses['icd_code_converted'].map(lvl1)

            diagnoses = diagnoses[(diagnoses['diagnosis'].notnull()) & (diagnoses['diagnosis']!=14)]
            diagnoses.loc[diagnoses['diagnosis']>=14, 'diagnosis'] -= 1
            diagnoses = diagnoses.groupby(self.hadm_key)['diagnosis'].agg(lambda x: list(set(x))).to_frame()

            labeled_cohorts = labeled_cohorts.merge(diagnoses, on=self.hadm_key, how='inner')

            # Some of patients(21) does not have dx codes
            labeled_cohorts.dropna(subset=["diagnosis"], inplace=True)

            self.labeled_cohorts = labeled_cohorts
            self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled.dx")

            logger.info("Done preparing diagnosis prediction for the given cohorts")

        if self.bilirubin:
            labeled_cohorts = self.clinical_task(labeled_cohorts, "bilirubin", spark)

        if self.platelets:
            labeled_cohorts = self.clinical_task(labeled_cohorts, "platelets", spark)

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


    def clinical_task(self, cohorts, task, spark):

        cohorts = spark.createDataFrame(cohorts)
        fname = self.task_itemids[task]["fname"]
        timestamp = self.task_itemids[task]["timestamp"]
        timeoffsetunit = self.task_itemids[task]["timeoffsetunit"]
        excludes = self.task_itemids[task]["exclude"]
        code = self.task_itemids[task]["code"][0]
        value = self.task_itemids[task]["value"][0]
        itemid = self.task_itemids[task]["itemid"][0]

        table = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
        table = table.drop(*excludes)
        table = table.filter(F.col(code) == itemid).filter(F.col(value).isNotNull())

        merge = cohorts.join(table, on=self.hadm_key, how="inner")
        if timeoffsetunit == "abs":
            merge = merge.withColumn(timestamp, F.to_timestamp(timestamp))
            merge = (
                merge.withColumn(
                    timestamp,
                    F.round((F.col(timestamp).cast("long") - F.col("INTIME").cast("long")) / 60)
                )
            )

        # Events within (obs_size + gap_size) - (obs_size + pred_size / outtime)
        merge = merge.filter(
            ((self.obs_size + self.gap_size) * 60) <= F.col(timestamp)).filter(
                ((self.obs_size + self.pred_size) * 60) >= F.col(timestamp)).filter(
                    F.col("OUTTIME") >= F.col(timestamp)
            )

        # Average value of events
        value_agg = merge.groupBy(self.icustay_key).agg(F.mean(value).alias("avg_value")) # TODO: mean/min/max?

        # Labeling
        if task == 'bilirubin':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.avg_value < 1.2, 0).when(
                    (value_agg.avg_value >= 1.2) & (value_agg.avg_value < 2.0), 1).when(
                        (value_agg.avg_value >= 2.0) & (value_agg.avg_value < 6.0), 2).when(
                            (value_agg.avg_value >= 6.0) & (value_agg.avg_value < 12.0), 3).when(
                                value_agg.avg_value >= 12.0, 4)
                )
        elif task == 'platelets':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.avg_value >= 150, 0).when(
                    (value_agg.avg_value >= 100) & (value_agg.avg_value < 150), 1).when(
                        (value_agg.avg_value >= 50) & (value_agg.avg_value < 100), 2).when(
                            (value_agg.avg_value >= 20) & (value_agg.avg_value < 50), 3).when(
                                value_agg.avg_value < 20, 4)
                )

        cohorts = cohorts.join(value_agg.select(self.icustay_key, task), on=self.icustay_key, how="left")
        cohorts = cohorts.na.fill(value=5, subset=[task])
        breakpoint()

        return cohorts
    
    
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