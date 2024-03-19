import glob
import logging
import os
import tarfile
from functools import reduce

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from ehrs import EHR, register_ehr

logger = logging.getLogger(__name__)


@register_ehr("hirid")
class HIRID(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "hirid"

        if self.data_dir is None:
            self.data_dir = os.path.join(self.cache_dir, self.ehr_name)

            if not os.path.exists(self.data_dir):
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )
                self.download_ehr_from_url(
                    url="https://physionet.org/files/hirid/1.1.1/", dest=self.data_dir
                )

        logger.info("Data directory is set to {}".format(self.data_dir))

        self.ext = "/csv"
        self._diagnosis_fname = "observation_tables" + self.ext

        required_files = [
            "observation_tables",
            "pharma_records",
            "general_table.csv",
            "hirid_variable_reference.csv",
        ]
        if all(os.path.exists(os.path.join(self.cache_dir, f)) for f in required_files):
            logger.info("Data is already extracted.")
        else:
            with tarfile.open(
                os.path.join(self.data_dir, "reference_data.tar.gz"), "r:gz"
            ) as tar:
                tar.extract("general_table.csv", self.cache_dir)
                tar.extract("hirid_variable_reference.csv", self.cache_dir)

            # Reading hirid file directly using spark cause parsing error.
            # Therefore, unzip in cache directory and change the data_dir
            with tarfile.open(
                os.path.join(
                    self.data_dir, "raw_stage", "observation_tables_csv.tar.gz"
                ),
                "r:gz",
            ) as tar:
                tar.extractall(self.cache_dir)

            with tarfile.open(
                os.path.join(self.data_dir, "raw_stage", "pharma_records_csv.tar.gz"),
                "r:gz",
            ) as tar:
                tar.extractall(self.cache_dir)

            logger.info("Data is extracted to {}".format(self.cache_dir))

        self._icustay_fname = os.path.join(self.cache_dir, "general_table.csv")
        self._ref_fname = os.path.join(self.cache_dir, "hirid_variable_reference.csv")
        self.data_dir = os.path.join(self.cache_dir)

        self.tables = [
            {
                "fname": "observation_tables" + self.ext,
                "timestamp": "datetime",
                "timeoffsetunit": "abs",
                "exclude": ["status", "type", "entertime"],
                "code": ["variableid"],
                "desc": [self._ref_fname],
                "desc_code_col": ["ID"],
                "desc_key": [["Variable Name", "Unit", "Additional information"]],
                "desc_filter_col": ["Source Table"],
                "desc_filter_val": ["Observation"],
                "rename_map": [{"Variable Name": "variableid"}],
            },
            {
                "fname": "pharma_records" + self.ext,
                "timestamp": "givenat",
                "timeoffsetunit": "abs",
                "exclude": [
                    "enteredentryat",
                    "infusionid",
                    "typeid",
                    "subtypeid",
                    "recordstatus",
                ],
                "code": ["pharmaid"],
                "desc": [self._ref_fname],
                "desc_code_col": ["ID"],
                "desc_key": [["Variable Name", "Unit", "Additional information"]],
                "desc_filter_col": ["Source Table"],
                "desc_filter_val": ["Pharma"],
                "rename_map": [{"Variable Name": "variableid"}],
            },
        ]

        if (
            self.creatinine
            or self.platelets
            or self.wbc
            or self.hb
            or self.bicarbonate
            or self.sodium
        ):
            self.task_itemids = {
                "creatinine": {
                    "fname": "observation_tables" + self.ext,
                    "timestamp": "datetime",
                    "timeoffsetunit": "abs",
                    "code": ["variableid"],
                    "value": ["value"],
                    "itemid": [20000600],  # umol/l
                },
                "platelets": {
                    "fname": "observation_tables" + self.ext,
                    "timestamp": "datetime",
                    "timeoffsetunit": "abs",
                    "code": ["variableid"],
                    "value": ["value"],
                    "itemid": [20000110],  # G/l
                },
                "wbc": {
                    "fname": "observation_tables" + self.ext,
                    "timestamp": "datetime",
                    "timeoffsetunit": "abs",
                    "code": ["variableid"],
                    "value": ["value"],
                    "itemid": [20000700],  # G/l
                },
                "dialysis": {
                    "tables": {
                        "observation_tables": {
                            "fname": "observation_tables" + self.ext,
                            "timestamp": "datetime",
                            "timeoffsetunit": "abs",
                            "code": ["variableid"],
                            "value": ["value"],
                            "itemid": [10002508],  # IFF value==1.0
                        }
                    }
                },
                "hb": {
                    "fname": "observation_tables" + self.ext,
                    "timestamp": "datetime",
                    "timeoffsetunit": "abs",
                    "code": ["variableid"],
                    "value": ["value"],
                    "itemid": [20000900, 24000836],
                },
                "bicarbonate": {
                    "fname": "observation_tables" + self.ext,
                    "timestamp": "datetime",
                    "timeoffsetunit": "abs",
                    "code": ["variableid"],
                    "value": ["value"],
                    "itemid": [20004200],
                },
                "sodium": {
                    "fname": "observation_tables" + self.ext,
                    "timestamp": "datetime",
                    "timeoffsetunit": "abs",
                    "code": ["variableid"],
                    "value": ["value"],
                    "itemid": [20000400, 24000519, 24000658, 24000835, 24000866],
                },
            }

        if not cfg.use_more_tables:
            raise NotImplementedError()

        if cfg.use_ed:
            raise NotImplementedError()

        self._icustay_key = "patientid"
        self._hadm_key = None
        self._patient_key = None

        self._determine_first_icu = None

        self.mortality = None
        self.los = None

    def prepare_tasks(self, cohorts, spark, cached=False):
        labeled_cohorts = super().prepare_tasks(cohorts, spark, cached)
        if cached:
            return labeled_cohorts

        if self.diagnosis:
            logger.info("Start labeling cohorts for diagnosis prediction.")

            diagnoses = spark.read.csv(
                os.path.join(self.data_dir, self._diagnosis_fname), header=True
            ).select("variableid", "value", self.icustay_key)
            diagnoses = diagnoses.filter(
                diagnoses["variableid"].isin([9990002, 9990004])
            )
            diagnoses = diagnoses.withColumn(
                "value", F.col("value").cast("float").cast("int")
            ).withColumn(self.icustay_key, F.col(self.icustay_key).cast("int"))
            diagnoses = diagnoses.toPandas()
            # # 9990002: APACHE 2, 9990004: APACHE 4, each has non-overlapping value range

            def get_diagnosis_category(x):
                return {
                    98: 0,  # Cardiovascular
                    190: 0,  # Cardiovascular
                    99: 1,  # Pulmonary
                    191: 1,  # Pulmonary
                    100: 2,  # Gastointestinal
                    192: 2,  # Gastointestinal
                    101: 3,  # Neurological
                    193: 3,  # Neurological
                    102: 4,  # Sepsis + Intoxication
                    197: 5,  # Urogenital
                    103: 6,  # Trauma
                    194: 6,  # Trauma
                    104: 7,  # Metabolic/Endocrinology
                    195: 7,  # Metabolic/Endocrinology
                    105: 8,  # Hematology
                    196: 8,  # Hematology
                    106: 9,  # Other
                    198: 9,  # Other
                    107: 10,  # Surgical Cardiovascular
                    199: 10,  # Surgical Cardiovascular
                    108: 11,  # Surgical Respiratory
                    200: 11,  # Surgical Respiratory
                    109: 12,  # Surgical Gastrointestinal
                    201: 12,  # Surgical Gastrointestinal
                    110: 13,  # Surgical Neurological
                    202: 13,  # Surgical Neurological
                    111: 14,  # Surgical Trauma
                    203: 14,  # Surgical Trauma
                    112: 15,  # Surgical Urogenital
                    204: 15,  # Surgical Urogenital
                    113: 16,  # Surgical Gynecology + Surgeical Orthopedics + Surgical Others
                    205: 16,  # Surgical Gynecology + Surgeical Orthopedics + Surgical Others
                    114: 16,  # Surgical Gynecology + Surgeical Orthopedics + Surgical Others
                    206: 4,  # Sepsis + Intoxication
                }[x]

            diagnoses["diagnosis"] = diagnoses["value"].map(get_diagnosis_category)
            diagnoses = (
                diagnoses.groupby(self.icustay_key)["diagnosis"]
                .agg(lambda x: list(set(x)))
                .to_frame()
            )
            diagnoses.reset_index(inplace=True)
            diagnoses[self.icustay_key] = diagnoses[self.icustay_key].astype(int)
            labeled_cohorts = labeled_cohorts.merge(
                diagnoses, on=self.icustay_key, how="left"  # Use all
            )
            labeled_cohorts["diagnosis"] = labeled_cohorts["diagnosis"].map(
                lambda x: x if isinstance(x, list) else []
            )
            logger.info("Done preparing diagnosis prediction for the given cohorts")

            self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        logger.info("Start labeling cohorts for clinical task prediction.")
        labeled_cohorts = spark.createDataFrame(labeled_cohorts)
        for clinical_task in [
            "creatinine",
            "platelets",
            "wbc",
            "hb",
            "bicarbonate",
            "sodium",
        ]:
            horizons = self.__getattribute__(clinical_task)
            if horizons:
                labeled_cohorts = self.clinical_task(
                    labeled_cohorts,
                    clinical_task,
                    horizons,
                    spark,
                )

        logger.info("Done preparing clinical task prediction for the given cohorts")
        labeled_cohorts = labeled_cohorts.toPandas()
        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")
        return labeled_cohorts

    def make_compatible(self, icustays, spark):
        # prepare icustays according to the appropriate format
        icustays.rename(columns={"age": "AGE", "admissiontime": "INTIME"}, inplace=True)
        icustays["ADMITTIME"] = pd.to_datetime(icustays["INTIME"])
        icustays["INTIME"] = 0

        icustays["DEATHTIME"] = icustays["discharge_status"].map(
            lambda x: True if x == "death" else np.nan,
        )
        # Calculate LOS using last timestamp
        # Load all files & Calculate LOS
        dfs = []
        for table in self.tables:
            df = spark.read.csv(
                os.path.join(self.data_dir, table["fname"]), header=True
            )
            df = df.select(table["timestamp"], self.icustay_key)
            df = df.withColumn("timestamp", F.to_timestamp(table["timestamp"])).drop(
                table["timestamp"]
            )
            dfs.append(df)
        merged = reduce(lambda x, y: x.union(y), dfs)
        merged = merged.groupBy(self.icustay_key).agg(
            F.max("timestamp").alias("max_time")
        )
        last_stamps = merged.toPandas()
        last_stamps[self.icustay_key] = last_stamps[self.icustay_key].astype(int)
        icustays = icustays.merge(last_stamps, on=self.icustay_key, how="inner")
        icustays["LOS"] = (
            (icustays["max_time"] - icustays["ADMITTIME"]).dt.total_seconds()
            / 60
            / 60
            / 24  # days
        )
        return icustays

    def clinical_task(self, cohorts, task, horizons, spark):
        fname = self.task_itemids[task]["fname"]
        timestamp = self.task_itemids[task]["timestamp"]
        code = self.task_itemids[task]["code"][0]
        value = self.task_itemids[task]["value"][0]
        itemid = self.task_itemids[task]["itemid"]

        table = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
        table = table.select(self.icustay_key, code, value, timestamp)
        table.cache()
        table = table.filter(F.col(code).isin(itemid)).filter(F.col(value).isNotNull())
        table = table.withColumn(timestamp, F.to_timestamp(timestamp))

        merge = cohorts.join(table, on=self.icustay_key, how="inner")

        if task == "creatinine":
            dialysis_tables = []
            for dialysis_dict in self.task_itemids["dialysis"]["tables"].values():
                dialysis_table = spark.read.csv(
                    os.path.join(self.data_dir, dialysis_dict["fname"]), header=True
                )
                dialysis_table = dialysis_table.filter(
                    F.col(code).isin(dialysis_dict["itemid"])
                )
                dialysis_table = dialysis_table.filter(F.col("value") == 1.0)
                dialysis_table = dialysis_table.withColumn(
                    "_DIALYSIS_TIME",
                    F.to_timestamp(dialysis_dict["timestamp"]),
                )
                dialysis_tables.append(
                    dialysis_table.select(self.icustay_key, "_DIALYSIS_TIME")
                )
            dialysis = reduce(lambda x, y: x.union(y), dialysis_tables)

            dialysis = dialysis.groupby(self.icustay_key).agg(
                F.min("_DIALYSIS_TIME").alias("_DIALYSIS_TIME")
            )
            merge = merge.join(dialysis, on=self.icustay_key, how="left")
            # Only leave events with no dialysis / before first dialysis
            merge = merge.filter(
                F.isnull("_DIALYSIS_TIME")
                | (F.col("_DIALYSIS_TIME") > F.col(timestamp))
            )
            merge = merge.drop("_DIALYSIS_TIME")

        merge = merge.withColumn(
            timestamp,
            F.round(
                (F.col(timestamp).cast("long") - F.col("ADMITTIME").cast("long")) / 60
            ),
        )

        merge = merge.filter(F.col(timestamp) >= F.col("INTIME") + self.pred_size * 60)
        window = Window.partitionBy(self.icustay_key).orderBy(F.desc(timestamp))

        merge.cache()

        for horizon in horizons:
            horizon_merge = merge.filter(
                F.col(timestamp)
                < F.col("INTIME") + (self.pred_size + horizon * 24) * 60
            ).filter(
                F.col(timestamp)
                >= F.col("INTIME") + (self.pred_size + (horizon - 1) * 24) * 60
            )
            horizon_agg = (
                horizon_merge.withColumn("row", F.row_number().over(window))
                .filter(F.col("row") == 1)
                .drop("row")
                .withColumnRenamed(value, "value")
            )

            task_name = task + "_" + str(horizon)
            # Labeling
            if task == "platelets":  # K/ul=G/l
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value >= 150, 0)
                    .when((horizon_agg.value >= 100) & (horizon_agg.value < 150), 1)
                    .when((horizon_agg.value >= 50) & (horizon_agg.value < 100), 2)
                    .when((horizon_agg.value >= 20) & (horizon_agg.value < 50), 3)
                    .when(horizon_agg.value < 20, 4),
                )

            elif task == "creatinine":  # mg/dL -> umol/L
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 1.2 * 88.42, 0)
                    .when(
                        (horizon_agg.value >= 1.2 * 88.42)
                        & (horizon_agg.value < 2.0 * 88.42),
                        1,
                    )
                    .when(
                        (horizon_agg.value >= 2.0 * 88.42)
                        & (horizon_agg.value < 3.5 * 88.42),
                        2,
                    )
                    .when(
                        (horizon_agg.value >= 3.5 * 88.42)
                        & (horizon_agg.value < 5 * 88.42),
                        3,
                    )
                    .when(horizon_agg.value >= 5 * 88.42, 4),
                )

            elif task == "wbc":
                # NOTE: unit is mg/L
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 4, 0)
                    .when((horizon_agg.value >= 4) & (horizon_agg.value <= 12), 1)
                    .when((horizon_agg.value > 12), 2),
                )

            elif task == "hb":  # g/dL to g/L
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 8 * 10, 0)
                    .when(
                        (horizon_agg.value >= 8 * 10) & (horizon_agg.value < 10 * 10),
                        1,
                    )
                    .when(
                        (horizon_agg.value >= 10 * 10) & (horizon_agg.value < 12 * 10),
                        2,
                    )
                    .when((horizon_agg.value >= 12 * 10), 3),
                )

            elif task == "bicarbonate":
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when((horizon_agg.value < 22), 0)
                    .when((horizon_agg.value >= 22) & (horizon_agg.value < 29), 1)
                    .when((horizon_agg.value >= 29), 2),
                )

            elif task == "sodium":
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 135, 0)
                    .when((horizon_agg.value >= 135) & (horizon_agg.value < 145), 1)
                    .when((horizon_agg.value >= 145), 2),
                )

            cohorts = cohorts.join(
                horizon_agg.select(self.icustay_key, task_name),
                on=self.icustay_key,
                how="left",
            )
            cohorts = cohorts.toPandas()
            cohorts = spark.createDataFrame(cohorts)
        return cohorts
