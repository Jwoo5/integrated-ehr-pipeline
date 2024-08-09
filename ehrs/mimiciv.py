import glob
import logging
import os

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType

from ehrs import EHR, Table, register_ehr

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
                    url="https://physionet.org/files/mimiciv/2.0/", dest=self.data_dir
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
        self._d_diagnosis_fname = "hosp/d_icd_diagnoses" + self.ext
        self._determine_first_icu = "INTIME"

        labevents = Table(
            fname="hosp/labevents" + self.ext,
            timestamp="charttime",
            endtime=None,
            itemid="itemid",
            value=["value", "valuenum"],
            uom="valueuom",
            text=None,
            code="itemid",
            desc="hosp/d_labitems" + self.ext,
            desc_key="label",
            merge_value=True,
        )

        microbiologyevents = Table(
            fname="hosp/microbiologyevents" + self.ext,
            timestamp="charttime",
            endtime=None,
            itemid=None,
            value=None,
            uom=None,
            text=["test_name", "org_name", "ab_name", "interpretation", "comments"],
            code=None,
            desc=None,
            desc_key=None,
        )
        prescriptions = Table(
            fname="hosp/prescriptions" + self.ext,
            timestamp="starttime",
            endtime="stoptime",
            itemid=["drug", "prod_strength"],
            value=["dose_val_rx"],
            uom="dose_unit_rx",
            text=["route"],
            code=None,
            desc=None,
            desc_key=None,
        )
        inputevents = Table(
            fname="icu/inputevents" + self.ext,
            timestamp="starttime",
            endtime="endtime",
            itemid="itemid",
            value=["amount"],
            uom="amountuom",
            text=None,
            code="itemid",
            desc="icu/d_items" + self.ext,
            desc_key="label",
        )
        outputevents = Table(
            fname="icu/outputevents" + self.ext,
            timestamp="charttime",
            endtime=None,
            itemid="itemid",
            value=["value"],
            uom="valueuom",
            text=None,
            code="itemid",
            desc="icu/d_items" + self.ext,
            desc_key="label",
        )
        procedureevents = Table(
            fname="icu/procedureevents" + self.ext,
            timestamp="starttime",
            endtime="endtime",
            itemid="itemid",
            value=["value"],
            uom="valueuom",
            text=["location", "locationcategory"],
            code="itemid",
            desc="icu/d_items" + self.ext,
            desc_key="label",
        )

        self.tables = [
            labevents,
            microbiologyevents,
            prescriptions,
            inputevents,
            outputevents,
            procedureevents,
        ]

        if self.add_chart:
            self.tables += [
                Table(
                    fname="icu/chartevents" + self.ext,
                    timestamp="charttime",
                    endtime=None,
                    itemid="itemid",
                    value=["value", "valuenum"],
                    uom="valueuom",
                    text=None,
                    code="itemid",
                    desc="icu/d_items" + self.ext,
                    desc_key="label",
                )
            ]

        if self.lab_only:
            self.tables = [self.tables[0]]

        self._icustay_key = "stay_id"
        self._hadm_key = "hadm_id"
        self._patient_key = "subject_id"

    def build_cohorts(self, spark, cached=False):
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustay_fname))

        icustays = self.make_compatible(icustays, spark)
        self.icustays = icustays

        cohorts = super().build_cohorts(icustays, spark, cached=cached)

        return cohorts

    def make_compatible(self, icustays, spark):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patient_fname))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))

        # prepare icustays according to the appropriate format
        icustays = icustays.rename(
            columns={
                "los": "LOS",
                "intime": "INTIME",
                "outtime": "OUTTIME",
            }
        )
        admissions = admissions.rename(
            columns={
                "dischtime": "DISCHTIME",
                "admittime": "ADMITTIME",
            }
        )

        icustays = icustays[icustays["first_careunit"] == icustays["last_careunit"]]
        icustays["INTIME"] = pd.to_datetime(
            icustays["INTIME"], infer_datetime_format=True, utc=True
        )
        icustays["OUTTIME"] = pd.to_datetime(
            icustays["OUTTIME"], infer_datetime_format=True, utc=True
        )

        icustays = icustays.merge(patients, on="subject_id", how="left")
        icustays["AGE"] = (
            icustays["INTIME"].dt.year
            - icustays["anchor_year"]
            + icustays["anchor_age"]
        )

        icustays = icustays.merge(
            admissions[
                [self.hadm_key, "DISCHTIME", "ADMITTIME", "race", "discharge_location"]
            ],
            how="left",
            on=self.hadm_key,
        )

        icustays["ADMITTIME"] = pd.to_datetime(
            icustays["ADMITTIME"], infer_datetime_format=True, utc=True
        )
        icustays["DISCHTIME"] = pd.to_datetime(
            icustays["DISCHTIME"], infer_datetime_format=True, utc=True
        )
        icustays["IN_ICU_MORTALITY"] = (
            (icustays["INTIME"] < icustays["DISCHTIME"])
            & (icustays["DISCHTIME"] <= icustays["OUTTIME"])
            & (icustays["discharge_location"] == "DIED")
        )

        sofa_df = pd.read_csv(
            os.path.join(self.cfg.derived_path, "sofa.csv"),
            usecols=["stay_id", "starttime", "sofa_24hours"],
        )
        sofa_df = pd.merge(
            icustays[["stay_id", "INTIME"]], sofa_df, on="stay_id", how="left"
        )
        sofa_df["starttime"] = pd.to_datetime(
            sofa_df["starttime"], infer_datetime_format=True, utc=True
        )
        sofa_df["offset"] = sofa_df.apply(
            lambda x: (x["starttime"] - x["INTIME"]).total_seconds() // 60,
            axis=1,
        )
        sofa_df = (
            sofa_df.sort_values("offset")
            .groupby("stay_id")
            .agg({"offset": list, "sofa_24hours": list})
        )
        sofa_df["sofa"] = sofa_df.apply(
            lambda x: (x["offset"], x["sofa_24hours"]), axis=1
        )
        sofa_df.reset_index(inplace=True)
        icustays = pd.merge(
            icustays, sofa_df[["stay_id", "sofa"]], on="stay_id", how="left"
        )

        return icustays

    def vis_score(self, cohorts, spark):
        weights = spark.read.csv(
            os.path.join(self.cfg.derived_path, "weight_durations.csv"), header=True
        ).drop("weight_type")

        weights = (
            weights.withColumn("weight_starttime", F.to_timestamp("starttime"))
            .withColumn("weight_endtime", F.to_timestamp("endtime"))
            .drop("starttime", "endtime")
            .withColumnRenamed("stay_id", "weight_stay_id")
        )

        ie = spark.read.csv(
            os.path.join(self.data_dir, "icu/inputevents" + self.ext), header=True
        )
        ie = ie.select("stay_id", "starttime", "endtime", "itemid", "rate", "rateuom")
        ie = ie.withColumn("starttime", F.to_timestamp("starttime"))
        ie = ie.withColumn("endtime", F.to_timestamp("endtime"))

        ie = ie.filter(
            F.col("itemid").isin([221662, 221653, 221289, 221986, 221906, 222315])
        )

        ie = ie.join(
            F.broadcast(weights),
            on=[
                ie.stay_id == weights.weight_stay_id,
                ie.starttime >= weights.weight_starttime,
                ie.starttime < weights.weight_endtime,
            ],
            how="left",
        ).drop("weight_stay_id", "weight_starttime", "weight_endtime")

        cohorts = spark.createDataFrame(cohorts)

        ie = ie.join(
            F.broadcast(cohorts.select("stay_id", "INTIME")), on="stay_id", how="left"
        )

        ie = (
            ie.withColumn(
                "starttime",
                ((F.col("starttime") - F.col("INTIME")).cast("long") / 60).cast("long"),
            )
            .withColumn(
                "endtime",
                ((F.col("endtime") - F.col("INTIME")).cast("long") / 60).cast("long"),
            )
            .drop("INTIME")
        )

        ie = ie.dropna(subset=["rate", "rateuom", "starttime", "endtime"])

        ie = ie.withColumn(
            "VIS",
            F.when(
                F.col("itemid").isin([221662, 221653]), F.col("rate")
            )  # dopamine, dobutamine
            .when(F.col("itemid") == 221289, F.col("rate") * 100)  # epinephrine
            .when(F.col("itemid") == 221986, F.col("rate") * 10)  # milrinone
            .when(
                F.col("itemid") == 222315,  # vasopressin
                F.when(
                    F.col("rateuom") == "units/hour",
                    F.col("rate") / 60 * 10000 / F.col("weight"),
                ).otherwise(
                    F.col("rate") * 10000 / F.col("weight")
                ),  # units/minute
            )
            .when(
                F.col("itemid") == 221906,  # norepinephrine
                F.when(F.col("rateuom") == "mcg/kg/min", F.col("rate") * 100).otherwise(
                    F.col("rate") * 1000 * 100  # mg/kg/min
                ),
            )
            .otherwise(F.lit(None)),
        )

        ie = ie.dropna(subset="VIS")

        @F.udf(ArrayType(DoubleType()))
        def cumulate_vis(starttimes, endtimes, viss):
            viss = list(map(float, viss))
            unique_timestamps = sorted(set(starttimes).union(set(endtimes)))
            vis = []
            for timestamp in unique_timestamps:
                total_vis = sum(
                    [
                        viss[i]
                        for i in range(len(starttimes))
                        if starttimes[i] <= timestamp < endtimes[i]
                    ]
                )
                vis.append(float(total_vis))

            return vis

        @F.udf(ArrayType(IntegerType()))
        def get_unique_timestamps(starttimes, endtimes):
            return sorted(set(starttimes).union(set(endtimes)))

        ie = ie.groupBy("stay_id").agg(
            F.collect_list("starttime").alias("starttimes"),
            F.collect_list("endtime").alias("endtimes"),
            F.collect_list("VIS").alias("viss"),
        )
        ie = ie.withColumn(
            "vis_time",
            get_unique_timestamps(F.col("starttimes"), F.col("endtimes")),
        ).withColumn(
            "vis_score",
            cumulate_vis(F.col("starttimes"), F.col("endtimes"), F.col("viss")),
        )
        cohorts = cohorts.join(
            F.broadcast(ie.select("stay_id", "vis_time", "vis_score")),
            on="stay_id",
            how="left",
        )

        cohorts = cohorts.toPandas()

        cohorts["vis"] = cohorts.apply(
            lambda x: (
                (x["vis_time"], x["vis_score"])
                if x["vis_time"] is not None
                else ([], [])
            ),
            axis=1,
        )
        cohorts.to_pickle("cohorts.pkl")
        return cohorts.drop(columns=["vis_time", "vis_score"])

    def infer_data_extension(self) -> str:
        if (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv.gz"))) == 22
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv.gz"))) == 9
        ):
            ext = ".csv.gz"
        elif (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv"))) == 22
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv"))) == 9
        ):
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext
