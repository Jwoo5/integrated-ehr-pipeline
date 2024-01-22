import logging
import os
import re
import shutil
import subprocess
import sys
from functools import reduce
from itertools import chain
from typing import List, Union

import pandas as pd
import pyspark.sql.functions as F
from datasets import Dataset, Features, Sequence, Value
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class EHR(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.cache = cfg.cache
        cache_dir = os.path.expanduser("~/.cache/ehr")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        if self.cache:
            logger.warn(
                "--cache is set to True. Note that it forces to load cached"
                " data from {},".format(cache_dir)
                + " which may ignore some arguments such as --first_icu, as well as task related arguments (--mortality, --los_3day, etc.)"
                " If you want to avoid this, do not set --cache to True."
            )

        self.data_dir = cfg.data
        self.ccs_path = cfg.ccs
        self.gem_path = cfg.gem
        self.ext = cfg.ext

        self.max_event_size = (
            cfg.max_event_size if cfg.max_event_size is not None else sys.maxsize
        )
        self.min_event_size = (
            cfg.min_event_size if cfg.min_event_size is not None else 1
        )
        assert self.min_event_size <= self.max_event_size, (
            self.min_event_size,
            self.max_event_size,
        )

        self.max_event_token_len = cfg.max_event_token_len
        self.max_patient_token_len = cfg.max_patient_token_len

        self.max_age = cfg.max_age if cfg.max_age is not None else sys.maxsize
        self.min_age = cfg.min_age if cfg.min_age is not None else 0
        assert self.min_age <= self.max_age, (self.min_age, self.max_age)

        self.obs_size = cfg.obs_size
        self.pred_size = cfg.pred_size

        # tasks
        self.mortality = cfg.mortality
        self.los = cfg.los
        self.readmission = cfg.readmission
        self.diagnosis = cfg.diagnosis
        self.creatinine = cfg.creatinine
        self.platelets = cfg.platelets
        self.wbc = cfg.wbc
        self.hb = cfg.hb
        self.bicarbonate = cfg.bicarbonate
        self.sodium = cfg.sodium

        self.dest = cfg.dest
        self.valid_percent = cfg.valid_percent
        self.seed = [int(s) for s in cfg.seed.replace(" ", "").split(",")]
        assert 0 <= cfg.valid_percent and cfg.valid_percent <= 0.5

        self.special_tokens_dict = dict()
        self.max_special_tokens = 100

        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        self._icustay_fname = None
        self._patient_fname = None
        self._admission_fname = None
        self._diagnosis_fname = None

        self._icustay_key = None
        self._hadm_key = None

        self.use_ed = cfg.use_ed
        self.lab_only = cfg.lab_only
        self.debug = cfg.debug
        self.add_chart = cfg.add_chart

    @property
    def icustay_fname(self):
        return self._icustay_fname

    @property
    def patient_fname(self):
        return self._patient_fname

    @property
    def admission_fname(self):
        return self._admission_fname

    @property
    def diagnosis_fname(self):
        return self._diagnosis_fname

    @property
    def icustay_key(self):
        return self._icustay_key

    @property
    def hadm_key(self):
        return self._hadm_key

    @property
    def patient_key(self):
        return self._patient_key

    @property
    def determine_first_icu(self):
        return self._determine_first_icu

    @property
    def num_special_tokens(self):
        return len(self.special_tokens_dict)

    def build_cohorts(self, icustays, cached=False):
        if cached:
            cohorts = self.load_from_cache(self.ehr_name + ".cohorts")
            if cohorts is not None:
                return cohorts

        if not self.is_compatible(icustays):
            raise AssertionError(
                "{} do not have required columns to build cohorts.".format(
                    self.icustay_fname
                )
                + " Please make sure that dataframe for icustays is compatible with other ehrs."
            )

        logger.info("Start building cohorts for {}".format(self.ehr_name))

        icustays = icustays[icustays["LOS"] >= self.obs_size / 24]

        icustays = icustays[
            (self.min_age <= icustays["AGE"]) & (icustays["AGE"] <= self.max_age)
        ]

        # we define labels for the readmission task in this step
        # since it requires to observe each next icustays,
        # which would have been excluded in the final cohorts
        icustays.sort_values([self.hadm_key, self.icustay_key], inplace=True)

        if self.readmission:
            icustays["readmission"] = 1
            icustays.loc[
                icustays.groupby(self.hadm_key)[self.determine_first_icu].idxmax(),
                "readmission",
            ] = 0

        logger.info(
            "cohorts have been built successfully. Loaded {} cohorts.".format(
                len(icustays)
            )
        )
        self.save_to_cache(icustays, self.ehr_name + ".cohorts")

        return icustays

    # TODO process specific tasks according to user choice?
    def prepare_tasks(self, cohorts, spark, cached=False):
        if cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled")
            if labeled_cohorts is not None:
                return labeled_cohorts
            else:
                raise RuntimeError()

        logger.info("Start labeling cohorts for predictive tasks.")

        labeled_cohorts = cohorts[
            [
                self.hadm_key,
                self.icustay_key,
                self.patient_key,
                "INTIME",
                "INTIME_DATE",
                "ADMITTIME",
                "DEATHTIME",
                "LOS",
                # "readmission",
            ]
        ].copy()

        # los prediction
        if self.los:
            for i in self.los:
                labeled_cohorts["los_{}".format(i)] = (cohorts["LOS"] > i).astype(int)

        if self.mortality:
            for i in self.mortality:
                labeled_cohorts["mortality_{}".format(i)] = (
                    cohorts["DEATHTIME"] - cohorts["INTIME"]
                    < self.pred_size * 60 + i * 60 * 24
                ).astype(int)

        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        logger.info("Done preparing tasks except for diagnosis prediction.")

        return labeled_cohorts

    def process_tables(self, cohorts, spark):
        # in: cohorts, sparksession
        # out: Spark DataFrame with (stay_id, time offset, inp)
        if isinstance(cohorts, pd.DataFrame):
            logger.info(
                "Start Preprocessing Tables, Cohort Numbers: {}".format(len(cohorts))
            )
            cohorts = spark.createDataFrame(cohorts)
            print("Converted Cohort to Pyspark DataFrame")
        else:
            logger.info("Start Preprocessing Tables")

        if self.use_ed:
            ed = spark.read.csv(
                os.path.join(self.data_dir, self._ed_fname), header=True
            ).select(self.hadm_key, self._ed_key, "intime", "outtime")

        events_dfs = []
        for table in self.tables:
            fname = table["fname"]
            table_name = fname.split("/")[-1][: -len(self.ext)]
            timestamp_key = table["timestamp"]
            includes = table["include"]
            logger.info("{} in progress.".format(fname))

            code_to_descriptions = None
            if "code" in table:
                code_to_descriptions = {
                    k: pd.read_csv(os.path.join(self.data_dir, v))
                    for k, v in zip(table["code"], table["desc"])
                }
                code_to_descriptions = {
                    k: dict(zip(v[k], v[d_k]))
                    for (k, v), d_k in zip(
                        code_to_descriptions.items(), table["desc_key"]
                    )
                }

            events = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
            if self.debug:
                events = events.limit(100000)

            if self.icustay_key not in events.columns:
                if self.hadm_key not in events.columns:
                    raise AssertionError(
                        "{} doesn't have one of these columns: {}".format(
                            fname, [self.icustay_key, self.hadm_key]
                        )
                    )

            if "ed/" in fname and table["timeoffsetunit"] == "abs":
                # join ed table & set as hadm_id & drop ed key
                # Since unit is abs -> do not need to additionally process time
                events = events.join(
                    F.broadcast(ed),
                    on=self._ed_key,
                    how="left",
                )
                if table["timestamp"] == "ED_INTIME":
                    events = events.withColumnRenamed("intime", "ED_INTIME")
                elif table["timestamp"] == "ED_OUTTIME":
                    events = events.withColumnRenamed("outtime", "ED_OUTTIME")

                events = events.drop(self._ed_key, "intime", "outtime")

            events = events.select(*includes)
            if table["timeoffsetunit"] == "abs":
                if self.icustay_key in events.columns:
                    events = events.drop(self.icustay_key)
                # HADM Join -> duplicated by icy ids -> can process w.  intime
                events = events.join(
                    cohorts.select(
                        self.hadm_key,
                        self.icustay_key,
                        "INTIME",
                        "INTIME_DATE",
                        "ADMITTIME",
                        "LOS",
                    ),
                    on=self.hadm_key,
                    how="right",
                )
                # To make as date + hours + minutes
                for col in [timestamp_key, "endtime", "stoptime"]:
                    if col in events.columns:
                        events = events.withColumn(col, F.to_timestamp(col))
                        new_name = "TIME" if col == timestamp_key else col
                        events = events.withColumn(
                            new_name,
                            F.expr(
                                f"concat('Day ', datediff({col}, INTIME_DATE), ' ', format_string('%02d', hour({col})), ':', format_string('%02d', minute({col})))"
                            ),
                        )
            elif table["timeoffsetunit"] == "min":
                raise NotImplementedError()
                # First, make all timestamps as offset from hospital admission
                events = events.join(
                    cohorts.select(self.hadm_key, self.icustay_key, "INTIME"),
                    on=self.icustay_key,
                    how="right",
                )
                events = events.withColumn(
                    "TIME", F.col(timestamp_key).cast("int") + F.col("INTIME")
                )
                # Second, duplicate events to handle multiple icustays
                events = events.drop(self.icustay_key, "INTIME")
                events = events.join(
                    cohorts.select(self.hadm_key, self.icustay_key),
                    on=self.hadm_key,
                    how="right",
                )
                events = events.join(
                    cohorts.select(self.icustay_key, "INTIME", "ADMITTIME", "LOS"),
                    on=self.icustay_key,
                    how="right",
                )
                # Third, make all timestamps as offset from icu admission
                events = events.withColumn("TIME", F.col("TIME") - F.col("INTIME"))
                events = events.drop(timestamp_key)

            else:
                raise NotImplementedError()
            events = (
                events.withColumn(
                    "_TIME",
                    (
                        F.col(timestamp_key).cast("long")
                        - F.col("ADMITTIME").cast("long")
                    )
                    / 60
                    - F.col("INTIME"),
                )
                .filter(F.col("_TIME") >= 0)
                .filter(F.col("_TIME") < F.col("LOS") * 60 * 24)
            )  # Only within icu stay

            events = events.drop(
                "LOS",
                timestamp_key,
                "INTIME",
                "INTIME_DATE",
                "ADMITTIME",
                self.hadm_key,
            )

            if code_to_descriptions:
                for col in code_to_descriptions.keys():
                    mapping_expr = F.create_map(
                        [F.lit(x) for x in chain(*code_to_descriptions[col].items())]
                    )
                    events = events.withColumn(col, mapping_expr[F.col(col)])

            events = events.withColumn("MASK_TARGET", F.col(table["mask_target"][0]))

            def process_row(table_name):
                def _process_row(row):
                    """
                    input: row (cols: icustay_id, timestamp, ...)
                    output: (text)
                    """
                    row = row.asDict()
                    # Should INITIALIZE with blank arrays to prevent corruption in Pyspark... Why??
                    text = table_name
                    for col, val in row.items():
                        if col in [self.icustay_key, "TIME", "_TIME", "MASK_TARGET"]:
                            continue
                        if val is None:
                            continue
                        # text += " " + col + " " + str(val)
                        text += " " + str(val)
                    return text

                return F.udf(_process_row, returnType=StringType())

            events = events.withColumn(
                "TEXT",
                process_row(table_name)(F.struct(*events.columns)),
            ).select(self.icustay_key, "TIME", "_TIME", "TEXT", "MASK_TARGET")
            events_dfs.append(events)
        return reduce(lambda x, y: x.union(y), events_dfs)

    def make_input(self, cohorts, events, spark):
        schema = StructType(
            [
                StructField("stay_id", IntegerType(), True),
                StructField("time", ArrayType(StringType()), True),
                StructField("text", ArrayType(StringType()), True),
                StructField("mask_target", ArrayType(StringType()), True),
            ]
        )

        def _make_input(events):
            # Actually, this function does not have to return anything.
            # However, return something(TIME) is required to satisfy the PySpark requirements.
            df = events.sort_values("_TIME")

            if len(df) <= self.min_event_size:
                return pd.DataFrame(columns=["stay_id", "time", "text", "mask_target"])
            # Remove duplicated glucoses (in lab/chart)
            df["glucose_value"] = df["TEXT"].str.extract(
                r"glucose.* (\d+)", flags=re.IGNORECASE, expand=False
            )
            df["is_glucose"] = df.apply(
                lambda x: x["TEXT"] if pd.isna(x["glucose_value"]) else None, axis=1
            )  # Prevent to drop dextrose/insulin
            df.drop_duplicates(
                subset=["glucose_value", "is_glucose", "TIME"],
                inplace=True,
                keep="last",
            )  # Only for Glucoses.... Not for others...
            return pd.DataFrame(
                [
                    {
                        "stay_id": int(df[self.icustay_key].values[0]),
                        "time": df["TIME"].values,
                        "text": df["TEXT"].values,
                        "mask_target": df["MASK_TARGET"].values,
                    }
                ]
            )

        if not isinstance(cohorts, pd.DataFrame):
            cohorts = cohorts.toPandas()
        # Allow duplication
        events = events.groupBy(self.icustay_key).applyInPandas(_make_input, schema)
        features = Features(
            {
                "stay_id": Value(dtype="int32"),
                "time": Sequence(feature=Value(dtype="string")),
                "text": Sequence(feature=Value(dtype="string")),
                "mask_target": Sequence(feature=Value(dtype="string")),
            }
        )
        dset = Dataset.from_spark(events, features=features)

        processed_df = events.select("stay_id", "time").toPandas()
        previous_len = len(cohorts)
        cohorts = pd.merge(cohorts, processed_df, on="stay_id", how="inner")
        logger.info(
            "Total {} patients are skipped due to few events".format(
                previous_len - len(cohorts)
            )
        )
        cohorts.reset_index(drop=True, inplace=True)

        # Should consider pat_id for split
        for seed in self.seed:
            shuffled = (
                cohorts.groupby(self.patient_key)[self.patient_key]
                .count()
                .sample(frac=1, random_state=seed)
            )
            cum_len = shuffled.cumsum()

            cohorts.loc[
                cohorts[self.patient_key].isin(
                    shuffled[cum_len < int(sum(shuffled) * self.valid_percent)].index
                ),
                f"split_{seed}",
            ] = "test"
            cohorts.loc[
                cohorts[self.patient_key].isin(
                    shuffled[
                        (cum_len >= int(sum(shuffled) * self.valid_percent))
                        & (cum_len < int(sum(shuffled) * 2 * self.valid_percent))
                    ].index
                ),
                f"split_{seed}",
            ] = "valid"
            cohorts.loc[
                cohorts[self.patient_key].isin(
                    shuffled[
                        cum_len >= int(sum(shuffled) * 2 * self.valid_percent)
                    ].index
                ),
                f"split_{seed}",
            ] = "train"

        def mapper(x):
            # Add metadata of each patient
            stay_id = x[self.icustay_key]
            row = cohorts.loc[cohorts[self.icustay_key] == stay_id].iloc[0]
            for col in cohorts.columns:
                if col in ["ADMITTIME", "time", "INTIME_DATE"]:
                    continue
                x[col] = row[col]
            return x

        dset = dset.map(mapper, num_proc=self.cfg.num_threads)
        dset.save_to_disk(os.path.join(self.dest, self.ehr_name))

        cohorts.to_csv(
            os.path.join(self.dest, f"{self.ehr_name}_cohort.csv"), index=False
        )

        logger.info("Done encoding events.")

        return

    def run_pipeline(self, spark) -> None:
        cohorts = self.build_cohorts(cached=self.cache)
        labeled_cohorts = self.prepare_tasks(cohorts, spark, cached=self.cache)
        events = self.process_tables(labeled_cohorts, spark)
        self.make_input(labeled_cohorts, events, spark)

    def add_special_tokens(self, new_special_tokens: Union[str, List]) -> None:
        if isinstance(new_special_tokens, str):
            new_special_tokens = [new_special_tokens]

        num_special_tokens = self.num_special_tokens
        overlapped = []
        for new_special_token in new_special_tokens:
            if new_special_token in self.special_tokens_dict:
                overlapped.append(new_special_token)

        if len(overlapped) > 0:
            logger.warn(
                "There are some tokens that have already been set to special tokens."
                " Please provide only NEW tokens. Aborted."
            )
            return None
        elif num_special_tokens + len(new_special_tokens) > self.max_special_tokens:
            logger.warn(
                f"Total additional special tokens should be less than {self.max_special_tokens}"
                " Aborted."
            )
            return None

        self.special_tokens_dict.update(
            {
                k: "[unused{}]".format(i)
                for i, k in enumerate(new_special_tokens, start=num_special_tokens + 1)
            }
        )

    def make_compatible(self, icustays):
        """
        make different ehrs compatible with one another here
        NOTE: timestamps are converted to relative minutes from admittime
        but, maintain the admittime as the original value for later use
        """
        raise NotImplementedError()

    def is_compatible(self, icustays):
        checklist = [
            self.hadm_key,
            self.icustay_key,
            self.patient_key,
            "LOS",
            "AGE",
            "INTIME",
            "ADMITTIME",
            "DEATHTIME",
        ]
        for item in checklist:
            if item not in icustays.columns.to_list():
                return False
        return True

    def save_to_cache(self, f, fname, use_pickle=False) -> None:
        if use_pickle:
            import pickle

            with open(os.path.join(self.cache_dir, fname), "wb") as fptr:
                pickle.dump(f, fptr)
        else:
            f.to_pickle(os.path.join(self.cache_dir, fname))

    def load_from_cache(self, fname):
        cached = os.path.join(self.cache_dir, fname)
        if os.path.exists(cached):
            data = pd.read_pickle(cached)

            logger.info("Loaded data from {}".format(cached))
            return data
        else:
            return None

    def infer_data_extension(self) -> str:
        raise NotImplementedError()

    def download_ehr_from_url(self, url, dest) -> None:
        username = input("Email or Username: ")
        subprocess.run(
            [
                "wget",
                "-r",
                "-N",
                "-c",
                "np",
                "--user",
                username,
                "--ask-password",
                url,
                "-P",
                dest,
            ]
        )
        output_dir = url.replace("https://", "").replace("http://", "")

        if not os.path.exists(os.path.join(dest, output_dir)):
            raise AssertionError(
                "Download failed. Please check your network connection or "
                "if you log in with a credentialed user"
            )

    def download_ccs_from_url(self, dest) -> None:
        subprocess.run(
            [
                "wget",
                "-N",
                "-c",
                "https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip",
                "-P",
                dest,
            ]
        )

        import zipfile

        with zipfile.ZipFile(
            os.path.join(dest, "Multi_Level_CCS_2015.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(os.path.join(dest, "foo.d"))
        os.rename(
            os.path.join(dest, "foo.d", "ccs_multi_dx_tool_2015.csv"),
            os.path.join(dest, "ccs_multi_dx_tool_2015.csv"),
        )
        os.remove(os.path.join(dest, "Multi_Level_CCS_2015.zip"))
        shutil.rmtree(os.path.join(dest, "foo.d"))

    def download_icdgem_from_url(self, dest) -> None:
        subprocess.run(
            [
                "wget",
                "-N",
                "-c",
                "https://data.nber.org/gem/icd10cmtoicd9gem.csv",
                "-P",
                dest,
            ]
        )
