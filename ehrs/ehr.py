import logging
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import List, Optional, Union

import numpy as np
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
from sklearn.mixture import GaussianMixture
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class Table:
    fname: str
    timestamp: str
    endtime: Optional[str] = None
    itemid: Optional[Union[str, list[str]]] = None
    value: Optional[List[str]] = None
    uom: Optional[str] = None
    text: Optional[List[str]] = None
    code: Optional[str] = None
    desc: Optional[str] = None
    desc_key: Optional[str] = None


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
    def d_diagnosis_fname(self):
        return self._d_diagnosis_fname

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

        events_dfs = []
        choices_dfs = []
        for table in self.tables:
            fname = table.fname
            table_name = fname.split("/")[-1][: -len(self.ext)]
            table.timestamp = table.timestamp
            logger.info("{} in progress.".format(fname))

            code_to_descriptions = None
            if table.code:
                desc_df = pd.read_csv(os.path.join(self.data_dir, table.desc))
                code_to_descriptions = {
                    table.code: dict(zip(desc_df[table.code], desc_df[table.desc_key]))
                }

            events = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
            if self.debug:
                events = events.limit(100000)

            if (
                self.icustay_key not in events.columns
                and self.hadm_key not in events.columns
            ):
                raise AssertionError(
                    "{} doesn't have one of these columns: {}".format(
                        fname, [self.icustay_key, self.hadm_key]
                    )
                )

            if self.icustay_key in events.columns:
                events = events.drop(self.icustay_key)
            # HADM Join -> duplicated by icy ids -> can process w.  intime
            events = events.join(
                F.broadcast(
                    cohorts.select(
                        self.hadm_key,
                        self.icustay_key,
                        "INTIME",
                        "INTIME_DATE",
                        "ADMITTIME",
                        "LOS",
                    )
                ),
                on=self.hadm_key,
                how="right",
            )
            # To make as date + hours + minutes
            for col in [table.timestamp, "endtime", "stoptime"]:
                if col in events.columns:
                    events = events.withColumn(col, F.to_timestamp(col))
                    new_name = "TIME" if col == table.timestamp else col
                    events = events.withColumn(
                        new_name,
                        F.expr(
                            f"concat('Day ', datediff({col}, INTIME_DATE), ' ', format_string('%02d', hour({col})), ':', format_string('%02d', minute({col})))"
                        ),
                    )

            events = (
                events.withColumn(
                    "_TIME",
                    (
                        F.col(table.timestamp).cast("long")
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
                table.timestamp,
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

            # Itemid Value UOM Text
            if table.endtime:
                events = (
                    events.withColumn("_ENDTIME", F.col(table.endtime))
                    .drop(table.endtime)
                    .withColumnRenamed("_ENDTIME", "ENDTIME")
                )
            else:
                events = events.withColumn("ENDTIME", F.lit(None).cast(StringType()))

            if table.itemid:
                if isinstance(table.itemid, list):
                    template = " ".join(["%s" for _ in table.itemid])
                    events = events.withColumn(
                        "_ITEMID", F.format_string(template, *table.itemid)
                    ).drop(*table.itemid)
                else:
                    events = events.withColumnRenamed(table.itemid, "_ITEMID").drop(
                        table.itemid
                    )
                events = events.withColumnRenamed("_ITEMID", "ITEMID")
            else:
                events = events.withColumn("ITEMID", F.lit(None).cast(StringType()))

            @F.udf(returnType=StringType())
            def merge_value(*args):
                merged = args[0]
                if len(args) >= 2:
                    for arg in args:
                        if arg is not None and arg != "___":
                            merged = arg
                            break
                merged = re.sub(
                    r"\d*\.\d+",
                    lambda x: str(round(float(x.group(0)), 4)),
                    str(merged),
                )
                return merged

            if table.value:
                events = (
                    events.withColumn(
                        "_VALUE", merge_value(*[F.col(i) for i in table.value])
                    )
                    .drop(*table.value)
                    .withColumnRenamed("_VALUE", "VALUE")
                )
            else:
                events = events.withColumn("VALUE", F.lit(None).cast(StringType()))

            if table.uom:
                events = events.withColumnRenamed(table.uom, "UOM")
            else:
                events = events.withColumn("UOM", F.lit(None).cast(StringType()))

            @F.udf(returnType=StringType())
            def process_text(*args):
                text = [str(i) for i in args if i is not None and i != "___"]
                text = " ".join(text)
                text = re.sub(
                    r"\d*\.\d+",
                    lambda x: str(round(float(x.group(0)), 4)),
                    str(text),
                )
                return text

            if table.text:
                events = (
                    events.withColumn(
                        "_TEXT", process_text(*[F.col(i) for i in table.text])
                    )
                    .drop(*table.text)
                    .withColumnRenamed("_TEXT", "TEXT")
                )
            else:
                events = events.withColumn("TEXT", F.lit(None).cast(StringType()))
            events = events.withColumn("TABLE_NAME", F.lit(table_name))

            events = events.select(
                self.icustay_key,
                "_TIME",
                "TIME",
                "ENDTIME",
                "ITEMID",
                "VALUE",
                "TEXT",
                "UOM",
                "TABLE_NAME",
            )
            events_dfs.append(events)

            choices = events.select("ITEMID", "VALUE")
            choices = choices.groupBy("ITEMID").agg(
                F.collect_list("VALUE").alias("CHOICES")
            )
            choices = choices.withColumn("TABLE_NAME", F.lit(table_name))
            choices_dfs.append(choices)

        return reduce(lambda x, y: x.union(y), events_dfs), reduce(
            lambda x, y: x.union(y), choices_dfs
        )

    def make_input(self, cohorts, events, choices, spark):
        schema = StructType(
            [
                StructField("stay_id", IntegerType(), True),
                StructField("time", ArrayType(StringType()), True),
                StructField("endtime", ArrayType(StringType()), True),
                StructField("itemid", ArrayType(StringType()), True),
                StructField("value", ArrayType(StringType()), True),
                StructField("text", ArrayType(StringType()), True),
                StructField("uom", ArrayType(StringType()), True),
                StructField("table_name", ArrayType(StringType()), True),
            ]
        )

        def _make_input(events):
            # To ensure sorting, udf is necessary
            df = events.sort_values("_TIME")
            df = df.drop(columns="_TIME")
            if len(df) <= self.min_event_size:
                return pd.DataFrame(columns=df.columns).rename(
                    columns=lambda x: x.lower()
                )
            return (
                df.groupby(self.icustay_key)
                .agg(list)
                .reset_index(drop=False)
                .rename(columns=lambda x: x.lower())
            )

        if not isinstance(cohorts, pd.DataFrame):
            cohorts = cohorts.toPandas()
        # Allow duplication
        events = events.groupBy(self.icustay_key).applyInPandas(_make_input, schema)
        features = Features(
            {
                "stay_id": Value(dtype="int32"),
                "time": Sequence(feature=Value(dtype="string")),
                "endtime": Sequence(feature=Value(dtype="string")),
                "itemid": Sequence(feature=Value(dtype="string")),
                "value": Sequence(feature=Value(dtype="string")),
                "text": Sequence(feature=Value(dtype="string")),
                "uom": Sequence(feature=Value(dtype="string")),
                "table_name": Sequence(feature=Value(dtype="string")),
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

        def mapper(x, cohorts):
            # Add metadata of each patient
            samples = []
            for values in zip(*x.values()):
                sample = {k: v for k, v in zip(x.keys(), values)}
                stay_id = sample[self.icustay_key]
                row = cohorts.loc[stay_id]
                for col in cohorts.columns:
                    if col in ["ADMITTIME", "time", "INTIME_DATE", "dod"]:
                        continue
                    sample[col] = row[col]
                samples.append(sample)
            x = {
                k: list(v)
                for k, v in zip(samples[0].keys(), zip(*[i.values() for i in samples]))
            }
            return x

        _cohorts = cohorts.set_index(self.icustay_key, drop=True)

        dset = dset.map(mapper, batched=True, fn_kwargs={"cohorts": _cohorts})
        dset.save_to_disk(os.path.join(self.dest, self.ehr_name))

        cohorts.to_csv(
            os.path.join(self.dest, f"{self.ehr_name}_cohort.csv"), index=False
        )

        logger.info("Done encoding events.")

        # Save choices
        choices = choices.toPandas()
        # multiindex (table_name, itemid)
        choices = choices.set_index(["TABLE_NAME", "ITEMID"])
        choices = choices.drop("microbiologyevents", level=0)

        def _value_to_sampling(x):
            floats = []
            strings = []
            for i in x["CHOICES"]:
                try:
                    floats.append(float(i))
                except:
                    strings.append(i)

            # If the number of unique floats is less than 20, treat as categorical
            if len(set(floats)) <= 20:
                strings += [
                    str(i) for i in floats
                ]
                floats = []

            counts = dict(Counter(strings))

            vocab, weights = list(counts.keys()), counts.values()
            weights = [i / sum(weights) for i in weights]

            float_ratio = len(floats) / (len(floats) + len(strings))

            x["VOCAB"] = vocab
            x["ALL_VOCAB"] = set(strings + [str(i) for i in floats])
            x["WEIGHTS"] = weights
            x["FLOAT_RATIO"] = float_ratio

            if len(floats) == 0:
                x["GM"] = {"means": [0, 0, 0], "stds": [0, 0, 0], "weights": [0, 0, 0]}
            else:
                gm = GaussianMixture(n_components=3, random_state=42)
                gm.fit(np.array(floats).reshape(-1, 1))
                x["GM"] = {
                    "means": gm.means_.reshape(-1).tolist(),
                    "stds": np.sqrt(gm.covariances_.reshape(-1)).tolist(),
                    "weights": gm.weights_.tolist(),
                }
            return x

        choices = choices.apply(_value_to_sampling, axis=1).drop(columns="CHOICES")
        choices.to_pickle(os.path.join(self.dest, f"{self.ehr_name}_choices.pkl"))

        return

    def run_pipeline(self, spark) -> None:
        cohorts = self.build_cohorts(cached=self.cache)
        events, choices = self.process_tables(cohorts, spark)
        self.make_input(cohorts, events, choices, spark)

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
