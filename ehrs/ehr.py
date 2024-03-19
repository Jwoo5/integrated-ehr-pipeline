import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
from functools import reduce
from itertools import chain
from typing import List, Union

import h5py
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
from tqdm import tqdm
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

        self.table_type_id = 1
        self.column_type_id = 2
        self.value_type_id = 3
        self.timeint_type_id = 4
        self.cls_type_id = 5
        self.sep_type_id = 6

        self.others_dpe_id = 0

        self._icustay_fname = None
        self._patient_fname = None
        self._admission_fname = None
        self._diagnosis_fname = None

        self._icustay_key = None
        self._hadm_key = None

        self.use_ed = cfg.use_ed

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

    def build_cohorts(self, spark, cached=False):
        if cached:
            cohorts = self.load_from_cache(self.ehr_name + ".cohorts")
            if cohorts is not None:
                return cohorts

        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustay_fname))
        icustays = self.make_compatible(icustays, spark)

        logger.info("Start building cohorts for {}".format(self.ehr_name))

        icustays = icustays[icustays["LOS"] >= self.obs_size / 24]

        icustays = icustays[
            (self.min_age <= icustays["AGE"]) & (icustays["AGE"] <= self.max_age)
        ]

        # we define labels for the readmission task in this step
        # since it requires to observe each next icustays,
        # which would have been excluded in the final cohorts
        if self.readmission:
            icustays.sort_values([self.hadm_key, self.icustay_key], inplace=True)
            icustays["readmission"] = 1
            icustays.loc[
                icustays.groupby(self.hadm_key)[self.determine_first_icu].idxmax(),
                "readmission",
            ] = 0
        else:
            icustays.sort_values(self.icustay_key, inplace=True)

        logger.info(
            "cohorts have been built successfully. Loaded {} cohorts.".format(
                len(icustays)
            )
        )
        self.save_to_cache(icustays, self.ehr_name + ".cohorts")

        return icustays

    def prepare_tasks(self, cohorts, spark, cached=False):
        if cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled")
            if labeled_cohorts is not None:
                return labeled_cohorts
            else:
                raise RuntimeError()

        logger.info("Start labeling cohorts for predictive tasks.")

        required_cols = [
            self.icustay_key,
            "INTIME",
            "ADMITTIME",
            "DEATHTIME",
        ]
        if self.readmission:
            required_cols.append("readmission")
        if self.los:
            required_cols.append("LOS")
        if self.hadm_key:
            required_cols.append(self.hadm_key)
        if self.patient_key:
            required_cols.append(self.patient_key)
        labeled_cohorts = cohorts[required_cols].copy()

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
        # out: Spark DataFrame with (stay_id, time offset, inp, type, dpe)
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
            table_name = fname.split(self.ext)[0].split("/")[-1]
            timestamp_key = table["timestamp"]
            excludes = table["exclude"]
            logger.info("{} in progress.".format(fname))

            events = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
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

            events = events.drop(*excludes)
            if table["timeoffsetunit"] == "abs":
                events = events.withColumn(timestamp_key, F.to_timestamp(timestamp_key))
                # HADM Join -> duplicated by icy ids -> can process w.  intime
                if self.hadm_key:
                    if self.icustay_key in events.columns:
                        events = events.drop(self.icustay_key)
                    events = events.join(
                        cohorts.select(
                            self.hadm_key, self.icustay_key, "INTIME", "ADMITTIME"
                        ),
                        on=self.hadm_key,
                        how="right",
                    )
                else:
                    events = events.join(
                        cohorts.select(self.icustay_key, "INTIME", "ADMITTIME"),
                        on=self.icustay_key,
                        how="right",
                    )
                events = events.withColumn(
                    "TIME",
                    F.round(
                        (
                            (
                                F.col(timestamp_key).cast("long")
                                - F.col("ADMITTIME").cast("long")
                            )
                            / 60
                        )
                    )
                    - F.col("INTIME"),
                )
                events = events.drop(timestamp_key)

            elif table["timeoffsetunit"] == "min":
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
                    cohorts.select(self.icustay_key, "INTIME", "ADMITTIME"),
                    on=self.icustay_key,
                    how="right",
                )
                # Third, make all timestamps as offset from icu admission
                events = events.withColumn("TIME", F.col("TIME") - F.col("INTIME"))
                events = events.drop(timestamp_key)

            elif table["timeoffsetunit"] == "ms":
                events = events.join(
                    cohorts.select(self.icustay_key, "INTIME"),
                    on=self.icustay_key,
                    how="right",
                )
                events = events.withColumn(
                    "TIME",
                    F.col(timestamp_key) / 1000 / 60 - F.col("INTIME"),
                )
                events = events.drop(timestamp_key)

            else:
                raise NotImplementedError()

            events = events.filter(F.col("TIME") < self.pred_size * 60)

            events = events.drop("INTIME", "ADMITTIME")
            if self.hadm_key in events.columns:
                events = events.drop(self.hadm_key)

            if "code" in table:
                for code_idx in range(len(table["code"])):
                    code = table["code"][code_idx]
                    desc = table["desc"][code_idx]

                    mapping_table = spark.read.csv(
                        os.path.join(self.data_dir, desc), header=True
                    )
                    if "desc_filter_col" in table:
                        mapping_table = mapping_table.filter(
                            F.col(table["desc_filter_col"][code_idx])
                            == table["desc_filter_val"][code_idx]
                        )
                    mapping_table = mapping_table.withColumnRenamed(
                        table["desc_code_col"][code_idx], code
                    )
                    mapping_table = mapping_table.select(
                        code, *table["desc_key"][code_idx]
                    )

                    events = events.join(
                        F.broadcast(mapping_table), on=code, how="left"
                    )
                    events = events.drop(code)
                    for k, v in table["rename_map"][code_idx].items():
                        events = events.withColumnRenamed(k, v)

            def process_unit(text, type_id):
                # Given (table_name|col|val), generate ([inp], [type], [dpe])
                text = re.sub(
                    r"\d*\.\d+", lambda x: str(round(float(x.group(0)), 4)), str(text)
                )
                number_groups = [
                    g for g in re.finditer(r"([0-9]+([.][0-9]*)?|[0-9]+|\.+)", text)
                ]
                text = re.sub(r"([0-9\.])", r" \1 ", text)
                input_ids = self.tokenizer.encode(text, add_special_tokens=False)
                types = [type_id] * len(input_ids)

                def get_dpe(tokens, number_groups):
                    number_ids = [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 119]
                    numbers = [i for i, j in enumerate(tokens) if j in number_ids]
                    numbers_cnt = 0
                    data_dpe = [0] * len(tokens)
                    for group in number_groups:
                        if group[0] == "." * len(group[0]):
                            numbers_cnt += len(group[0])
                            continue

                        start = numbers[numbers_cnt]
                        end = numbers[numbers_cnt + len(group[0]) - 1] + 1
                        corresponding_numbers = tokens[start:end]
                        digits = [
                            i for i, j in enumerate(corresponding_numbers) if j == 119
                        ]

                        # Case Integer
                        if len(digits) == 0:
                            data_dpe[start:end] = list(range(len(group[0]) + 5, 5, -1))
                        # Case Float
                        elif len(digits) == 1:
                            digit_idx = len(group[0]) - digits[0]
                            data_dpe[start:end] = list(
                                range(len(group[0]) + 5 - digit_idx, 5 - digit_idx, -1)
                            )
                        else:
                            logger.warn(
                                f"{data_dpe[start:end]} has irregular numerical formats"
                            )

                        numbers_cnt += len(group[0])
                    return data_dpe

                dpes = get_dpe(input_ids, number_groups)
                return input_ids, types, dpes

            encoded_table_name = process_unit(table_name, self.table_type_id)
            encoded_cols = {
                k: process_unit(k, self.column_type_id) for k in events.columns
            }

            schema = StructType(
                [
                    StructField("INPUTS", ArrayType(IntegerType()), False),
                    StructField("TYPES", ArrayType(IntegerType()), False),
                    StructField("DPES", ArrayType(IntegerType()), False),
                ]
            )

            def process_row(encoded_table_name, encoded_cols):
                def _process_row(row):
                    """
                    input: row (cols: icustay_id, timestamp, ...)
                    output: (input, type, dpe)
                    """
                    row = row.asDict()
                    # Should INITIALIZE with blank arrays to prevent corruption in Pyspark... Why??
                    input_ids, types, dpes = [], [], []
                    input_ids += encoded_table_name[0]
                    types += encoded_table_name[1]
                    dpes += encoded_table_name[2]
                    encoded_table_name
                    for col, val in row.items():
                        if col in [self.icustay_key, "TIME"]:
                            continue
                        if val is None:
                            continue
                        encoded_col = encoded_cols[col]
                        encoded_val = process_unit(val, self.value_type_id)
                        if (
                            len(input_ids)
                            + len(encoded_col[0])
                            + len(encoded_val[0])
                            + 2
                            <= self.max_event_token_len
                        ):
                            input_ids += encoded_col[0] + encoded_val[0]
                            types += encoded_col[1] + encoded_val[1]
                            dpes += encoded_col[2] + encoded_val[2]
                        else:
                            break
                    return input_ids, types, dpes

                return F.udf(_process_row, returnType=schema)

            events = (
                events.withColumn(
                    "tmp",
                    process_row(encoded_table_name, encoded_cols)(
                        F.struct(*events.columns)
                    ),
                )
                .withColumn("INPUTS", F.col("tmp.INPUTS"))
                .withColumn("TYPES", F.col("tmp.TYPES"))
                .withColumn("DPES", F.col("tmp.DPES"))
                .select(self.icustay_key, "TIME", "INPUTS", "TYPES", "DPES")
            )
            events_dfs.append(events)
        return reduce(lambda x, y: x.union(y), events_dfs)

    def make_input(self, cohorts, events, spark):
        @F.pandas_udf(returnType="TIME int", functionType=F.PandasUDFType.GROUPED_MAP)
        def _make_input(events):
            # Actually, this function does not have to return anything.
            # However, return something(TIME) is required to satisfy the PySpark requirements.
            df = events.sort_values("TIME")
            flatten_cut_idx = -1
            # Consider SEP
            flatten_lens = np.cumsum(df["INPUTS"].str.len() + 1).values
            event_length = len(df)

            if flatten_lens[-1] > self.max_patient_token_len - 1:
                # Consider CLS token at first of the flatten input
                flatten_cut_idx = np.searchsorted(
                    flatten_lens, flatten_lens[-1] - self.max_patient_token_len + 1
                )
                flatten_lens = (flatten_lens - flatten_lens[flatten_cut_idx])[
                    flatten_cut_idx + 1 :
                ]
                event_length = len(flatten_lens)

            # Event length should not be longer than max_event_size
            event_length = min(event_length, self.max_event_size)
            df = df.iloc[-event_length:]

            flatten_lens = [0] + list(flatten_lens + 1)

            hi_start = []
            fl_start = []
            for time in range(self.pred_size):
                event_idx = np.searchsorted(df["TIME"].values, time * 60)
                hi_start.append(event_idx)
                fl_start.append(flatten_lens[event_idx])

            make_hi = lambda cls_id, sep_id, iterable: [
                [cls_id] + list(i) + [sep_id] for i in iterable
            ]
            make_fl = lambda cls_id, sep_id, iterable: [cls_id] + list(
                chain(*[list(i) + [sep_id] for i in iterable])
            )

            hi_input = make_hi(self.cls_token_id, self.sep_token_id, df["INPUTS"])
            hi_type = make_hi(self.cls_type_id, self.sep_type_id, df["TYPES"])
            hi_dpe = make_hi(self.others_dpe_id, self.others_dpe_id, df["DPES"])

            fl_input = make_fl(self.cls_token_id, self.sep_token_id, df["INPUTS"])
            fl_type = make_fl(self.cls_type_id, self.sep_type_id, df["TYPES"])
            fl_dpe = make_fl(self.others_dpe_id, self.others_dpe_id, df["DPES"])

            assert len(hi_input) <= self.max_event_size, hi_input
            assert all([len(i) <= self.max_event_token_len for i in hi_input]), hi_input
            assert len(fl_input) <= self.max_patient_token_len, fl_input

            # Add padding to save as numpy array
            hi_input = np.array(
                [
                    np.pad(i, (0, self.max_event_token_len - len(i)), mode="constant")
                    for i in hi_input
                ]
            )
            hi_type = np.array(
                [
                    np.pad(i, (0, self.max_event_token_len - len(i)), mode="constant")
                    for i in hi_type
                ]
            )
            hi_dpe = np.array(
                [
                    np.pad(i, (0, self.max_event_token_len - len(i)), mode="constant")
                    for i in hi_dpe
                ]
            )

            stay_id = df[self.icustay_key].values[0]
            # Create caches (cannot write to hdf5 directly with pyspark)
            data = {
                "hi": np.stack([hi_input, hi_type, hi_dpe], axis=1).astype(np.int16),
                "fl": np.stack([fl_input, fl_type, fl_dpe], axis=0).astype(np.int16),
                "hi_start": hi_start,
                "fl_start": fl_start,
                "fl_lens": flatten_lens,
                "time": df["TIME"].values,
            }
            with open(
                os.path.join(self.cache_dir, self.ehr_name, f"{stay_id}.pkl"), "wb"
            ) as f:
                pickle.dump(data, f)
            return events["TIME"].to_frame()

        shutil.rmtree(os.path.join(self.cache_dir, self.ehr_name), ignore_errors=True)
        os.makedirs(os.path.join(self.cache_dir, self.ehr_name), exist_ok=True)

        if isinstance(cohorts, pd.DataFrame):
            cohorts = spark.createDataFrame(cohorts)

        # Allow duplication
        events.groupBy(self.icustay_key).apply(_make_input).write.mode(
            "overwrite"
        ).format("noop").save()

        logger.info("Finish Data Preprocessing. Start to write to hdf5")

        f = h5py.File(os.path.join(self.dest, f"{self.ehr_name}.h5"), "w")
        ehr_g = f.create_group("ehr")

        if not isinstance(cohorts, pd.DataFrame):
            cohorts = cohorts.toPandas()
        cohorts[["hi_start", "fl_start", "time", "fl_lens"]] = None

        active_stay_ids = [
            int(i.split(".")[0])
            for i in os.listdir(os.path.join(self.cache_dir, self.ehr_name))
        ]
        logger.info(
            "Total {} patients in the cohort are skipped due to few events".format(
                len(cohorts) - len(active_stay_ids)
            )
        )

        cohorts.reset_index(drop=True, inplace=True)

        # Should consider pat_id for split
        shuffle_key = self.patient_key if self.patient_key else self.icustay_key
        for seed in self.seed:
            shuffled = (
                cohorts.groupby(shuffle_key)[shuffle_key]
                .count()
                .sample(frac=1, random_state=seed)
            )
            cum_len = shuffled.cumsum()

            cohorts.loc[
                cohorts[shuffle_key].isin(
                    shuffled[cum_len < int(sum(shuffled) * self.valid_percent)].index
                ),
                f"split_{seed}",
            ] = "test"
            cohorts.loc[
                cohorts[shuffle_key].isin(
                    shuffled[
                        (cum_len >= int(sum(shuffled) * self.valid_percent))
                        & (cum_len < int(sum(shuffled) * 2 * self.valid_percent))
                    ].index
                ),
                f"split_{seed}",
            ] = "valid"
            cohorts.loc[
                cohorts[shuffle_key].isin(
                    shuffled[
                        cum_len >= int(sum(shuffled) * 2 * self.valid_percent)
                    ].index
                ),
                f"split_{seed}",
            ] = "train"

        for stay_id in tqdm(cohorts[self.icustay_key].values):
            # Although the events does not exist, we still need to create the empty array
            if os.path.exists(
                os.path.join(self.cache_dir, self.ehr_name, f"{stay_id}.pkl")
            ):
                with open(
                    os.path.join(self.cache_dir, self.ehr_name, f"{stay_id}.pkl"), "rb"
                ) as f:
                    data = pickle.load(f)
            else:
                data = {
                    "hi": np.ones(
                        (1, 3, self.max_event_token_len), dtype=np.int16
                    ),  # if zero -> cause nan
                    "fl": np.ones((3, 1), dtype=np.int16),
                    "hi_start": np.zeros(self.pred_size, dtype=np.int16),
                    "fl_start": np.zeros(self.pred_size, dtype=np.int16),
                    "fl_lens": np.zeros(1, dtype=np.int16),
                    "time": np.zeros(1, dtype=np.int16),
                }
            stay_g = ehr_g.create_group(str(stay_id))
            stay_g.create_dataset(
                "hi", data=data["hi"], dtype="i2", compression="lzf", shuffle=True
            )
            stay_g.create_dataset(
                "fl", data=data["fl"], dtype="i2", compression="lzf", shuffle=True
            )
            stay_g.create_dataset("hi_start", data=data["hi_start"], dtype="i")
            stay_g.create_dataset("fl_start", data=data["fl_start"], dtype="i")
            stay_g.create_dataset("fl_lens", data=data["fl_lens"], dtype="i")
            stay_g.create_dataset("time", data=data["time"], dtype="i")

            corresponding_idx = cohorts.index[
                cohorts[self.icustay_key] == int(stay_id)
            ][0]
            row = cohorts.loc[corresponding_idx]
            for col in cohorts.columns:
                if col in ["ADMITTIME", "hi_start", "fl_start", "fl_lens", "time"]:
                    continue
                stay_g.attrs[col] = row[col]

            # If want to acceleate keys split in datasets, read df selectively
            cohorts.at[corresponding_idx, "hi_start"] = str(list(data["hi_start"]))
            cohorts.at[corresponding_idx, "fl_start"] = str(list(data["fl_start"]))
            cohorts.at[corresponding_idx, "fl_lens"] = str(list(data["fl_lens"]))
            cohorts.at[corresponding_idx, "time"] = str(list(data["time"]))

        shutil.rmtree(os.path.join(self.cache_dir, self.ehr_name), ignore_errors=True)
        # Drop patients with few events
        cohorts.to_csv(
            os.path.join(self.dest, f"{self.ehr_name}_cohort.csv"), index=False
        )

        f.close()
        logger.info("Done encoding events.")

        return

    def run_pipeline(self, spark) -> None:
        cohorts = self.build_cohorts(spark, cached=self.cache)
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

    def make_compatible(self, icustays, spark):
        """
        make different ehrs compatible with one another here
        NOTE: timestamps are converted to relative minutes from admittime
        but, maintain the admittime as the original value for later use
        """
        raise NotImplementedError()

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
