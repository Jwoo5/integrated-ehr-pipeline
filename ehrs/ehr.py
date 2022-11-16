import sys
import os
import re
import shutil
import subprocess
import logging
import pickle
import h5py

import pandas as pd
import numpy as np
import pyspark.sql.functions as F

from typing import Union, List
from functools import reduce
from itertools import chain
from transformers import AutoTokenizer
from tqdm import tqdm
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType


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
                + " which may ignore some arguments such as --first_icu."
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
        self.min_ds_event_size = (
            cfg.min_ds_event_size if cfg.min_ds_event_size is not None else 1
        )
        assert self.min_event_size > 0, (
            "--min_event_size could not be negative or zero", self.min_event_size
        )
        assert self.min_ds_event_size > 0, (
            "--min_ds_event_size could not be negative or zero", self.min_ds_event_size,
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
        self.gap_size = cfg.gap_size
        self.pred_size = cfg.pred_size

        self.first_icu = cfg.first_icu

        self.chunk_size = cfg.chunk_size

        self.dest = cfg.dest
        self.valid_percent = cfg.valid_percent
        self.seed = cfg.seed
        assert 0 <= cfg.valid_percent and cfg.valid_percent <= 0.5

        self.bins = cfg.bins

        self.special_tokens_dict = dict()
        self.max_special_tokens = 100

        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
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
        self._admission_fname  = None
        self._diagnosis_fname = None

        self._icustay_key = None
        self._hadm_key = None

        self.rolling_from_last = cfg.rolling_from_last
        self.data_sampling = cfg.data_sampling
        assert not (cfg.use_more_tables and cfg.ehr=='mimiciii')

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
                "{} do not have required columns to build cohorts.".format(self.icustay_fname)
                + " Please make sure that dataframe for icustays is compatible with other ehrs."
            )

        logger.info(
            "Start building cohorts for {}".format(self.ehr_name)
        )

        obs_size = self.obs_size
        gap_size = self.gap_size

        icustays = icustays[icustays["LOS"] >= (obs_size + gap_size) / 24]
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
            "readmission"
        ] = 0
        if self.first_icu:
            icustays = icustays.loc[
                icustays.groupby(self.hadm_key)[self.determine_first_icu].idxmin()
            ]

        logger.info(
            "cohorts have been built successfully. Loaded {} cohorts.".format(
                len(icustays)
            )
        )
        self.save_to_cache(icustays, self.ehr_name + ".cohorts")

        return icustays

    # TODO process specific tasks according to user choice?
    def prepare_tasks(self, cohorts, cached=False):
        if cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled")
            if labeled_cohorts is not None:
                return labeled_cohorts
            else:
                raise RuntimeError()

        logger.info(
            "Start labeling cohorts for predictive tasks."
        )

        labeled_cohorts = cohorts[[
            self.hadm_key,
            self.icustay_key,
            self.patient_key,
            "readmission",
            "INTIME",
            "OUTTIME",
            "DISCHTIME",
            "IN_ICU_MORTALITY",
            "HOS_DISCHARGE_LOCATION",
        ]].copy()

        # los prediction
        if not self.rolling_from_last:
            labeled_cohorts["los_3day"] = (cohorts["LOS"] > 3).astype(int)
            labeled_cohorts["los_7day"] = (cohorts["LOS"] > 7).astype(int)

        # mortality prediction
        # if the discharge location of an icustay is 'Death'
        #   & intime + obs_size + gap_size <= dischtime <= intime + obs_size + pred_size
        # it is assigned positive label on the mortality prediction
        if self.rolling_from_last:
            labeled_cohorts["mortality"] = (
                (
                (labeled_cohorts["IN_ICU_MORTALITY"] == 1)
                | (labeled_cohorts["HOS_DISCHARGE_LOCATION"] == "Death")
                )
                & (
                    labeled_cohorts["DISCHTIME"] <= labeled_cohorts["OUTTIME"] + self.pred_size * 60 - self.gap_size * 60
                )
            ).astype(int)
        else:
            labeled_cohorts["mortality"] = (
                (
                    (labeled_cohorts["IN_ICU_MORTALITY"] == "Death")
                    | (labeled_cohorts["HOS_DISCHARGE_LOCATION"] == "Death")
                )
                & (
                    self.obs_size * 60 + self.gap_size * 60 < labeled_cohorts["DISCHTIME"]
                )
                & (
                    labeled_cohorts["DISCHTIME"] <= self.obs_size * 60 + self.pred_size * 60
                )
            ).astype(int)
        # if the discharge of 'Death' occurs in icu or hospital
        # we retain these cases for the imminent discharge task
        labeled_cohorts["IN_HOSPITAL_MORTALITY"] = (
            (~labeled_cohorts["IN_ICU_MORTALITY"])
            & (labeled_cohorts["HOS_DISCHARGE_LOCATION"] == "Death")
        ).astype(int)

        # define final acuity prediction task
        labeled_cohorts["final_acuity"] = labeled_cohorts["HOS_DISCHARGE_LOCATION"]
        labeled_cohorts.loc[
            labeled_cohorts["IN_ICU_MORTALITY"] == 1, "final_acuity"
        ] = "IN_ICU_MORTALITY"
        labeled_cohorts.loc[
            labeled_cohorts["IN_HOSPITAL_MORTALITY"] == 1, "final_acuity"
        ] = "IN_HOSPITAL_MORTALITY"
        # NOTE we drop null value samples
        labeled_cohorts = labeled_cohorts[~labeled_cohorts["final_acuity"].isna()]

        with open(os.path.join(self.dest, self.ehr_name + "_final_acuity_classes.tsv"), "w") as f:
            for i, cat in enumerate(
                labeled_cohorts["final_acuity"].astype("category").cat.categories
            ):
                print("{}\t{}".format(i, cat), file=f)
        labeled_cohorts["final_acuity"] = (
            labeled_cohorts["final_acuity"].astype("category").cat.codes
        )


        # define imminent discharge prediction task
        if self.rolling_from_last:
            is_discharged = (
                labeled_cohorts['DISCHTIME'] <= labeled_cohorts['OUTTIME'] + self.pred_size * 60 - self.gap_size * 60
            )
        else:
            is_discharged = (
                (
                    self.obs_size * 60 + self.gap_size * 60 <= labeled_cohorts["DISCHTIME"]
                )
                & (
                    labeled_cohorts["DISCHTIME"] <= self.obs_size * 60 + self.pred_size * 60)
            )
        labeled_cohorts.loc[is_discharged, "imminent_discharge"] = labeled_cohorts.loc[
            is_discharged, "HOS_DISCHARGE_LOCATION"
        ]
        labeled_cohorts.loc[
            is_discharged & (
                (labeled_cohorts["IN_ICU_MORTALITY"] == 1)
                | (labeled_cohorts["IN_HOSPITAL_MORTALITY"] == 1)
            ),
            "imminent_discharge"
        ] = "Death"
        labeled_cohorts.loc[~is_discharged, "imminent_discharge"] = "No Discharge"
        # NOTE we drop null value samples
        labeled_cohorts = labeled_cohorts[~labeled_cohorts["imminent_discharge"].isna()]

        with open(
            os.path.join(self.dest, self.ehr_name + "_imminent_discharge_classes.tsv"), "w"
        ) as f:
            for i, cat in enumerate(
                labeled_cohorts["imminent_discharge"].astype("category").cat.categories
            ):
                print("{}\t{}".format(i, cat), file=f)
        labeled_cohorts["imminent_discharge"] = (
            labeled_cohorts["imminent_discharge"].astype("category").cat.codes
        )

        # clean up unnecessary columns
        labeled_cohorts = labeled_cohorts.drop(
            columns=[
                "IN_ICU_MORTALITY",
                "IN_HOSPITAL_MORTALITY",
                "DISCHTIME",
                "HOS_DISCHARGE_LOCATION"
            ]
        )

        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        logger.info("Done preparing tasks except for diagnosis prediction.")

        return labeled_cohorts


    def process_tables(self, cohorts, spark):
        # in: cohorts, sparksession
        # out: Spark DataFrame with (stay_id, time offset, inp, type, dpe)
        logger.info("Start Preprocessing Tables, Cohort Numbers: {}".format(len(cohorts)))
        cohorts = spark.createDataFrame(cohorts)

        events_dfs = []
        for table in self.tables:
            fname = table["fname"]
            table_name = fname.split('/')[-1][: -len(self.ext)]
            timestamp_key = table["timestamp"]
            excludes = table["exclude"]
            obs_size = self.obs_size
            gap_size = self.gap_size
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

            infer_icustay_from_hadm_key = False

            events = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
            if self.icustay_key not in events.columns:
                infer_icustay_from_hadm_key = True
                if self.hadm_key not in events.columns:
                    raise AssertionError(
                        "{} doesn't have one of these columns: {}".format(
                            fname, [self.icustay_key, self.hadm_key]
                        )
                    )

            events = events.drop(*excludes)
            if table["timeoffsetunit"]=='abs':
                events = events.withColumn(timestamp_key, F.to_timestamp(timestamp_key))

            if infer_icustay_from_hadm_key:
                events = events.join(
                        cohorts.select(self.hadm_key, self.icustay_key, "INTIME", "OUTTIME"),
                        on=self.hadm_key, how="inner"
                    )
                if table["timeoffsetunit"] =='abs':
                    events = (
                        events.withColumn(
                            "TEMP_TIME",
                            F.round((F.col(timestamp_key).cast("long") - F.col("INTIME").cast("long")) / 60)
                        ).filter(F.col("TEMP_TIME") >= 0)
                        .filter(F.col("TEMP_TIME")<=F.col("OUTTIME"))
                        .drop("TEMP_TIME")
                    )
                else:
                    # All tables in eICU has icustay_key -> no need to handle
                    raise NotImplementedError()
                events = events.join(cohorts.select(self.icustay_key), on=self.icustay_key, how='leftsemi')

            else:
                events = events.join(cohorts.select(self.icustay_key, "INTIME", "OUTTIME"), on=self.icustay_key, how="inner")

            if table["timeoffsetunit"] == 'abs':
                events = events.withColumn("TIME", F.round((F.col(timestamp_key).cast("long") - F.col("INTIME").cast("long")) / 60))
                events = events.drop(timestamp_key)
            elif table["timeoffsetunit"] == "min":
                events = events.withColumnRenamed(timestamp_key, "TIME")
            else:
                raise NotImplementedError()

            if self.rolling_from_last:
                events = events.filter(F.col("TIME") >= 0).filter(F.col("TIME") <= F.col("OUTTIME") - gap_size * 60)
                events = events.withColumn("TIME", F.col("TIME") - F.col("OUTTIME") + gap_size * 60)
            elif not self.data_sampling:
                events = events.filter(F.col("TIME") >= 0).filter(F.col("TIME") <= obs_size * 60)

            events = events.drop("INTIME", "OUTTIME", self.hadm_key)

            if code_to_descriptions:
                for col in code_to_descriptions.keys():
                    mapping_expr = F.create_map([F.lit(x) for x in chain(*code_to_descriptions[col].items())])
                    events = events.withColumn(col, mapping_expr[F.col(col)])

            def process_unit(text, type_id):
                # Given (table_name|col|val), generate ([inp], [type], [dpe])
                text = re.sub(r"\d*\.\d+", lambda x: str(round(float(x.group(0)), 4)), str(text))
                number_groups = [g for g in re.finditer(r"([0-9]+([.][0-9]*)?|[0-9]+|\.+)", text)]
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
                        digits = [i for i, j in enumerate(corresponding_numbers) if j==119]

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
                            logger.warn(f"{data_dpe[start:end]} has irregular numerical formats")

                        numbers_cnt += len(group[0])
                    return data_dpe

                dpes = get_dpe(input_ids, number_groups)
                return input_ids, types, dpes

            encoded_table_name = process_unit(table_name, self.table_type_id)
            encoded_cols = {k: process_unit(k, self.column_type_id) for k in events.columns}

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
                        if col in [self.icustay_key, "TIME"] or val is None:
                            continue
                        encoded_col = encoded_cols[col]
                        encoded_val = process_unit(val, self.value_type_id)
                        if len(input_ids) + len(encoded_col[0]) + len(encoded_val[0]) + 2 <= self.max_event_token_len:
                            input_ids += encoded_col[0] + encoded_val[0]
                            types += encoded_col[1] + encoded_val[1]
                            dpes += encoded_col[2] + encoded_val[2]
                        else:
                            break
                    return input_ids, types, dpes
                return F.udf(_process_row, returnType=schema)

            events = (
                events.withColumn("tmp", process_row(encoded_table_name, encoded_cols)(F.struct(*events.columns)))
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
            flatten_lens = np.cumsum(df["INPUTS"].str.len()+1).values
            event_length = len(df)

            if flatten_lens[-1] > self.max_patient_token_len-1:
                # Consider CLS token at first of the flatten input
                flatten_cut_idx = np.searchsorted(flatten_lens, flatten_lens[-1]-self.max_patient_token_len+1)
                flatten_lens = (flatten_lens - flatten_lens[flatten_cut_idx])[flatten_cut_idx+1:]
                event_length = len(flatten_lens)
            
            # Event length should not be longer than max_event_size
            event_length = min(event_length, self.max_event_size)
            df = df.iloc[-event_length:]

            if self.rolling_from_last:
                # For icu len:
                # Iteratively add hi_start and hi_end and time_len
                # Note: Time is arranges as charttime - outtime + gap_size (minus offset)
                # Remove last n hours
                max_obs_len = int(-((df["TIME"].min() // (self.obs_size * 60)) * self.obs_size* 60))
                hi_start = []
                for obs_len in range(self.obs_size * 60, max_obs_len, self.obs_size * 60):
                    if np.searchsorted(df["TIME"].values, -obs_len + self.obs_size * 60) - np.searchsorted(df["TIME"].values, -obs_len) <= self.min_event_size:
                        return events["TIME"].to_frame()
                    hi_start.append(np.searchsorted(df["TIME"].values, -obs_len))
                # If LOS is exactly same with (obs_size+gap_size)*60
                if len(hi_start)==0:
                    hi_start = [0]
                # To allocate list to cell
                fl_start = [flatten_lens[i-1]+1 for i in hi_start]
            else:
                if len(df)<=self.min_event_size:
                    return events["TIME"].to_frame()
                hi_start = 0
                fl_start = 0

            make_hi = lambda cls_id, sep_id, iterable: [[cls_id] + list(i) + [sep_id] for i in iterable]
            make_fl = lambda cls_id, sep_id, iterable: [cls_id] + list(chain(*[list(i) + [sep_id] for i in iterable]))
            
            hi_input = make_hi(self.cls_token_id, self.sep_token_id, df["INPUTS"])
            hi_type = make_hi(self.cls_type_id, self.sep_type_id, df["TYPES"])
            hi_dpe = make_hi(self.others_dpe_id, self.others_dpe_id, df["DPES"])

            fl_input = make_fl(self.cls_token_id, self.sep_token_id, df["INPUTS"])
            fl_type = make_fl(self.cls_type_id, self.sep_type_id, df["TYPES"])
            fl_dpe = make_fl(self.others_dpe_id, self.others_dpe_id, df["DPES"])

            assert len(hi_input) <= self.max_event_size, hi_input
            assert all([len(i)<=self.max_event_token_len for i in hi_input]), hi_input
            assert len(fl_input) <= self.max_patient_token_len, fl_input

            # Add padding to save as numpy array
            hi_input = np.array([np.pad(i, (0, self.max_event_token_len - len(i)), mode='constant') for i in hi_input])
            hi_type = np.array([np.pad(i, (0, self.max_event_token_len - len(i)), mode='constant') for i in hi_type])
            hi_dpe = np.array([np.pad(i, (0, self.max_event_token_len - len(i)), mode='constant') for i in hi_dpe])

            fl_input = np.pad(fl_input, (0, self.max_patient_token_len - len(fl_input)), mode='constant')
            fl_type = np.pad(fl_type, (0, self.max_patient_token_len - len(fl_type)), mode='constant')
            fl_dpe = np.pad(fl_dpe, (0, self.max_patient_token_len - len(fl_dpe)), mode='constant')
            
            stay_id = df[self.icustay_key].values[0]
            # Create caches (cannot write to hdf5 directly with pyspark)
            data = {
                "hi": np.stack([hi_input, hi_type, hi_dpe], axis=1).astype(np.int16),
                "fl": np.stack([fl_input, fl_type, fl_dpe], axis=0).astype(np.int16),
                "hi_start": hi_start,
                "fl_start": fl_start,
                "time": df["TIME"].values,
            }
            with open(os.path.join(self.cache_dir, self.ehr_name, f"{stay_id}.pkl"), "wb") as f:
                pickle.dump(data, f)
            return events["TIME"].to_frame()

        shutil.rmtree(os.path.join(self.cache_dir, self.ehr_name), ignore_errors=True)
        os.makedirs(os.path.join(self.cache_dir, self.ehr_name), exist_ok=True)

        events.groupBy(self.icustay_key).apply(_make_input).write.mode("overwrite").format("noop").save()

        logger.info("Finish Data Preprocessing. Start to write to hdf5")

        f = h5py.File(os.path.join(self.dest, f"{self.ehr_name}.h5"), "w")
        ehr_g = f.create_group("ehr")
        cohorts[['hi_start', 'fl_start', 'time']] = None
        cohorts.reset_index(inplace=True, drop=True)
        for stay_id_file in tqdm(os.listdir(os.path.join(self.cache_dir, self.ehr_name))):
            stay_id = stay_id_file.split(".")[0]
            with open(os.path.join(self.cache_dir, self.ehr_name, stay_id_file), 'rb') as f:
                data = pickle.load(f)
            stay_g = ehr_g.create_group(str(stay_id))
            stay_g.create_dataset('hi', data=data['hi'], dtype='i2', compression='lzf', shuffle=True)
            stay_g.create_dataset('fl', data=data['fl'], dtype='i2', compression='lzf', shuffle=True)
            stay_g.create_dataset('hi_start', data=data['hi_start'], dtype='i')
            stay_g.create_dataset('fl_start', data=data['fl_start'], dtype='i')
            stay_g.create_dataset('time', data = data['time'], dtype='i')

            # If want to acceleate keys split in datasets, read df selectively
            corresponding_idx = cohorts.index[cohorts[self.icustay_key]==int(stay_id)][0]
            cohorts.at[corresponding_idx, 'hi_start'] = data['hi_start']
            cohorts.at[corresponding_idx, 'fl_start'] = data['fl_start']
            cohorts.at[corresponding_idx, 'time'] = data['time']

        shutil.rmtree(os.path.join(self.cache_dir, self.ehr_name), ignore_errors=True)
        # Drop patients with few events
        prev_len = len(cohorts)
        cohorts = cohorts.dropna(subset=['hi_start', 'fl_start', 'time'], how='any')
        logger.info("Total {} patients in the cohort are skipped due to few events".format(prev_len - len(cohorts)))
        cohorts.reset_index(drop=True, inplace=True)
        # Should consider pat_id for split

        shuffled = cohorts.groupby(self.patient_key)[self.patient_key].count().sample(frac=1, random_state=self.seed)
        cum_len = shuffled.cumsum()

        cohorts.loc[cohorts[self.patient_key].isin(
            shuffled[cum_len < int(len(shuffled)*self.valid_percent)].index), 'split'] = 'test'
        cohorts.loc[cohorts[self.patient_key].isin(
            shuffled[(cum_len >= int(len(shuffled)*self.valid_percent)) 
            & (cum_len < int(len(shuffled)*2*self.valid_percent))].index), 'split'] = 'valid'
        cohorts.loc[cohorts[self.patient_key].isin(
            shuffled[cum_len >= int(len(shuffled)*2*self.valid_percent)].index), 'split'] = 'train'

        cohorts.to_csv(os.path.join(self.dest, f'{self.ehr_name}_cohort.csv'), index=False)

        # Record corhots df to hdf5
        for _, row in cohorts.iterrows():
            group = ehr_g[str(row[self.icustay_key])]
            for col in cohorts.columns:
                if col in ["INTIME", "OUTTIME"] or isinstance(row[col], (pd.Timestamp, pd.Timedelta)):
                    continue
                group.attrs[col] = row[col]
        f.close()
        logger.info("Done encoding events.")

        return

    def run_pipeline(self, spark) -> None:
        cohorts = self.build_cohorts(cached=self.cache)
        labeled_cohorts = self.prepare_tasks(cohorts, cached=self.cache)
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

        self.special_tokens_dict.update({
            k: "[unused{}]".format(i)
            for i, k in enumerate(new_special_tokens, start=num_special_tokens+1)
        })

    def make_compatible(self, icustays):
        """
        make different ehrs compatible with one another here
        NOTE: outtime/dischtime is converted to relative minutes from intime
            but, maintain the intime as the original value for later use
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
            "OUTTIME",
            "DISCHTIME",
            "IN_ICU_MORTALITY",
            "HOS_DISCHARGE_LOCATION"
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
            f.to_pickle(
                os.path.join(self.cache_dir, fname)
            )

    def load_from_cache(self, fname):
        cached = os.path.join(self.cache_dir, fname)
        if os.path.exists(cached):
            data = pd.read_pickle(cached)

            logger.info(
                "Loaded data from {}".format(cached)
            )
            return data
        else:
            return None

    def infer_data_extension(self) -> str:
        raise NotImplementedError()

    def download_ehr_from_url(self, url, dest) -> None:
        username = input("Email or Username: ")
        subprocess.run(
            [
                "wget", "-r", "-N", "-c", "np",
                "--user", username,
                "--ask-password", url,
                "-P", dest,
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
                "wget", "-N", "-c",
                "https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip",
                "-P", dest
            ]
        )

        import zipfile

        with zipfile.ZipFile(
            os.path.join(dest, "Multi_Level_CCS_2015.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(os.path.join(dest, "foo.d"))
        os.rename(
            os.path.join(dest, "foo.d", "ccs_multi_dx_tool_2015.csv"),
            os.path.join(dest, "ccs_multi_dx_tool_2015.csv")
        )
        os.remove(os.path.join(dest, "Multi_Level_CCS_2015.zip"))
        shutil.rmtree(os.path.join(dest, "foo.d"))

    def download_icdgem_from_url(self, dest) -> None:
        subprocess.run(
            [
                "wget", "-N", "-c",
                "https://data.nber.org/gem/icd10cmtoicd9gem.csv",
                "-P", dest,
            ]
        )