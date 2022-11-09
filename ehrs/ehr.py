import sys
import os
import re
import shutil
import subprocess
import logging

from typing import Union, List

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sortedcontainers import SortedList
from pandarallel import pandarallel

logger = logging.getLogger(__name__)

class EHR(object):
    def __init__(self, cfg):
        self.cfg = cfg

        pandarallel.initialize(progress_bar=False, nb_workers=cfg.num_threads)

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
        assert self.min_event_size > 0, (
            "--min_event_size could not be negative or zero", self.min_event_size
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

        self.table_type_id = 0
        self.column_type_id = 1
        self.value_type_id = 2
        self.timeint_type_id = 3
        self.cls_type_id = 4
        self.sep_type_id = 5

        self.others_dpe_id = 0

        self._icustay_fname = None
        self._patient_fname = None
        self._admission_fname  = None
        self._diagnosis_fname = None

        self._icustay_key = None
        self._hadm_key = None

        self.rolling_from_last = cfg.rolling_from_last
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
            icustays.groupby(self.hadm_key)["INTIME"].idxmax(),
            "readmission"
        ] = 0
        if self.first_icu:
            icustays = icustays.groupby(self.hadm_key).first().reset_index()

        cohorts = icustays

        logger.info(
            "cohorts have been built successfully. Loaded {} cohorts.".format(
                len(cohorts)
            )
        )
        self.save_to_cache(cohorts, self.ehr_name + ".cohorts")

        return cohorts

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

    def prepare_events(self, cohorts, cached=False):
        if cached:
            cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled.dx.dropped")
            cohort_events = self.load_from_cache(self.ehr_name + ".events")
            if (cohorts is not None) and (cohort_events is not None):
                return cohorts, cohort_events
            else:
                raise RuntimeError()

        logger.info(
            "Start preparing medical events for each cohort."
        )

        cohort_events = {id: SortedList() for id in cohorts[self.icustay_key].to_list()}

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
            columns = pd.read_csv(
                os.path.join(self.data_dir, fname), index_col=0, nrows=0
            ).columns.to_list()
            if self.icustay_key not in columns:
                infer_icustay_from_hadm_key = True
                if self.hadm_key not in columns:
                    raise AssertionError(
                        "{} doesn't have one of these columns: {}".format(
                            fname, [self.icustay_key, self.hadm_key]
                        )
                    )
            # Parallelize this part
            # Use pandarallel
            events = pd.read_csv(
                os.path.join(self.data_dir, fname)
            )
            events.drop(columns=excludes, inplace=True)
            if infer_icustay_from_hadm_key:
                events = events[events[self.hadm_key].isin(cohorts[self.hadm_key])]
            else:
                events = events[events[self.icustay_key].isin(cohorts[self.icustay_key])]

            if table["timeoffsetunit"] == 'abs':
                events[timestamp_key] = pd.to_datetime(
                    events[timestamp_key], infer_datetime_format=True
                )

            if infer_icustay_from_hadm_key:
                events = events.merge(
                    cohorts[[self.hadm_key, self.icustay_key, "INTIME", "OUTTIME"]],
                    on=self.hadm_key,
                    how="inner"
                )
                if table['timeoffsetunit'] == 'abs':
                    events["TEMP_TIME"] = (events[timestamp_key] - events["INTIME"]).dt.total_seconds() // 60
                    events = events[(events["TEMP_TIME"]>= 0) & (events["TEMP_TIME"]<=events["OUTTIME"])]
                    events.drop(columns=["TEMP_TIME"], inplace=True)
                    events = events[events[self.icustay_key].isin(cohorts[self.icustay_key])]
                else:
                    # All tables in eICU has icustay_key -> no need to handle
                    raise NotImplementedError()

            else:
                events = events.merge(cohorts[[self.icustay_key, "INTIME", "OUTTIME"]], on=self.icustay_key, how="inner")
            # Convert time compatible to relative minutes from intime
            # Type timedelta
            if table['timeoffsetunit'] =='abs':
                events[timestamp_key] = (events[timestamp_key] - events["INTIME"]).dt.total_seconds() // 60
            # Type int, relative minute from intime
            # Outtime is also relative offset from intime
            elif table['timeoffsetunit'] =='min':
                pass
            else:
                raise NotImplementedError()
            
            # which means that the event has been charted before / after the icustay
            if self.rolling_from_last:
                events = events[(events[timestamp_key]>=0) & (events[timestamp_key]<=events['OUTTIME']-gap_size*60)]
            else:
                events = events[(events[timestamp_key]>=0) & (events[timestamp_key]<=obs_size*60)]

            # Rearrange TIME to be relative to outtime (for rolling_from_last)
            events[timestamp_key] = events[timestamp_key] - events['OUTTIME'] + gap_size * 60
            
            events.drop(columns=["INTIME", "OUTTIME", self.hadm_key], inplace=True, errors="ignore")

            def linearize_event(event):
                cols = []
                vals = []
                for col, val in event.to_dict().items():
                    if pd.isna(val) or col in [self.icustay_key, timestamp_key]:
                        continue
                    # convert code to description if applicable
                    if (
                        code_to_descriptions is not None
                        and col in code_to_descriptions
                    ):
                        val = code_to_descriptions[col][val]
                    if re.search(r"\d+\.\d+\.\d+", str(val)) or re.search(r"\d+\.\d+\.\d+", str(col)):
                        logger.warn("val: {} col:{} has the irregular numeric format".format(val, col))
                    val = re.sub(r"\d*\.\d+", lambda x: str(round(float(x.group(0)), 4)), str(val))
                    col = re.sub(r"\d*\.\d+", lambda x: str(round(float(x.group(0)), 4)), str(col))

                    cols.append(col)
                    vals.append(val)
                event_string = (cols, vals)

                return (event[self.icustay_key], event[timestamp_key], table_name, event_string)

            linearized_events = events.parallel_apply(linearize_event, axis=1)
            
            [cohort_events[stay_id].add((i, j, k)) for stay_id, i, j, k in linearized_events]

        cohort_events = {
            k: list(v) if len(v) <= self.max_event_size else list(v[-self.max_event_size:])
            for k, v in cohort_events.items()
            if len(v) >= self.min_event_size
        }
        skipped = len(cohorts) - len(cohort_events)
        # Should remove skipped patients from cohort
        cohorts = cohorts[cohorts[self.icustay_key].isin(cohort_events.keys())].reset_index(drop=True)

        logger.info(
            "Done preparing events for the given cohorts."
            f" Skipped {skipped} cohorts since they have too few"
            " (or no) corresponding medical events."
        )

        self.save_to_cache(
            cohorts, self.ehr_name + ".cohorts.labeled.dx.dropped", use_pickle=True
        )
        self.save_to_cache(
            cohort_events, self.ehr_name + ".events", use_pickle=True
        )

        return cohorts, cohort_events

    def encode_events(self, cohorts, cohort_events, cached=False):
        # if cached:
        #     encoded_events = self.load_from_cache(self.ehr_name + ".cohorts.labeled.events.encoded")
        #     if encoded_events is not None:
        #         self.encoded_events = encoded_events
        #         return self.encoded_events
        total_events = sum([len(v) for v in cohort_events.values()])

        np.memmap(
            os.path.join(self.dest, f'{self.ehr_name}.hi.npy'),
            dtype=np.int16,
            mode='w+',
            shape=(total_events, 3, self.max_event_token_len)
        )
        np.memmap(
            os.path.join(self.dest, f'{self.ehr_name}.fl.npy'),
            dtype=np.int16,
            mode='w+',
            shape=(len(cohorts),3, self.max_patient_token_len)
        )

        # XXX special tokens for the time interval (zero, last)
        zero_time_interval = 0
        last_time_interval = -1

        collated_timestamps = {
            icustay_id: [event[0] for event in events]
            for icustay_id, events in cohort_events.items()
        }
        # calc time interval to the next event (i, i+1)
        time_intervals = {
            icustay_id: np.subtract(t[1:], t[:-1])
            for icustay_id, t in collated_timestamps.items()
        }
        # exclude zero time intervals for quantizing
        time_intervals_flattened = {
            k: v[v != zero_time_interval] for k, v in time_intervals.items()
        }
        time_intervals_flattened = np.sort(
            np.concatenate([v for v in time_intervals_flattened.values()])
        )

        # XXX append "special" time interval for the last event
        time_intervals = {
            k: np.append(v, last_time_interval) for k, v in time_intervals.items()
        }

        # XXX exclude +- 1%?
        bins = np.arange(self.bins + 1)
        bins = bins / bins[-1]
        bins = np.quantile(time_intervals_flattened, bins)
        with open(os.path.join(self.dest, self.ehr_name + "_time_gap_bins.tsv"), "w") as f:
            for i, quantile in enumerate(bins):
                print("{}\t{}".format(i, quantile), file=f)
        logger.info("Time buckets: {}".format(bins))
        
        self.add_special_tokens(
            ["[TIME_INT_ZERO]", "[TIME_INT_LAST]"]
            + ["[TIME_INT_{}]".format(i) for i in range(1, len(bins))]
        )

        def _quantize(t: int):
            if t == zero_time_interval:
                return self.special_tokens_dict["[TIME_INT_ZERO]"]
            elif t == last_time_interval:
                return self.special_tokens_dict["[TIME_INT_LAST]"]
            token_idx = np.searchsorted(bins[1:], t)+1
            return self.special_tokens_dict["[TIME_INT_{}]".format(token_idx)]

        time_intervals = {
            k: map(_quantize, v) for k, v in time_intervals.items()
        }

        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)
        tokenizer.add_special_tokens({
            "additional_special_tokens": list(self.special_tokens_dict.values())
        })
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id

        # icustay_id: [(rel time, table name, events string, time token)...]
        encoded_events = {
            k: [
                (*event, t_int)
                for event, t_int in zip(v, time_intervals[k])
            ]
            for k, v in cohort_events.items()
        }
        logger.info("Finished time interval processing.")
        del time_intervals_flattened
        del time_intervals
        del cohort_events
        cohorts.reset_index(names='fl_idx', inplace=True)
        icustay_id_to_hi_start = dict(zip(encoded_events.keys(), np.cumsum([len(i) for i in encoded_events.values()])))
        cohorts['hi_end'] = cohorts[self.icustay_key].map(icustay_id_to_hi_start)
        """
        1) we can batch_encode list of table_names / list of event_strings / list of time_intervals
        2) recover events: list of [tokenized(table_name), tokenized(event_string), [TIME_INT_{}]]
            for zip(table_names, event_strings, time_intervals, ...)]
        3) assign token type embeddings
        4) assign digit place embeddings (only for event_string, otherwise 0)
        4) flatten events: list of [tokenized(table_name, event_string), [TIME_INT_{}]] (token type embedding + dpe embedding)
            4-1) (for flattened structure) flatten them: if flattened size > self.max_token_size,
            4-2) calculate number of events for the corresponding icustay --> num_events
            4-3) while i
                4-3-1) get the next index where token type == "[TIME]"
                4-3-2) if flattened_size - index <= self.max_token_size: break
                        otherwise, loop
            4-4) num_events -= i, flattened = flattened[-(flattened_size - index):], hier = hier[-(num_events):]
        5) prepend [CLS], append [SEP] for each hierarchical event, for each flattened event
        """
        # Cannot know hi idx at here -> approximate
        # Parallelize encoding    
        

        def encode_stay(input):
            events_per_icustay, hi_end, fl_idx = input
            times = [x[0] for x in events_per_icustay]
            # make list of table names
            table_names = [x[1] for x in events_per_icustay]
            # tokenize table names
            table_names_tokenized = tokenizer(table_names, add_special_tokens=False).input_ids
            # generate the same shape list filled with self.table_type_id
            table_type_tokens = [[self.table_type_id] * len(x) for x in table_names_tokenized]
            table_dpe_tokens = [[self.others_dpe_id] * len(x) for x in table_names_tokenized]
            # make list of events: (cols, vals)
            event_strings = [x[2] for x in events_per_icustay] # list of (List[cols], List[vals])

            # tokenize column names
            # cols_tokenized: List[List[List[col_tokens]]]
            #   e.g., cols_tokenized[event_i][col_j]: list of tokens of j-th column names of i-th event

            cols, col_number_groups_list = zip(
                *[self.split_and_get_number_groups(x[0]) for x in event_strings]
            )
            vals, val_number_groups_list = zip(
                *[self.split_and_get_number_groups(x[1]) for x in event_strings]
            )

            cols_tokenized = [
                tokenizer(x, add_special_tokens=False).input_ids for x in cols
            ]
            # generate the same shape list filled with self.column_type_id
            column_type_tokens = [
                [[self.column_type_id] * len(y) for y in x] for x in cols_tokenized
            ]
            column_dpe_tokens = self.get_dpe(cols_tokenized, col_number_groups_list)

            # tokenize column values
            # vals_tokenized: List[List[List[val_tokens]]]
            #   e.g., vals_tokenized[event_i][col_j]: list of tokens of j-th column values of i-th event
            vals_tokenized = [
                tokenizer(x, add_special_tokens=False).input_ids for x in vals
            ]
            # generate the same shape list filled with self.value_type_id
            value_type_tokens = [
                [[self.value_type_id] * len(y) for y in x] for x in vals_tokenized
            ]
            value_dpe_tokens = self.get_dpe(vals_tokenized, val_number_groups_list)
            # linearize column names & values as well as token types
            # events_tokenized: List[List[tokens]]
            #   e.g., events_tokenized[event_i]: list of tokens of i-th event such as
            #       [col0, val0, col1, val1, ...]
            # col_val_type_tokens: the same shape list with events_tokenized filled
            #   with corresponding token type ids
            events_tokenized = []
            col_val_type_tokens = []
            col_val_dpe_tokens = []
            for (
                cols, vals, col_type_tokens, val_type_tokens, col_dpe_tokens, val_dpe_tokens, table_name_tokenized
            ) in zip(
                cols_tokenized, vals_tokenized, column_type_tokens, value_type_tokens, column_dpe_tokens, value_dpe_tokens, table_names_tokenized
            ):
                event_tokenized = []
                col_val_type_token = []
                col_val_dpe_token = []
                max_len = self.max_event_token_len - len(table_name_tokenized) - 3 # 3 for [CLS], [SEP], [TIME]
                for (
                    col, val, col_type_token, val_type_token, col_dpe_token, val_dpe_token,
                ) in zip(
                    cols, vals, col_type_tokens, val_type_tokens, col_dpe_tokens, val_dpe_tokens,
                ):
                    # Should stop when the length over max_event_token_len
                    if len(event_tokenized) + len(col) + len(val) > max_len:
                        break
                    event_tokenized.extend(col + val)
                    col_val_type_token.extend(col_type_token + val_type_token)
                    col_val_dpe_token.extend(col_dpe_token + val_dpe_token)

                events_tokenized.append(event_tokenized)
                col_val_type_tokens.append(col_val_type_token)
                col_val_dpe_tokens.append(col_val_dpe_token)

            # make list of time intervals
            time_intervals = [x[3] for x in events_per_icustay]
            # tokenize time intervals
            # this should yield a corresponding special token id for each time interval
            time_intervals_tokenized = tokenizer(
                time_intervals, add_special_tokens=False
            ).input_ids
            # generate the same shape list filled with self.timeint_type_id
            timeint_type_tokens = [
                [self.timeint_type_id] for _ in time_intervals_tokenized
            ]
            timeint_dpe_tokens = [[self.others_dpe_id] for _ in time_intervals_tokenized]

            # encoded_events[icustay_id]: sequence of tokenized events [table_name, event_strings, time_interval]
            #   List[List[tokens]], where encoded_events[icustay_id][i] is a sequence of input ids for i-th event of icustay_id
            # token_type_embeddings[ocustay]: corresponding token type ids

            # NOTE: [CLS] and [SEP] only added at first/end of flatten input, but [TIME] inserted between events
            # Should retain recent events!
            flatten_cut_idx = -1
            flatten_lens = np.cumsum([len(i)+len(j)+1 for i,j in zip(table_names_tokenized, events_tokenized)])
            event_length = len(table_names_tokenized)
            if flatten_lens[-1] >self.max_patient_token_len-2:
                flatten_cut_idx = np.searchsorted(flatten_lens, flatten_lens[-1]-self.max_patient_token_len+2)
                flatten_lens = (flatten_lens - flatten_lens[flatten_cut_idx])[flatten_cut_idx+1:]
                event_length = len(flatten_lens)
                times = times[-event_length:]

            if self.rolling_from_last:
                # For icu len:
                # Iteratively add hi_start and hi_end and time_len
                # Note: Time is arranges as charttime - outtime + gap_size (minus offset)
                # Remove last n hours
                max_obs_len = int(-((times[0] // (self.obs_size * 60)) * self.obs_size* 60))
                starts = []
                for obs_len in range(self.obs_size * 60, max_obs_len, self.obs_size * 60):
                    if np.searchsorted(times, -obs_len + self.obs_size * 60) -np.searchsorted(times, -obs_len) <= self.min_event_size:
                        starts = []
                        break
                    starts.append(np.searchsorted(times, -obs_len))
                if starts == []:
                    {
                        "hi_start": None,
                        "fl_start": None,
                    }
                # To allocate list to cell
                hi_start = [-event_length + i + hi_end for i in starts]
                fl_start = [flatten_lens[i-1]+1 for i in starts]
            else:
                hi_start = hi_end - event_length
                fl_start = 0

            hi_input = [
                [cls_token_id] + table_name + event + time_interval + [sep_token_id]
                for event_idx, table_name, event, time_interval in zip(
                    range(len(table_names_tokenized)),
                    table_names_tokenized,
                    events_tokenized,
                    time_intervals_tokenized,
                ) if event_idx > flatten_cut_idx
            ]
            hi_type = [
                [self.cls_type_id] + table_type_token + col_val_type_token + timeint_type_token + [self.sep_type_id]
                for event_idx, table_type_token, col_val_type_token, timeint_type_token in zip(
                    range(len(table_names_tokenized)), table_type_tokens, col_val_type_tokens, timeint_type_tokens
                ) if event_idx > flatten_cut_idx
            ]
            hi_dpe = [
                [self.others_dpe_id] + table_dpe_token + col_val_dpe_token + timeint_dpe_token + [self.others_dpe_id]
                for event_idx, table_dpe_token, col_val_dpe_token, timeint_dpe_token in zip(
                    range(len(table_names_tokenized)), table_dpe_tokens, col_val_dpe_tokens, timeint_dpe_tokens
                ) if event_idx > flatten_cut_idx                
            ]
            
            fl_input = (
                [cls_token_id]
                + [j for i in [table_name + event + time_interval for event_idx, table_name, event, time_interval in zip(
                        range(len(table_names_tokenized)), table_names_tokenized, events_tokenized, time_intervals_tokenized,
                    ) if event_idx > flatten_cut_idx] for j in i]
                + [sep_token_id]
            )

            fl_type = (
                [self.cls_type_id]
                + [j for i in [table_type_token + col_val_type_token + timeint_type_token for event_idx, table_type_token, col_val_type_token, timeint_type_token in zip(
                        range(len(table_names_tokenized)), table_type_tokens, col_val_type_tokens, timeint_type_tokens,
                    ) if event_idx > flatten_cut_idx] for j in i]
                + [self.sep_type_id]
            )

            fl_dpe = (
                [self.others_dpe_id]
                + [j for i in [table_dpe_token + col_val_dpe_token + timeint_dpe_token for event_idx, table_dpe_token, col_val_dpe_token, timeint_dpe_token in zip(
                        range(len(table_names_tokenized)), table_dpe_tokens, col_val_dpe_tokens, timeint_dpe_tokens,
                    ) if event_idx > flatten_cut_idx] for j in i]
                + [self.others_dpe_id]
            )

            assert all([len(i)<=self.max_event_token_len for i in hi_input])
            assert len(fl_input) <= self.max_patient_token_len

            # Add padding to save as numpy array
            hi_input = np.array([np.pad(i, (0, self.max_event_token_len - len(i)), mode='constant') for i in hi_input])
            hi_type = np.array([np.pad(i, (0, self.max_event_token_len - len(i)), mode='constant') for i in hi_type])
            hi_dpe = np.array([np.pad(i, (0, self.max_event_token_len - len(i)), mode='constant') for i in hi_dpe])

            fl_input = np.pad(fl_input, (0, self.max_patient_token_len - len(fl_input)), mode='constant')
            fl_type = np.pad(fl_type, (0, self.max_patient_token_len - len(fl_type)), mode='constant')
            fl_dpe = np.pad(fl_dpe, (0, self.max_patient_token_len - len(fl_dpe)), mode='constant')
            
            hierarchical_data = np.memmap(
                os.path.join(self.dest, f'{self.ehr_name}.hi.npy'),
                dtype=np.int16,
                mode='r+',
                shape=(event_length, 3, self.max_event_token_len),
                offset = (hi_end - event_length) * 3 * self.max_event_token_len * 2,
            )
            flatten_data = np.memmap(
                os.path.join(self.dest, f'{self.ehr_name}.fl.npy'),
                dtype=np.int16,
                mode='r+',
                shape=(1, 3, self.max_patient_token_len),
                offset = fl_idx * 3 * self.max_patient_token_len * 2,
            )

            # Save data into memmap format
            hierarchical_data[:, :, :] = np.stack([hi_input, hi_type, hi_dpe], axis=1).astype(np.int16)

            flatten_data[0, :, :] = np.stack([fl_input, fl_type, fl_dpe], axis=0).astype(np.int16)

            hierarchical_data.flush()
            flatten_data.flush()
            return {
                "hi_start": hi_start,
                "fl_start": fl_start,
            }
        # Do not need to pass whole cohorts df, only pass icustay_id, hi_end, fl_idx

        indices = [encode_stay((encoded_events[row[self.icustay_key]], row["hi_end"], row["fl_idx"])) for _, row in tqdm(cohorts.iterrows())]
        indices_df = pd.DataFrame(indices)
        del encoded_events
        cohorts = pd.concat([cohorts, indices_df], axis=1)
        if self.rolling_from_last:
            cohorts = cohorts[cohorts['hi_start'].str.len()!=0]
        # Should consider hadm_id for split
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

        logger.info("Done encoding events.")

        return

    def run_pipeline(self) -> None:
        cohorts = self.build_cohorts(cached=self.cache)
        labeled_cohorts = self.prepare_tasks(cohorts, cached=self.cache)
        cohorts, cohort_events = self.prepare_events(labeled_cohorts, cached=self.cache)
        encoded_events = self.encode_events(cohorts, cohort_events, cached=self.cache)

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

    def split_and_get_number_groups(self, row):
        """
        input:
            row: List[str], columns or values of an event
        output:
            splitteds: List[str]
            number_groups_list: List[List[int]]
        """
        splitted_list = []
        number_groups_list = []

        for data in row:
            # Use regex to find all int/floats in the string
            number_groups = [
                group
                for group in re.finditer(r"([0-9]+([.][0-9]*)?|[0-9]+|\.+)", data)
            ]
            number_groups_list.append(number_groups)
            splitted_list.append(re.sub(r"([0-9\.])", r" \1 ", data))

        return splitted_list, number_groups_list


    # Case 1. '2 5', '25'
    # Case 2. Too long number
    def get_dpe(self, data_tokenized, data_number_groups_list) -> list:
        """
        input:
            data_tokenized: List[List[List[int]]]
            data_number_groups_list: List[List[List[int]]]
        output:
            dpe: List[List[List[int]]]
        cf. numerical values are already rounded into 4 digits
        Not Digit/Padding -> 0
        if original string is '1111.1111 aaa'-> '987654321 000'
        """
        number_ids = [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 119]  # [0-9\.]
        dpes = []
        for event_id, number_groups_list in zip(
            data_tokenized, data_number_groups_list
        ):
            event_dpes = []
            for data_id, number_groups in zip(event_id, number_groups_list):
                numbers = [i for i, j in enumerate(data_id) if j in number_ids]
                numbers_cnt = 0
                data_dpe = [0] * len(data_id)
                for group in number_groups:
                    if group[0] == "." * len(group[0]):
                        numbers_cnt += len(group[0])
                        continue

                    start = numbers[numbers_cnt]
                    end = numbers[numbers_cnt + len(group[0]) - 1] + 1
                    corresponding_numbers = data_id[start:end]
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
                event_dpes.append(data_dpe)
            dpes.append(event_dpes)
        return dpes
