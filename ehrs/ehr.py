import sys
import os
import random
import re
import shutil
import glob
import subprocess
import logging

from typing import Union, List

import datetime
import pandas as pd
import numpy as np

from sortedcontainers import SortedList
from tqdm import tqdm

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
        assert self.min_event_size > 0, (
            "--min_event_size could not be negative or zero", self.min_event_size
        )
        assert self.min_event_size <= self.max_event_size, (
            self.min_event_size,
            self.max_event_size,
        )

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

        self._icustay_fname = None
        self._patient_fname = None
        self._admission_fname  = None
        self._diagnosis_fname = None

        self._icustay_key = None
        self._hadm_key = None

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
        if isinstance(obs_size, datetime.timedelta):
            obs_size = obs_size.total_seconds() / 3600
            gap_size = gap_size.total_seconds() / 3600

        icustays = icustays[icustays["LOS"] >= (obs_size + gap_size) / 24]
        icustays = icustays[
            (self.min_age <= icustays["AGE"]) & (icustays["AGE"] <= self.max_age)
        ]

        # we define labels for the readmission task in this step
        # since it requires to observe each next icustays,
        # which would have been excluded in the final cohorts
        icustays.sort_values([self.hadm_key, self.icustay_key], inplace=True)
        if self.first_icu:
            is_readmitted = (
                icustays.groupby(self.hadm_key)[self.icustay_key].count() > 1
            ).astype(int)
            is_readmitted = is_readmitted.to_frame().rename(columns={self.icustay_key: "readmission"})

            icustays = icustays.groupby(self.hadm_key).first().reset_index()
            icustays = icustays.join(is_readmitted, on=self.hadm_key)
        else:
            icustays["readmission"] = 1
            icustays.loc[
                icustays.groupby(self.hadm_key)["INTIME"].idxmax(),
                "readmission"
            ] = 0

        cohorts = icustays

        logger.info(
            "cohorts have been built successfully. Loaded {} cohorts.".format(
                len(cohorts)
            )
        )
        self.save_to_cache(cohorts, self.ehr_name + ".cohorts")

        return cohorts

    # TODO process specific tasks according to user choice?
    def prepare_tasks(self, cohorts=None, cached=False):
        if cohorts is None and cached:
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
            "readmission",
            "INTIME",
            "OUTTIME",
            "DISCHTIME",
            "ICU_DISCHARGE_LOCATION",
            "HOS_DISCHARGE_LOCATION",
        ]].copy()

        # los prediction
        labeled_cohorts["los_3day"] = (cohorts["LOS"] > 3).astype(int)
        labeled_cohorts["los_7day"] = (cohorts["LOS"] > 7).astype(int)

        # mortality prediction
        # if the discharge location of an icustay is 'Death'
        #   & intime + obs_size + gap_size <= dischtime <= intime + obs_size + pred_size
        # it is assigned positive label on the mortality prediction
        labeled_cohorts["mortality"] = (
            (
                (labeled_cohorts["ICU_DISCHARGE_LOCATION"] == "Death")
                | (labeled_cohorts["HOS_DISCHARGE_LOCATION"] == "Death")
            )
            & (
                (
                    labeled_cohorts["INTIME"] + self.obs_size + self.gap_size
                    < labeled_cohorts["DISCHTIME"]
                )
                & (
                    labeled_cohorts["DISCHTIME"]
                    <= labeled_cohorts["INTIME"] + self.obs_size + self.pred_size
                )
            )
        ).astype(int)
        # if the discharge of 'Death' occurs in icu or hospital
        # we retain these cases for the imminent discharge task
        labeled_cohorts["in_icu_mortality"] = (
            labeled_cohorts["ICU_DISCHARGE_LOCATION"] == "Death"
        ).astype(int)
        labeled_cohorts["in_hospital_mortality"] = (
            (labeled_cohorts["ICU_DISCHARGE_LOCATION"] != "Death")
            & (labeled_cohorts["HOS_DISCHARGE_LOCATION"] == "Death")
        ).astype(int)

        # define final acuity prediction task
        labeled_cohorts["final_acuity"] = labeled_cohorts["HOS_DISCHARGE_LOCATION"]
        labeled_cohorts.loc[
            labeled_cohorts["in_icu_mortality"] == 1, "final_acuity"
        ] = "IN_ICU_MORTALITY"
        labeled_cohorts.loc[
            labeled_cohorts["in_hospital_mortality"] == 1, "final_acuity"
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
        is_discharged = (
            (
                labeled_cohorts["INTIME"] + self.obs_size + self.gap_size
                <= labeled_cohorts["DISCHTIME"]
            )
            & (
                labeled_cohorts["DISCHTIME"]
                <= labeled_cohorts["INTIME"] + self.obs_size + self.pred_size
            )
        )
        labeled_cohorts.loc[is_discharged, "imminent_discharge"] = labeled_cohorts.loc[
            is_discharged, "HOS_DISCHARGE_LOCATION"
        ]
        labeled_cohorts.loc[
            is_discharged & (
                (labeled_cohorts["in_icu_mortality"] == 1)
                | (labeled_cohorts["in_hospital_mortality"] == 1)
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
                "OUTTIME",
                "in_icu_mortality",
                "in_hospital_mortality",
                "DISCHTIME",
                "ICU_DISCHARGE_LOCATION",
                "HOS_DISCHARGE_LOCATION"
            ]
        )

        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        logger.info("Done preparing tasks except for diagnosis prediction.")

        return labeled_cohorts

    def prepare_events(self, cohorts, cached=False):
        if cohorts is None and cached:
            cohort_events = self.load_from_cache(self.ehr_name + ".cohorts.labeled.events")
            if cohort_events is not None:
                return cohort_events
            else:
                raise RuntimeError()

        logger.info(
            "Start preparing medical events for each cohort."
        )

        icustays_by_hadm_key = cohorts.groupby(self.hadm_key)[
            self.icustay_key
        ].apply(list)
        icustay_to_intime = dict(
            zip(cohorts[self.icustay_key], cohorts["INTIME"])
        )

        cohorts.drop(columns=["INTIME"], inplace=True)

        cohort_events = {
            id: SortedList() for id in cohorts[self.icustay_key].to_list()
        }

        patterns_for_numeric = re.compile("\d+(\.\d+)*")

        for feat in self.features:
            fname = feat["fname"]
            timestamp_key = feat["timestamp"]
            excludes = feat["exclude"]
            hour_to_offset_unit = None
            obs_size = self.obs_size
            if feat["timeoffsetunit"] == "min":
                hour_to_offset_unit = 60
                obs_size *= 60

            logger.info("{} in progress.".format(fname))

            code_to_descriptions = None
            if "code" in feat:
                code_to_descriptions = {
                    k: pd.read_csv(os.path.join(self.data_dir, v))
                    for k, v in zip(feat["code"], feat["desc"])
                }
                code_to_descriptions = {
                    k: dict(zip(v[k], v[d_k]))
                    for (k, v), d_k in zip(
                        code_to_descriptions.items(), feat["desc_key"]
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

            chunks = pd.read_csv(
                os.path.join(self.data_dir, fname), chunksize=self.chunk_size
            )
            for events in tqdm(chunks):
                events = events.drop(columns=excludes)
                if infer_icustay_from_hadm_key:
                    events = events[events[self.hadm_key].isin(list(icustays_by_hadm_key.keys()))]
                else:
                    events = events[events[self.icustay_key].isin(cohorts[self.icustay_key])]

                if len(events) == 0:
                    continue

                if events[timestamp_key].dtype != int:
                    events[timestamp_key] = pd.to_datetime(
                        events[timestamp_key], infer_datetime_format=True
                    )
                dummy_for_sanity_check = events[timestamp_key].iloc[0]
                if (
                    isinstance(dummy_for_sanity_check, datetime.datetime)
                    and isinstance(self.obs_size, datetime.timedelta)
                ) or (
                    isinstance(dummy_for_sanity_check, (int, np.integer))
                    and isinstance(self.obs_size, (int, np.integer))
                ):
                    pass
                else:
                    raise AssertionError(
                        (type(dummy_for_sanity_check), type(self.obs_size))
                    )

                for _, event in events.iterrows():
                    charttime = event[timestamp_key]
                    if infer_icustay_from_hadm_key:
                        # infer icustay id for the event based on `self.hadm_key`
                        hadm_key_icustays = icustays_by_hadm_key[
                            event[self.hadm_key]
                        ]
                        for icustay_id in hadm_key_icustays:
                            intime = icustay_to_intime[icustay_id]
                            if hour_to_offset_unit is not None:
                                intime *= hour_to_offset_unit
    
                            if intime <= charttime and charttime <= intime + obs_size:
                                event[self.icustay_key] = icustay_id
                                break

                        # which means that the event has no corresponding icustay
                        if self.icustay_key not in event:
                            continue
                    else:
                        intime = icustay_to_intime[event[self.icustay_key]]
                        if hour_to_offset_unit is not None:
                            intime *= hour_to_offset_unit
                        # which means that the event has been charted before / after the icustay
                        if not (intime <= charttime and charttime <= intime + obs_size):
                            continue
                        
                    icustay_id = event[self.icustay_key]
                    if icustay_id in cohort_events:
                        event = event.drop(
                            labels=[self.icustay_key, self.hadm_key, timestamp_key],
                            errors='ignore'
                        )
                        cols = []
                        vals = []
                        # TODO process based on which embedding strategy is adopted: desc-based or code-based
                        for col, val in event.to_dict().items():
                            if pd.isna(val):
                                continue
                            # convert code to description if applicable
                            if (
                                code_to_descriptions is not None
                                and col in code_to_descriptions
                            ):
                                val = code_to_descriptions[col][val]

                            # NOTE if col / val contains numeric, split by digit place
                            val_str = str(val).strip()
                            val = ""
                            prev_end = 0
                            for matched in re.finditer(patterns_for_numeric, val_str):
                                start, end = matched.span()
                                # NOTE round to 4 decimals
                                numeric = float(val_str[start:end])
                                numeric = round(numeric, 4)
                                numeric = str(int(numeric)) if numeric.is_integer() else str(numeric)

                                # possible to duplicate unnecessary white space but it's okay
                                val += val_str[prev_end:start] + " " + numeric + " "
                                prev_end = end
                            val += val_str[prev_end:]

                            col_str = str(col).strip()
                            col = ""
                            prev_end = 0
                            for matched in re.finditer(patterns_for_numeric, col_str):
                                start, end = matched.span()
                                # NOTE round to 4 decimals
                                numeric = float(col_str[start:end])
                                numeric = round(numeric, 4)
                                numeric = str(int(numeric)) if numeric.is_integer() else str(numeric)

                                # possible to duplicate unnecessary white space but it's okay
                                col += col_str[prev_end:start] + " " + numeric + " "
                                prev_end = end
                            col += col_str[prev_end:]

                            cols.append(col)
                            vals.append(val)

                        event_string = (cols, vals)

                        cohort_events[icustay_id].add(
                            (charttime, fname[: -len(self.ext)], event_string)
                        )
        total = len(cohort_events)
        cohort_events = {
            k: list(v) if len(v) <= self.max_event_size else list(v[-self.max_event_size:])
            for k, v in cohort_events.items()
            if len(v) >= self.min_event_size
        }
        skipped = total - len(cohort_events)

        logger.info(
            "Done preparing events for the given cohorts."
            f" Skipped {skipped} cohorts since they have too few"
            " (or no) corresponding medical events."
        )

        self.save_to_cache(
            cohort_events, self.ehr_name + ".cohorts.labeled.events", use_pickle=True
        )

        return cohort_events

    def encode_events(self, cohort_events, cached=False):
        # if cached:
        #     encoded_events = self.load_from_cache(self.ehr_name + ".cohorts.labeled.events.encoded")
        #     if encoded_events is not None:
        #         self.encoded_events = encoded_events
        #         return self.encoded_events
        data_dir = os.path.join(self.dest, "data")

        # XXX special tokens for the time interval (zero, last)
        zero_time_interval = pd.Timedelta(0)
        last_time_interval = pd.Timedelta(-1)

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

        self.add_special_tokens(
            ["[TIME_INT_ZERO]", "[TIME_INT_LAST]"]
            + ["[TIME_INT_{}]".format(i) for i in range(1, len(bins))]
        )

        def _quantize(t: pd.Timedelta):
            if t == zero_time_interval:
                return self.special_tokens_dict["[TIME_INT_ZERO]"]
            elif t == last_time_interval:
                return self.special_tokens_dict["[TIME_INT_LAST]"]
            for i, b in enumerate(bins[1:], start=1):
                if t <= b:
                    return self.special_tokens_dict["[TIME_INT_{}]".format(i)]

            raise AssertionError("???")

        time_intervals = {
            k: map(_quantize, v) for k, v in time_intervals.items()
        }

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)
        tokenizer.add_special_tokens({
            "additional_special_tokens": list(self.special_tokens_dict.values())
        })
        cls_token_id = tokenizer.encode("[CLS]")
        sep_token_id = tokenizer.encode("[SEP]")

        encoded_events = {
            k: [
                (event[1], event[2], t_int)
                for event, t_int in zip(v, time_intervals[k])
            ]
            for k, v in cohort_events.items()
        }

        del collated_timestamps
        del time_intervals_flattened
        del time_intervals

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

        rand = random.Random(self.seed)
        manifest_dir = os.path.join(self.dest, "manifest")

        # TODO digit_place_embeddings = dict()
        token_type_embeddings = dict()

        with open(os.path.join(dest_dir, "train.tsv"), "w") as train_f, open(
            os.path.join(dest_dir, "valid.tsv"), "w") as valid_f, open(
            os.path.join(dest_dir, "test.tsv"), "w"
        ) as test_f:
            print(data_dir, file=train_f)
            print(data_dir, file=valid_f)
            print(data_dir, file=test_f)

            for icustay_id, events_per_icustay in encoded_events.items():
                # make list of table names
                table_names = [x[0] for x in events_per_icustay]
                # tokenize table names
                table_names_tokenized = tokenizer(table_names, add_special_tokens=False).input_ids
                # generate the same shape list filled with self.table_type_id
                table_type_tokens = [[self.table_type_id] * len(x) for x in table_names_tokenized]

                # make list of events: (cols, vals)
                event_strings = [x[1] for x in events_per_icustay] # list of (List[cols], List[vals])

                # tokenize column names
                # cols_tokenized: List[List[List[col_tokens]]]
                #   e.g., cols_tokenized[event_i][col_j]: list of tokens of j-th column names of i-th event
                cols_tokenized = [tokenizer(x[0], add_special_tokens=False).input_ids for x in event_strings]
                # generate the same shape list filled with self.column_type_id
                column_type_tokens = [[[self.column_type_id] * len(y) for y in x] for x in cols_tokenized]

                # tokenize column values
                # vals_tokenized: List[List[List[val_tokens]]]
                #   e.g., vals_tokenized[event_i][col_j]: list of tokens of j-th column values of i-th event
                vals_tokenized = [tokenizer(x[1], add_special_tokens=False).input_ids for x in event_strings]
                # generate the same shape list filled with self.value_type_id
                value_type_tokens = [[[self.value_type_id] * len(y) for y in x] for x in vals_tokenized]

                # linearize column names & values as well as token types
                # events_tokenized: List[List[tokens]]
                #   e.g., events_tokenized[event_i]: list of tokens of i-th event such as
                #       [col0, val0, col1, val1, ...]
                # col_val_type_tokens: the same shape list with events_tokenized filled
                #   with corresponding token type ids
                events_tokenized = []
                col_val_type_tokens = []
                for cols, vols, col_type_tokens, val_type_tokens in zip(
                    cols_tokenized, vals_tokenized, column_type_tokens, value_type_tokens
                ):
                    event_tokenized = []
                    col_val_type_token = []
                    for col, vol, col_type_token, val_type_token in zip(
                        cols, vols, col_type_tokens, val_type_tokens
                    ):
                        event_tokenized.extend(col + vol)
                        col_val_type_token.extend(col_type_token + val_type_token)

                    events_tokenized.append(event_tokenized)
                    col_val_type_tokens.append(col_val_type_token)

                # make list of time intervals
                time_intervals = [x[2] for x in events_per_icustay]
                # tokenize time intervals
                # this should yield a corresponding special token id for each time interval
                time_intervals_tokenized = tokenizer(time_intervals, add_special_tokens=False).input_ids
                # generate the same shape list filled with self.timeint_type_id
                timeint_type_tokens = [[self.timeint_type_id] for _ in time_intervals_tokenized]

                # encoded_events[icustay_id]: sequence of tokenized events [table_name, event_strings, time_interval]
                #   List[List[tokens]], where encoded_events[icustay_id][i] is a sequence of input ids for i-th event of icustay_id
                # token_type_embeddings[ocustay]: corresponding token type ids
                # NOTE prepend [CLS], append [SEP] here (temporarily only for hierarchical)
                encoded_events[icustay_id] = [
                    [cls_token_id] + table_name + event + time_interval + [sep_token_id]
                    for table_name, event, time_interval
                    in zip(table_names_tokenized, events_tokenized, time_intervals_tokenized)
                ]
                token_type_embeddings[icustay_id] = [
                    [self.cls_type_id] + table_type_token + col_val_type_token + timeint_type_token + [self.sep_type_id]
                    for table_type_token, col_val_type_token, timeint_type_token
                    in zip(table_type_tokens, col_val_type_tokens, timeint_type_tokens)
                ]
                # TODO define digit_place_embeddings[icustay_id] for encoded_events[icustay_id][...] here

                # TODO if flattened input, process more ...
                # encoded_events = {
                #     k: " ".join(v) for k, v in encoded_events.items()
                # }

                # save to data directory
                np.save(
                    os.path.join(data_dir, "input_ids", icustay_id),
                    np.array(encoded_events[icustay_id], dtype=object)
                )
                np.save(
                    os.path.join(data_dir, "token_type_ids", icustay_id),
                    np.array(token_type_embeddings[icustay_id], dtype=object)
                )
                # TODO save digit place embeddings
                # np.save(
                #     os.path.join(data_dir, "digit_place_ids", icustay_id),
                #     np.array(digit_place_embeddings[icustay_id], dtype=object)
                # )

                # manifest
                p = rand.random()
                if self.valid_percent * 2 <= p:
                    dest = train_f
                elif self.valid_percent <= p:
                    dest = valid_f
                else:
                    dest = test_f
                
                print(
                    f"{icustay_id}\t{len(encoded_events[icustay_id])}", file=dest
                )

        logger.info("Done encoding events.")

        # NOTE return hierarchical input
        return encoded_events

    def run_pipeline(self) -> None:
        cohorts = self.build_cohorts(cached=self.cache)
        labeled_cohorts = self.prepare_tasks(cohorts, cached=self.cache)
        cohort_events = self.prepare_events(labeled_cohorts, cached=self.cache)
        encoded_events = self.encode_events(cohort_events, cached=self.cache)

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
        """
        raise NotImplementedError()

    def is_compatible(self, icustays):
        checklist = [
            self.hadm_key,
            self.icustay_key,
            "LOS",
            "AGE",
            "INTIME",
            "OUTTIME",
            "DISCHTIME",
            "ICU_DISCHARGE_LOCATION",
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

    def infer_data_extension(self, threshold=10) -> str:
        ext = None
        if len(glob.glob(os.path.join(self.data_dir, "*.csv.gz"))) >= threshold:
            ext = ".csv.gz"
        elif len(glob.glob(os.path.join(self.data_dir, ".csv"))) >= threshold:
            ext = ".csv"

        if ext is None:
            raise AssertionError(
                "Cannot infer data extension from {}. ".format(self.data_dir)
                + "Please provide --ext explicitly."
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext

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

        fnames = glob.glob(os.path.join(dest, output_dir, "**/*.csv.gz"), recursive=True)
        for fname in fnames:
            os.rename(fname, os.path.join(dest, os.path.basename(fname)))

        shutil.rmtree(os.path.join(dest, output_dir.split("/")[0]))

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