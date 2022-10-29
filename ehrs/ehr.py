import sys
import os
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

        self.bins = cfg.bins

        self.special_tokens_dict = dict()
        self.max_special_tokens = 100

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
                self.cohorts = cohorts
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

        self.cohorts = icustays

        logger.info(
            "cohorts have been built successfully. Loaded {} cohorts.".format(
                len(self.cohorts)
            )
        )
        self.save_to_cache(self.cohorts, self.ehr_name + ".cohorts")

        return self.cohorts

    # TODO process specific tasks according to user choice?
    def prepare_tasks(self, cohorts=None, cached=False):
        if cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled")
            if labeled_cohorts is not None:
                self.labeled_cohorts = labeled_cohorts
                return labeled_cohorts

        logger.info(
            "Start labeling cohorts for predictive tasks."
        )

        if cohorts is None:
            cohorts = self.cohorts

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
                "INTIME",
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

    def prepare_events(self, cohorts=None, cached=False):
        if cached:
            cohort_events = self.load_from_cache(self.ehr_name + ".cohorts.labeled.events")
            if cohort_events is not None:
                self.cohort_events = cohort_events
                return self.cohort_events

        logger.info(
            "Start preparing medical events for each cohort."
        )

        if hasattr(self, "icustays"):
            icustays = self.icustays
        else:
            icustays = pd.read_csv(os.path.join(self.data_dir, self.icustays))
            icustays = self.make_compatible(icustays)

        icustays_by_hadm_key = icustays.groupby(self.hadm_key)[
            self.icustay_key
        ].apply(list)
        icustay_to_intime = dict(
            zip(icustays[self.icustay_key], icustays["INTIME"])
        )
        icustay_to_outtime = dict(
            zip(icustays[self.icustay_key], icustays["OUTTIME"])
        )

        if cohorts is None:
            cohorts = self.labeled_cohorts

        cohort_events = {
            id: SortedList() for id in cohorts[self.icustay_key].to_list()
        }

        for feat in self.features:
            fname = feat["fname"]
            timestamp_key = feat["timestamp"]
            excludes = feat["exclude"]
            hour_to_offset_unit = None
            if feat["timeoffsetunit"] == "min":
                hour_to_offset_unit = 60

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
                events = events.drop(columns=excludes)
                if infer_icustay_from_hadm_key:
                    events = events[~events[self.hadm_key].isna()]
                else:
                    events = events[~events[self.icustay_key].isna()]

                for _, event in events.iterrows():
                    charttime = event[timestamp_key]
                    if infer_icustay_from_hadm_key:
                        if event[self.hadm_key] not in icustays_by_hadm_key:
                            continue

                        # infer icustay id for the event based on `self.hadm_key`
                        hadm_key_icustays = icustays_by_hadm_key[
                            event[self.hadm_key]
                        ]
                        for icustay_id in hadm_key_icustays:
                            intime = icustay_to_intime[icustay_id]
                            outtime = icustay_to_outtime[icustay_id]
                            if hour_to_offset_unit is not None:
                                intime *= hour_to_offset_unit
                                outtime *= hour_to_offset_unit

                            if intime <= charttime and charttime <= outtime:
                                event[self.icustay_key] = icustay_id
                                break

                        # which means that the event has no corresponding icustay
                        if self.icustay_key not in event:
                            continue
                    else:
                        if event[self.icustay_key] not in cohort_events:
                            continue

                        intime = icustay_to_intime[event[self.icustay_key]]
                        outtime = icustay_to_outtime[event[self.icustay_key]]
                        if hour_to_offset_unit is not None:
                            intime *= hour_to_offset_unit
                            outtime *= hour_to_offset_unit
                        # which means that the event has been charted before / after the icustay
                        if not (intime <= charttime and charttime <= outtime):
                            continue

                    icustay_id = event[self.icustay_key]
                    if icustay_id in cohort_events:
                        event = event.drop(
                            labels=[self.icustay_key, self.hadm_key, timestamp_key],
                            errors='ignore'
                        )
                        event_string = []
                        # TODO process based on which embedding strategy is adopted: desc-based or code-based
                        for name, val in event.to_dict().items():
                            if pd.isna(val):
                                continue
                            # convert code to description if applicable
                            if (
                                code_to_descriptions is not None
                                and name in code_to_descriptions
                            ):
                                val = code_to_descriptions[name][val]

                            if isinstance(val, (float, np.floating)):
                                # NOTE round to 4 decimals
                                val = round(val, 4)

                            val = str(val).strip()
                            event_string.append(" ".join([name, val]))
                        event_string = " ".join(event_string)

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

        self.cohort_events = cohort_events
        self.save_to_cache(
            self.cohort_events, self.ehr_name + ".cohorts.labeled.events", use_pickle=True
        )

        return cohort_events

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