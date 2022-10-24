import os
import sys
import logging
import glob
import pickle
import subprocess
import shutil
from sortedcontainers import SortedList

from datetime import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm

from ehrs import register_ehr, EHR
from utils.utils import get_physionet_dataset, get_ccs

logger = logging.getLogger(__name__)


@register_ehr("mimiciii")
class MIMICIII(EHR):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.data_dir = cfg.data
        self.ccs_path = cfg.ccs

        cache_dir = os.path.expanduser("~/.cache/ehr")
        physionet_file_path = "mimiciii/1.4/"

        self.data_dir = get_physionet_dataset(cfg, physionet_file_path, cache_dir)
        self.ccs_path = get_ccs(cfg, cache_dir)

        self.ext = ".csv.gz"
        if len(glob.glob(os.path.join(self.data_dir, "*" + self.ext))) != 26:
            self.ext = ".csv"
            if len(glob.glob(os.path.join(self.data_dir, "*" + self.ext))) != 26:
                raise AssertionError(
                    "Provided data directory is not correct. Please check if --data is correct. "
                    "--data: {}".format(self.data_dir)
                )

        self.icustays = "ICUSTAYS" + self.ext
        self.patients = "PATIENTS" + self.ext
        self.admissions = "ADMISSIONS" + self.ext
        self.diagnoses = "DIAGNOSES_ICD" + self.ext

        # XXX more features? user choice?
        self.features = [
            {
                "fname": "LABEVENTS" + self.ext,
                "type": "lab",
                "timestamp": "CHARTTIME",
                "exclude": ["ROW_ID", "SUBJECT_ID"],
                "code": ["ITEMID"],
                "desc": ["D_LABITEMS" + self.ext],
                "desc_key": ["LABEL"],
            },
            {
                "fname": "PRESCRIPTIONS" + self.ext,
                "type": "med",
                "timestamp": "STARTDATE",
                "exclude": ["ENDDATE", "GSN", "NDC", "ROW_ID", "SUBJECT_ID"],
            },
            {
                "fname": "INPUTEVENTS_MV" + self.ext,
                "type": "inf",
                "timestamp": "STARTTIME",
                "exclude": [
                    "ENDTIME",
                    "STORETIME",
                    "CGID",
                    "ORDERID",
                    "LINKORDERID",
                    "ROW_ID",
                    "SUBJECT_ID",
                ],
                "code": ["ITEMID"],
                "desc": ["D_ITEMS" + self.ext],
                "desc_key": ["LABEL"],
            },
            {
                "fname": "INPUTEVENTS_CV" + self.ext,
                "type": "inf",
                "timestamp": "CHARTTIME",
                "exclude": [
                    "STORETIME",
                    "CGID",
                    "ORDERID",
                    "LINKORDERID",
                    "ROW_ID",
                    "SUBJECT_ID",
                ],
                "code": ["ITEMID"],
                "desc": ["D_ITEMS" + self.ext],
                "desc_key": ["LABEL"],
            },
        ]
        # TODO variablize? or leave constants as is
        self.icustay_key = "ICUSTAY_ID"
        self.icustay_start_key = "INTIME"
        self.icustay_end_key = "OUTTIME"
        self.second_key = "HADM_ID"

        self.max_event_size = (
            cfg.max_event_size if cfg.max_event_size is not None else sys.maxsize
        )
        self.min_event_size = (
            cfg.min_event_size if cfg.min_event_size is not None else 0
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

    def build_cohort(self):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patients))
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustays))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admissions))

        icustays = icustays[icustays["FIRST_CAREUNIT"] == icustays["LAST_CAREUNIT"]]
        icustays = icustays[icustays["LOS"] >= (self.obs_size + self.gap_size) / 24]
        icustays = icustays.drop(columns=["ROW_ID"])
        icustays[self.icustay_start_key] = pd.to_datetime(
            icustays[self.icustay_start_key], infer_datetime_format=True
        )
        icustays[self.icustay_end_key] = pd.to_datetime(
            icustays[self.icustay_end_key], infer_datetime_format=True
        )

        patients["DOB"] = pd.to_datetime(patients["DOB"], infer_datetime_format=True)
        patients = patients.drop(columns=["ROW_ID"])

        patients_with_icustays = patients[
            patients["SUBJECT_ID"].isin(icustays["SUBJECT_ID"])
        ]
        patients_with_icustays = icustays.merge(
            patients_with_icustays, on="SUBJECT_ID", how="left"
        )

        def calculate_age(birth: datetime, now: datetime):
            age = now.year - birth.year
            if now.month < birth.month:
                age -= 1
            elif (now.month == birth.month) and (now.day < birth.day):
                age -= 1

            return age

        patients_with_icustays["AGE"] = patients_with_icustays.apply(
            lambda x: calculate_age(x["DOB"], x[self.icustay_start_key]), axis=1
        )
        patients_with_icustays = patients_with_icustays[
            (self.min_age <= patients_with_icustays["AGE"])
            & (patients_with_icustays["AGE"] <= self.max_age)
        ]

        # merge with admissions to get discharge information
        patients_with_icustays = pd.merge(
            patients_with_icustays.reset_index(drop=True),
            admissions[["HADM_ID", "DISCHARGE_LOCATION", "DEATHTIME", "DISCHTIME"]],
            how="left",
            on="HADM_ID",
        )

        # we define labels for the readmission task in this step
        # since it requires to observe each next icustays,
        # which would have been excluded in the final cohorts
        if self.first_icu:
            # check if each HADM_ID has multiple icustays
            is_readmitted = patients_with_icustays.groupby("HADM_ID")[
                "ICUSTAY_ID"
            ].count()
            is_readmitted = (
                (is_readmitted > 1)
                .astype(int)
                .to_frame()
                .rename(columns={"ICUSTAY_ID": "readmission"})
            )

            # take the first icustays for each HADM_ID
            patients_with_icustays = patients_with_icustays.loc[
                patients_with_icustays.groupby("HADM_ID")[
                    self.icustay_start_key
                ].idxmin()
            ]
            # assign an appropriate label for the readmission task
            patients_with_icustays = patients_with_icustays.join(
                is_readmitted, on="HADM_ID"
            )
        else:
            patients_with_icustays["readmission"] = 1
            # the last icustay for each HADM_ID means that they have no icu readmission
            patients_with_icustays.loc[
                patients_with_icustays.groupby("HADM_ID")[
                    self.icustay_start_key
                ].idxmax(),
                "readmission",
            ] = 0

        patients_with_icustays["DEATHTIME"] = pd.to_datetime(
            patients_with_icustays["DEATHTIME"], infer_datetime_format=True
        )
        # XXX DISCHTIME --> HOSPITAL DISCHARGE TIME
        patients_with_icustays["DISCHTIME"] = pd.to_datetime(
            patients_with_icustays["DISCHTIME"], infer_datetime_format=True
        )

        self.cohort = patients_with_icustays
        logger.info(
            "cohort has been built successfully. Loaded {} cohorts.".format(
                len(self.cohort)
            )
        )
        patients_with_icustays.to_pickle(os.path.join(self.dest, "mimiciii.cohorts"))

        return patients_with_icustays

    def prepare_tasks(self):
        # readmission prediction
        labeled_cohort = self.cohort[["HADM_ID", "ICUSTAY_ID", "readmission"]].copy()

        # los prediction
        labeled_cohort["los_3day"] = (self.cohort["LOS"] > 3).astype(int)
        labeled_cohort["los_7day"] = (self.cohort["LOS"] > 7).astype(int)

        # mortality prediction
        # filter out dead patients
        dead_patients = self.cohort[~self.cohort["DEATHTIME"].isna()]
        dead_patients = dead_patients[
            [
                self.icustay_key,
                self.icustay_start_key,
                self.icustay_end_key,
                "DEATHTIME",
            ]
        ].copy()

        # if intime + obs_size + gap_size <= deathtime <= intime + obs_size + pred_size
        # it is assigned positive label on the mortality prediction
        is_dead = (
            (
                (
                    dead_patients[self.icustay_start_key]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.gap_size, unit="h")
                )
                <= dead_patients["DEATHTIME"]
            )
            & (
                dead_patients["DEATHTIME"]
                <= (
                    dead_patients[self.icustay_start_key]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.pred_size, unit="h")
                )
            )
        ).astype(int)
        dead_patients["mortality"] = np.array(is_dead)

        # if icu intime < deathtime <= icu outtime
        # we also retain this case as in_icu_mortality for the imminent discharge task
        is_dead_in_icu = (
            dead_patients[self.icustay_start_key] < dead_patients["DEATHTIME"]
        ) & (dead_patients["DEATHTIME"] <= dead_patients[self.icustay_end_key])
        dead_patients["in_icu_mortality"] = np.array(is_dead_in_icu.astype(int))

        labeled_cohort = pd.merge(
            labeled_cohort.reset_index(drop=True),
            dead_patients[[self.icustay_key, "mortality", "in_icu_mortality"]],
            on=self.icustay_key,
            how="left",
        ).reset_index(drop=True)
        labeled_cohort["mortality"] = labeled_cohort["mortality"].fillna(0).astype(int)
        labeled_cohort["in_icu_mortality"] = (
            labeled_cohort["in_icu_mortality"].fillna(0).astype(int)
        )

        # join with self.cohort to get information needed for imminent discharge task
        labeled_cohort = labeled_cohort.join(
            self.cohort[
                [
                    self.icustay_key,
                    self.icustay_start_key,
                    "DISCHTIME",
                    "DISCHARGE_LOCATION",
                ]
            ].set_index("ICUSTAY_ID"),
            on=self.icustay_key,
        )

        # if an icustay is DEAD/EXPIRED, but not in_icu_mortality, then it is in_hospital_mortality
        labeled_cohort["in_hospital_mortality"] = 0
        labeled_cohort.loc[
            (labeled_cohort["DISCHARGE_LOCATION"] == "DEAD/EXPIRED")
            & (labeled_cohort["in_icu_mortality"] == 0),
            "in_hospital_mortality",
        ] = 1
        # define new class whose discharge location was 'DEAD/EXPIRED'
        labeled_cohort.loc[
            labeled_cohort["in_icu_mortality"] == 1, "DISCHARGE_LOCATION"
        ] = "IN_ICU_MORTALITY"
        labeled_cohort.loc[
            labeled_cohort["in_hospital_mortality"] == 1, "DISCHARGE_LOCATION"
        ] = "IN_HOSPITAL_MORTALITY"

        # define final acuity prediction task
        logger.info("Fincal Acuity Categories")
        logger.info(
            labeled_cohort["DISCHARGE_LOCATION"].astype("category").cat.categories
        )
        labeled_cohort["final_acuity"] = (
            labeled_cohort["DISCHARGE_LOCATION"].astype("category").cat.codes
        )

        # define imminent discharge prediction task
        is_discharged = (
            (
                (
                    labeled_cohort[self.icustay_start_key]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.gap_size, unit="h")
                )
                <= labeled_cohort["DISCHTIME"]
            )
            & (
                labeled_cohort["DISCHTIME"]
                <= (
                    labeled_cohort[self.icustay_start_key]
                    + pd.Timedelta(self.obs_size, unit="h")
                    + pd.Timedelta(self.pred_size, unit="h")
                )
            )
        ).astype(bool)
        labeled_cohort.loc[is_discharged, "imminent_discharge"] = labeled_cohort[
            is_discharged
        ]["DISCHARGE_LOCATION"]
        labeled_cohort.loc[~is_discharged, "imminent_discharge"] = "No Discharge"
        labeled_cohort.loc[
            (labeled_cohort["imminent_discharge"] == "IN_HOSPITAL_MORTALITY")
            | (labeled_cohort["imminent_discharge"] == "IN_ICU_MORTALITY"),
            "imminent_discharge",
        ] = "Death"
        logger.info("Immminent Discharge Categories")
        logger.info(
            labeled_cohort["imminent_discharge"].astype("category").cat.categories
        )
        labeled_cohort["imminent_discharge"] = (
            labeled_cohort["imminent_discharge"].astype("category").cat.codes
        )

        # drop unnecessary columns
        labeled_cohort = labeled_cohort.drop(
            columns=[
                self.icustay_start_key,
                "in_icu_mortality",
                "in_hospital_mortality",
                "DISCHTIME",
                "DISCHARGE_LOCATION",
            ]
        )

        # define diagnosis prediction task
        diagnoses = pd.read_csv(os.path.join(self.data_dir, self.diagnoses))

        diagnoses_with_cohort = diagnoses[
            diagnoses["HADM_ID"].isin(labeled_cohort["HADM_ID"])
        ]
        diagnoses_with_cohort = (
            diagnoses_with_cohort.groupby("HADM_ID")["ICD9_CODE"].apply(list).to_frame()
        )
        labeled_cohort = labeled_cohort.join(diagnoses_with_cohort, on="HADM_ID")

        ccs_dx = pd.read_csv(self.ccs_path)
        ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
        ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
        lvl1 = {
            x: y for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()
        }

        labeled_cohort.dropna(subset=["ICD9_CODE"], inplace=True)
        labeled_cohort["diagnosis"] = labeled_cohort["ICD9_CODE"].map(
            lambda dxs: list(set([lvl1[dx] for dx in dxs if dx in lvl1]))
        )
        labeled_cohort.dropna(subset=["diagnosis"], inplace=True)
        labeled_cohort = labeled_cohort.drop(columns=["ICD9_CODE"])

        self.labeled_cohort = labeled_cohort
        logger.info("Done preparing tasks for the given cohorts")

        labeled_cohort.to_pickle(os.path.join(self.dest, "mimiciii.cohorts.labeled"))

        return labeled_cohort

    def prepare_events(self):
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustays))
        icustays[self.icustay_start_key] = pd.to_datetime(
            icustays[self.icustay_start_key], infer_datetime_format=True
        )
        icustays[self.icustay_end_key] = pd.to_datetime(
            icustays[self.icustay_end_key], infer_datetime_format=True
        )
        icustays_by_second_key = icustays.groupby(self.second_key)[
            self.icustay_key
        ].apply(list)
        icustay_to_intime = dict(
            zip(icustays[self.icustay_key], icustays[self.icustay_start_key])
        )
        icustay_to_outtime = dict(
            zip(icustays[self.icustay_key], icustays[self.icustay_end_key])
        )

        icustay_events = {
            id: SortedList() for id in self.labeled_cohort[self.icustay_key].to_list()
        }

        for feat in self.features:
            fname = feat["fname"]
            timestamp_key = feat["timestamp"]
            excludes = feat["exclude"]
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

            infer_icustay_from_second_key = False
            columns = pd.read_csv(
                os.path.join(self.data_dir, fname), index_col=0, nrows=0
            ).columns.to_list()
            if self.icustay_key not in columns:
                infer_icustay_from_second_key = True
                if self.second_key not in columns:
                    raise AssertionError(
                        "{} doesn't have one of these columns: {}".format(
                            fname, [self.icustay_key, self.second_key]
                        )
                    )

            chunks = pd.read_csv(
                os.path.join(self.data_dir, fname), chunksize=self.chunk_size
            )
            for events in tqdm(chunks):
                events[timestamp_key] = pd.to_datetime(
                    events[timestamp_key], infer_datetime_format=True
                )
                events = events.drop(columns=excludes)
                if infer_icustay_from_second_key:
                    events = events[~events[self.second_key].isna()]
                else:
                    events = events[~events[self.icustay_key].isna()]

                for _, event in events.iterrows():
                    charttime = event[timestamp_key]
                    if infer_icustay_from_second_key:
                        if event[self.second_key] not in icustays_by_second_key:
                            continue

                        # infer icustay id for the event based on `self.second_key`
                        second_key_icustays = icustays_by_second_key[
                            event[self.second_key]
                        ]
                        for icustay_id in second_key_icustays:
                            intime = icustay_to_intime[icustay_id]
                            outtime = icustay_to_outtime[icustay_id]
                            if intime <= charttime and charttime <= outtime:
                                event[self.icustay_key] = icustay_id
                                break

                        # which means that the event has no corresponding icustay
                        if self.icustay_key not in event:
                            continue
                    else:
                        intime = icustay_to_intime[event[self.icustay_key]]
                        outtime = icustay_to_outtime[event[self.icustay_key]]
                        # which means that the event has been charted before / after the icustay
                        if not (intime <= charttime and charttime <= outtime):
                            continue

                    icustay_id = event[self.icustay_key]
                    if icustay_id in icustay_events:
                        event = event.drop(
                            labels=[self.icustay_key, self.second_key, timestamp_key]
                        )
                        event_string = []
                        # TODO check which embedding strategy is adopted, desc-based or code-based
                        for name, val in event.to_dict().items():
                            if pd.isna(val):
                                continue
                            # convert code to description if needed
                            if (
                                code_to_descriptions is not None
                                and name in code_to_descriptions
                            ):
                                val = code_to_descriptions[name][val]
                            val = str(val)
                            event_string.append(" ".join([name, val]))
                        event_string = " ".join(event_string)

                        icustay_events[icustay_id].add(
                            (charttime, fname[: -len(self.ext)], event_string)
                        )
        # TODO put time-gap
        """
        2. based on the sorted list, define time offset --> (TABLE_NAME, events_string, time-gap)
        """

        self.cohort_events = icustay_events

        logger.info("Done preparing events for the given labeled cohorts.")

        with open(
            os.path.join(self.dest, "mimiciii.cohorts.labeled.events"), "wb"
        ) as f:
            pickle.dump(icustay_events, f)

        return icustay_events

    def encode_events(self):
        """
        encode event strings
        """
        pass

    # def run_pipeline(self):
    #     ...
