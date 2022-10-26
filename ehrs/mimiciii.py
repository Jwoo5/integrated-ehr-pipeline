import os
import logging
import glob
import pickle
from sortedcontainers import SortedList

from datetime import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm

from ehrs import register_ehr, EHR

logger = logging.getLogger(__name__)

@register_ehr("mimiciii")
class MIMICIII(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.obs_size = pd.Timedelta(self.obs_size, unit="h")
        self.gap_size = pd.Timedelta(self.gap_size, unit="h")
        self.pred_size = pd.Timedelta(self.pred_size, unit="h")

        self.ehr_name = "mimiciii"

        if self.data_dir is None:
            self.data_dir = os.path.join(self.cache_dir, self.ehr_name)

            if not os.path.exists(self.data_dir):
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )
                self.download_ehr_from_url(
                    url="https://physionet.org/files/mimiciii/1.4/",
                    dest=self.data_dir
                )

        logger.info("Data directory is set to {}".format(self.data_dir))

        if self.ccs_path is None:
            self.ccs_path = os.path.join(self.cache_dir, "ccs_multi_dx_tool_2015.csv")

            if not os.path.exists(self.ccs_path):
                logger.info(
                    "`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet."
                )
                self.download_ccs_from_url(self.cache_dir)

        if self.ext is None:
            self.ext = self.infer_data_extension()

        # constants
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
        self.icustay_key = "ICUSTAY_ID"
        self.hadm_key = "HADM_ID"

    def build_cohorts(self, cached=False):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patients))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admissions))
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustays))

        # prepare icustays according to the appropriate format
        icustays = icustays[icustays["FIRST_CAREUNIT"] == icustays["LAST_CAREUNIT"]]
        icustays["INTIME"] = pd.to_datetime(
            icustays["INTIME"], infer_datetime_format=True
        )
        icustays["OUTTIME"] = pd.to_datetime(
            icustays["OUTTIME"], infer_datetime_format=True
        )
        icustays = icustays.drop(columns=["ROW_ID"])

        # merge icustays with patients to get DOB
        patients["DOB"] = pd.to_datetime(patients["DOB"], infer_datetime_format=True)
        patients = patients[
            patients["SUBJECT_ID"].isin(icustays["SUBJECT_ID"])
        ]
        patients = patients.drop(columns=["ROW_ID"])[["SUBJECT_ID", "DOB"]]
        icustays = icustays.merge(patients, on="SUBJECT_ID", how="left")

        def calculate_age(birth: datetime, now: datetime):
            age = now.year - birth.year
            if now.month < birth.month:
                age -= 1
            elif (now.month == birth.month) and (now.day < birth.day):
                age -= 1

            return age

        icustays["AGE"] = icustays.apply(
            lambda x: calculate_age(x["DOB"], x["INTIME"]), axis=1
        )

        # merge with admissions to get discharge information
        icustays = pd.merge(
            icustays.reset_index(drop=True),
            admissions[["HADM_ID", "DISCHARGE_LOCATION", "DEATHTIME", "DISCHTIME"]],
            how="left",
            on="HADM_ID",
        )
        icustays["DISCHARGE_LOCATION"].replace("DEAD/EXPIRED", "Death", inplace=True)

        # icustays["DEATHTIME"] = pd.to_datetime(
        #     icustays["DEATHTIME"], infer_datetime_format=True
        # )
        # XXX DISCHTIME --> HOSPITAL DISCHARGE TIME?
        icustays["DISCHTIME"] = pd.to_datetime(
            icustays["DISCHTIME"], infer_datetime_format=True
        )

        icustays["ICU_DISCHARGE_LOCATION"] = np.nan
        is_discharged_in_icu = (
            (icustays["INTIME"] < icustays["DISCHTIME"])
            & (icustays["DISCHTIME"] <= icustays["OUTTIME"])
        )
        icustays.loc[
            is_discharged_in_icu, "ICU_DISCHARGE_LOCATION"
        ] = icustays.loc[is_discharged_in_icu, "DISCHARGE_LOCATION"]
        icustays.loc[
            ~is_discharged_in_icu, "HOS_DISCHARGE_LOCATION"
        ] = icustays.loc[~is_discharged_in_icu, "DISCHARGE_LOCATION"]


        cohorts = super().build_cohorts(icustays, cached=cached)

        return cohorts

    def prepare_tasks(self, cohorts=None, cached=False):
        labeled_cohorts = super().prepare_tasks(cohorts, cached)

        # define diagnosis prediction task
        diagnoses = pd.read_csv(os.path.join(self.data_dir, self.diagnoses))

        diagnoses_with_cohorts = diagnoses[
            diagnoses[self.hadm_key].isin(labeled_cohorts[self.hadm_key])
        ]
        diagnoses_with_cohorts = (
            diagnoses_with_cohorts.groupby(self.hadm_key)["ICD9_CODE"].apply(list).to_frame()
        )
        labeled_cohorts = labeled_cohorts.join(diagnoses_with_cohorts, on="HADM_ID")

        ccs_dx = pd.read_csv(self.ccs_path)
        ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
        ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
        lvl1 = {
            x: y for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()
        }

        labeled_cohorts.dropna(subset=["ICD9_CODE"], inplace=True)
        labeled_cohorts["diagnosis"] = labeled_cohorts["ICD9_CODE"].map(
            lambda dxs: list(set([lvl1[dx] for dx in dxs if dx in lvl1]))
        )
        labeled_cohorts.dropna(subset=["diagnosis"], inplace=True)
        labeled_cohorts = labeled_cohorts.drop(columns=["ICD9_CODE"])

        self.labeled_cohorts = labeled_cohorts
        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        logger.info("Done preparing tasks for the given cohorts")

        return labeled_cohorts

    def encode_events(self):
        """
        encode event strings
        # TODO put time-gap
        2. based on the sorted list, define time offset --> (events_string, time-gap)
        3. tokenize events_string 
        4. final output: [CLS] tokenized(events_string) [time_token] [SEP]
        """
        collated_timestamps = {
            icustay_id: [event[0] for event in events]
            for icustay_id, events in self.cohort_events.items()
        }
        # calc time interval to the next event (i, i+1)
        time_intervals = {
            icustay_id: np.subtract(t, [t[0]] + t[:-1])
            for icustay_id, t in collated_timestamps.items()
        }
        del collated_timestamps
        breakpoint()
        # exclude time_offset == 0 here
        time_intervals_flattened = np.sort(
            np.concatenate([v for v in time_intervals.values()])
        )

        # percentile --> quantile
        # exclude time_offset == 0 --> bucket #0
        # bins = [0, 5, 10, 15, ..., 100]
        # x <= 5 --> bucket #1
        # 5 <= x < 10 --> bucket #2
        # ...
        # 95 <= x --> bucket # 20
        breakpoint()
        bins = np.arange(self.bins + 1)
        bins = bins * 100 / bins[-1]
        bins.sort()
        breakpoint()

        bins = np.percentile(time_intervals_flattened, bins)
        

        breakpoint()
        # time_offsets = 

        """
        tokenize
        dpe
        token_type_embedding
        """

        pass

    # def run_pipeline(self):
    #     ...
