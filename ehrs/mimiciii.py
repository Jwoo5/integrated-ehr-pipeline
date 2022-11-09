import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import glob

from ehrs import register_ehr, EHR

logger = logging.getLogger(__name__)

@register_ehr("mimiciii")
class MIMICIII(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

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
        self._icustay_fname = "ICUSTAYS" + self.ext
        self._patient_fname = "PATIENTS" + self.ext
        self._admission_fname = "ADMISSIONS" + self.ext
        self._diagnosis_fname = "DIAGNOSES_ICD" + self.ext

        # XXX more features? user choice?
        self.tables = [
            {
                "fname": "LABEVENTS" + self.ext,
                "timestamp": "CHARTTIME",
                "timeoffsetunit": "abs",
                "exclude": ["ROW_ID", "SUBJECT_ID"],
                "code": ["ITEMID"],
                "desc": ["D_LABITEMS" + self.ext],
                "desc_key": ["LABEL"],
            },
            {
                "fname": "PRESCRIPTIONS" + self.ext,
                "timestamp": "STARTDATE",
                "timeoffsetunit": "abs",
                "exclude": ["ENDDATE", "GSN", "NDC", "ROW_ID", "SUBJECT_ID"],
            },
            {
                "fname": "INPUTEVENTS_MV" + self.ext,
                "timestamp": "STARTTIME",
                "timeoffsetunit": "abs",
                "exclude": [
                    "ENDTIME",
                    "STORETIME",
                    "CGID",
                    "ORDERID",
                    "LINKORDERID",
                    "ROW_ID",
                    "SUBJECT_ID",
                    "CONTINUEINNEXTDEPT",
                    "CANCELREASON",
                    "STATUSDESCRIPTION",
                    "COMMENTS_CANCELEDBY",
                    "COMMENTS_DATE"
                ],
                "code": ["ITEMID"],
                "desc": ["D_ITEMS" + self.ext],
                "desc_key": ["LABEL"],
            },
            {
                "fname": "INPUTEVENTS_CV" + self.ext,
                "timestamp": "CHARTTIME",
                "timeoffsetunit": "abs",
                "exclude": [
                    "STORETIME",
                    "CGID",
                    "ORDERID",
                    "LINKORDERID",
                    "ROW_ID",
                    "STOPPED",
                    "SUBJECT_ID",
                ],
                "code": ["ITEMID"],
                "desc": ["D_ITEMS" + self.ext],
                "desc_key": ["LABEL"],
            },
        ]
        self._icustay_key = "ICUSTAY_ID"
        self._hadm_key = "HADM_ID"
        self._patient_key = "SUBJECT_ID"

        self._determine_first_icu = "INTIME"

    def build_cohorts(self, cached=False):
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustay_fname))

        icustays = self.make_compatible(icustays)
        self.icustays = icustays

        cohorts = super().build_cohorts(icustays, cached=cached)

        return cohorts

    def prepare_tasks(self, cohorts, cached=False):
        if cohorts is None and cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled.dx")
            if labeled_cohorts is not None:
                return labeled_cohorts

        labeled_cohorts = super().prepare_tasks(cohorts, cached)

        logger.info(
            "Start labeling cohorts for diagnosis prediction."
        )

        # define diagnosis prediction task
        diagnoses = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))

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

        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled.dx")

        logger.info("Done preparing diagnosis prediction for the given cohorts")

        return labeled_cohorts

    def make_compatible(self, icustays):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patient_fname))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))

        # prepare icustays according to the appropriate format
        icustays = icustays[icustays["FIRST_CAREUNIT"] == icustays["LAST_CAREUNIT"]]
        icustays.loc[:, "INTIME"] = pd.to_datetime(
            icustays["INTIME"], infer_datetime_format=True
        )
        icustays.loc[:, "OUTTIME"] = pd.to_datetime(
            icustays["OUTTIME"], infer_datetime_format=True
        )
        icustays = icustays.drop(columns=["ROW_ID"])

        # merge icustays with patients to get DOB
        patients["DOB"] = pd.to_datetime(patients["DOB"], infer_datetime_format=True)
        patients = patients[
            patients["SUBJECT_ID"].isin(icustays["SUBJECT_ID"])
        ]
        patients = patients.drop(columns=["ROW_ID"])[["DOB", "SUBJECT_ID"]]
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

        icustays["IN_ICU_MORTALITY"] = (
            (icustays["INTIME"] < icustays["DISCHTIME"])
            & (icustays["DISCHTIME"] <= icustays["OUTTIME"])
            & (icustays["DISCHARGE_LOCATION"] == "Death")
        )
        icustays.rename(columns={"DISCHARGE_LOCATION": "HOS_DISCHARGE_LOCATION"}, inplace=True)

        icustays["DISCHTIME"] = (icustays["DISCHTIME"] - icustays["INTIME"]).dt.total_seconds() // 60
        icustays["OUTTIME"] = (icustays["OUTTIME"] - icustays["INTIME"]).dt.total_seconds() // 60
        return icustays


    
    def infer_data_extension(self) -> str:
        if (len(glob.glob(os.path.join(self.data_dir, "*.csv.gz"))) == 26):
            ext = ".csv.gz"
        elif (len(glob.glob(os.path.join(self.data_dir, "*.csv")))==26):
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext