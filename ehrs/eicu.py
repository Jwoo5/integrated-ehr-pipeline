import glob
import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import treelib
from pyspark.sql.window import Window

from ehrs import EHR, Table, register_ehr

logger = logging.getLogger(__name__)


@register_ehr("eicu")
class eICU(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "eicu"

        if self.data_dir is None:
            self.data_dir = os.path.join(self.cache_dir, self.ehr_name)

            if not os.path.exists(self.data_dir):
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )
                self.download_ehr_from_url(
                    url="https://physionet.org/files/eicu-crd/2.0/", dest=self.data_dir
                )

        logger.info("Data directory is set to {}".format(self.data_dir))

        if self.ccs_path is None:
            self.ccs_path = os.path.join(self.cache_dir, "ccs_multi_dx_tool_2015.csv")

            if not os.path.exists(self.ccs_path):
                logger.info(
                    "`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet."
                )
                self.download_ccs_from_url(self.cache_dir)

        if self.gem_path is None:
            self.gem_path = os.path.join(self.cache_dir, "icd10cmtoicd9gem.csv")

            if not os.path.exists(self.gem_path):
                logger.info(
                    "`icd10cmtoicd9gem.csv` is not found so try to download from the internet."
                )
                self.download_icdgem_from_url(self.cache_dir)

        if self.ext is None:
            self.ext = self.infer_data_extension()

        self._icustay_fname = "patient" + self.ext
        self._diagnosis_fname = "diagnosis" + self.ext

        lab = Table(
            fname="lab" + self.ext,
            timestamp="labresultoffset",
            endtime=None,
            itemid="labname",
            value=["labresulttext"],
            uom="labmeasurenameinterface",
        )
        medication = Table(
            fname="medication" + self.ext,
            timestamp="drugstartoffset",
            endtime="drugstopoffset",
            itemid="drugname",
            value=["dosage"],
            text=["frequency"],
        )
        infusionDrug = Table(
            fname="infusionDrug" + self.ext,
            timestamp="infusionoffset",
            endtime=None,
            itemid="drugname",
            value=["drugrate"],  # Infusionrate stands for total liquid amount
        )
        intakeOutput = Table(
            fname="intakeOutput" + self.ext,
            timestamp="intakeoutputoffset",
            endtime=None,
            itemid="celllabel",
            value=["cellvaluetext"],
        )
        microlab = Table(
            fname="microLab" + self.ext,
            timestamp="culturetakenoffset",
            itemid="culturesite",
            value=["organism", "antibiotic", "sensitivitylevel"],
        )
        treatment = Table(
            fname="treatment" + self.ext,
            timestamp="treatmentoffset",
            endtime=None,
            itemid="treatmentstring",
        )

        self.tables = [lab, medication, infusionDrug, intakeOutput, microlab, treatment]

        if self.add_chart:
            raise NotImplementedError("eICU does not support chart events.")

        if self.lab_only:
            self.tables = [lab]

        self._icustay_key = "patientunitstayid"
        self._hadm_key = "patienthealthsystemstayid"
        self._patient_key = "uniquepid"

        self._determine_first_icu = "unitvisitnumber"

    def build_cohorts(self, cached=False):
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustay_fname))

        icustays = self.make_compatible(icustays)
        self.icustays = icustays

        cohorts = super().build_cohorts(icustays, cached=cached)

        return cohorts

    def make_compatible(self, icustays):
        icustays.loc[:, "LOS"] = icustays["unitdischargeoffset"] / 60 / 24
        icustays.dropna(subset=["age"], inplace=True)
        icustays["AGE"] = icustays["age"].replace("> 89", 300).astype(int)

        # hacks for compatibility with other ehrs
        icustays["ADMITTIME"] = 0
        icustays["DEATHTIME"] = np.nan
        is_dead = icustays["hospitaldischargestatus"] == "Expired"
        icustays.loc[is_dead, "DEATHTIME"] = (
            icustays.loc[is_dead, "hospitaldischargeoffset"]
            - icustays.loc[is_dead, "hospitaladmitoffset"]
        )
        icustays["INTIME"] = -icustays["hospitaladmitoffset"]
        icustays["LOS"] = icustays["unitdischargeoffset"] / 60 / 24

        # TODO: build diagnosis here
        diagnosis = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))
        # If only ICD10 -> Convert to ICD9
        # Both ICD9/10 -> Remove ICD10
        diagnosis["icd9code"] = (
            diagnosis["icd9code"].str.replace(".", "").str.split(",")
        )
        diagnosis = diagnosis.explode("icd9code")

        gem = pd.read_csv(self.gem_path)
        map_cms = dict(zip(gem["icd10cm"], gem["icd9cm"]))

        def convert_icd_9(code, map_cms):
            # All V codes in eICU are ICD-9, E codes are ICD-10
            if pd.isnull(code):
                return None
            elif code.startswith("V") or code.isdigit():
                return code
            else:
                if code in map_cms:
                    return map_cms[code]
                else:
                    return None

        diagnosis["icd9code"] = diagnosis["icd9code"].map(
            lambda x: convert_icd_9(x, map_cms)
        )

        diagnosis = (
            diagnosis.groupby(self.icustay_key)[["diagnosisstring", "icd9code"]]
            .agg(lambda x: set(x) - set([None]))
            .reset_index()
        )

        diagnosis = diagnosis.rename(
            columns={"diagnosisstring": "icd_text", "icd9code": "icd_9"}
        )
        # ICD-9 에 None 들어간 경우 지워야 함!
        # left join -> NaN 발생!
        icustays = icustays.merge(diagnosis, on=self.icustay_key, how="left")

        icustays["icd_text"] = icustays["icd_text"].fillna("").map(list)
        icustays["icd_9"] = icustays["icd_9"].fillna("").map(list)

        return icustays

    def infer_data_extension(self) -> str:
        if len(glob.glob(os.path.join(self.data_dir, "*.csv.gz"))) == 31:
            ext = ".csv.gz"
        elif len(glob.glob(os.path.join(self.data_dir, "*.csv"))) == 31:
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext
