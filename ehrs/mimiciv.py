import glob
import logging
import os

import pandas as pd

from ehrs import EHR, Table, register_ehr

logger = logging.getLogger(__name__)


@register_ehr("mimiciv")
class MIMICIV(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "mimiciv"

        if self.data_dir is None:
            self.data_dir = os.path.join(self.cache_dir, self.ehr_name)

            if not os.path.exists(self.data_dir):
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )
                self.download_ehr_from_url(
                    url="https://physionet.org/files/mimiciv/2.0/", dest=self.data_dir
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

        self._icustay_fname = "icu/icustays" + self.ext
        self._patient_fname = "hosp/patients" + self.ext
        self._admission_fname = "hosp/admissions" + self.ext
        self._diagnosis_fname = "hosp/diagnoses_icd" + self.ext
        self._d_diagnosis_fname = "hosp/d_icd_diagnoses" + self.ext
        self._determine_first_icu = "INTIME"

        labevents = Table(
            fname="hosp/labevents" + self.ext,
            timestamp="charttime",
            endtime=None,
            itemid="itemid",
            value=["value", "valuenum"],
            uom="valueuom",
            text=None,
            code="itemid",
            desc="hosp/d_labitems" + self.ext,
            desc_key="label",
            merge_value=True,
        )

        microbiologyevents = Table(
            fname="hosp/microbiologyevents" + self.ext,
            timestamp="charttime",
            endtime=None,
            itemid=None,
            value=None,
            uom=None,
            text=["test_name", "org_name", "ab_name", "interpretation", "comments"],
            code=None,
            desc=None,
            desc_key=None,
        )
        prescriptions = Table(
            fname="hosp/prescriptions" + self.ext,
            timestamp="starttime",
            endtime="stoptime",
            itemid=["drug", "prod_strength"],
            value=["dose_val_rx"],
            uom="dose_unit_rx",
            text=["route"],
            code=None,
            desc=None,
            desc_key=None,
        )
        inputevents = Table(
            fname="icu/inputevents" + self.ext,
            timestamp="starttime",
            endtime="endtime",
            itemid="itemid",
            value=["amount"],
            uom="amountuom",
            text=None,
            code="itemid",
            desc="icu/d_items" + self.ext,
            desc_key="label",
        )
        outputevents = Table(
            fname="icu/outputevents" + self.ext,
            timestamp="charttime",
            endtime=None,
            itemid="itemid",
            value=["value"],
            uom="valueuom",
            text=None,
            code="itemid",
            desc="icu/d_items" + self.ext,
            desc_key="label",
        )
        procedureevents = Table(
            fname="icu/procedureevents" + self.ext,
            timestamp="starttime",
            endtime="endtime",
            itemid="itemid",
            value=["value"],
            uom="valueuom",
            text=["location", "locationcategory"],
            code="itemid",
            desc="icu/d_items" + self.ext,
            desc_key="label",
        )

        self.tables = [
            labevents,
            microbiologyevents,
            prescriptions,
            inputevents,
            outputevents,
            procedureevents,
        ]

        if self.add_chart:
            self.tables += [
                Table(
                    fname="icu/chartevents" + self.ext,
                    timestamp="charttime",
                    endtime=None,
                    itemid="itemid",
                    value=["value", "valuenum"],
                    uom="valueuom",
                    text=None,
                    code="itemid",
                    desc="icu/d_items" + self.ext,
                    desc_key="label",
                )
            ]

        if self.lab_only:
            self.tables = [self.tables[0]]

        self._icustay_key = "stay_id"
        self._hadm_key = "hadm_id"
        self._patient_key = "subject_id"

    def build_cohorts(self, cached=False):
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustay_fname))

        icustays = self.make_compatible(icustays)
        self.icustays = icustays

        cohorts = super().build_cohorts(icustays, cached=cached)

        return cohorts

    def make_compatible(self, icustays):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patient_fname))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))

        # prepare icustays according to the appropriate format
        icustays = icustays.rename(
            columns={
                "los": "LOS",
                "intime": "INTIME",
            }
        )
        admissions = admissions.rename(
            columns={
                "deathtime": "DEATHTIME",
                "admittime": "ADMITTIME",
            }
        )

        icustays = icustays[icustays["first_careunit"] == icustays["last_careunit"]]
        icustays["INTIME"] = pd.to_datetime(
            icustays["INTIME"], infer_datetime_format=True, utc=True
        )

        icustays = icustays.merge(patients, on="subject_id", how="left")
        icustays["AGE"] = (
            icustays["INTIME"].dt.year
            - icustays["anchor_year"]
            + icustays["anchor_age"]
        )

        icustays = icustays.merge(
            admissions[[self.hadm_key, "DEATHTIME", "ADMITTIME", "race"]],
            how="left",
            on=self.hadm_key,
        )
        diagnosis = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))
        d_diagnosis = pd.read_csv(os.path.join(self.data_dir, self.d_diagnosis_fname))

        diagnosis = diagnosis.merge(
            d_diagnosis, how="left", on=["icd_code", "icd_version"]
        )

        diagnosis = self.icd10toicd9(diagnosis)

        diagnosis = (
            diagnosis[[self.hadm_key, "icd_code_converted", "long_title"]]
            .groupby(self.hadm_key)
            .agg(list)
            .rename(columns={"icd_code_converted": "icd_9", "long_title": "icd_text"})
        )

        icustays = icustays.merge(diagnosis, how="left", on=self.hadm_key)
        icustays["icd_9"] = icustays["icd_9"].fillna("").apply(list)
        icustays["icd_text"] = icustays["icd_text"].fillna("").apply(list)

        icustays["ADMITTIME"] = pd.to_datetime(
            icustays["ADMITTIME"], infer_datetime_format=True, utc=True
        )
        icustays["DEATHTIME"] = pd.to_datetime(
            icustays["DEATHTIME"], infer_datetime_format=True, utc=True
        )

        sofa_df = pd.read_csv(
            os.path.join(self.cfg.derived_path, "sofa.csv"),
            usecols=["stay_id", "starttime", "sofa_24hours"],
        )
        sofa_df = pd.merge(
            icustays[["stay_id", "INTIME"]], sofa_df, on="stay_id", how="left"
        )
        sofa_df["starttime"] = pd.to_datetime(
            sofa_df["starttime"], infer_datetime_format=True, utc=True
        )
        sofa_df["offset"] = sofa_df.apply(
            lambda x: (x["starttime"] - x["INTIME"]).total_seconds() // 60,
            axis=1,
        )
        sofa_df = (
            sofa_df.sort_values("offset")
            .groupby("stay_id")
            .agg({"offset": list, "sofa_24hours": list})
        )
        sofa_df["sofa"] = sofa_df.apply(
            lambda x: (x["offset"], x["sofa_24hours"]), axis=1
        )
        sofa_df.reset_index(inplace=True)
        icustays = pd.merge(
            icustays, sofa_df[["stay_id", "sofa"]], on="stay_id", how="left"
        )

        icustays["INTIME_DATE"] = icustays["INTIME"].dt.date

        icustays["INTIME"] = (
            icustays["INTIME"] - icustays["ADMITTIME"]
        ).dt.total_seconds() // 60
        icustays["DEATHTIME"] = (
            icustays["DEATHTIME"] - icustays["ADMITTIME"]
        ).dt.total_seconds() // 60

        icustays = icustays.drop(
            columns=[
                "first_careunit",
                "last_careunit",
                "anchor_age",
                "anchor_year",
                "anchor_year_group",
            ]
        )

        return icustays

    def icd10toicd9(self, dx):
        gem = pd.read_csv(self.gem_path)
        dx_icd_10 = dx[dx["icd_version"] == 10]["icd_code"]

        unique_elem_no_map = set(dx_icd_10) - set(gem["icd10cm"])

        map_cms = dict(zip(gem["icd10cm"], gem["icd9cm"]))
        map_manual = dict.fromkeys(unique_elem_no_map, "NaN")

        for code_10 in map_manual:
            for i in range(len(code_10), 0, -1):
                tgt_10 = code_10[:i]
                if tgt_10 in gem["icd10cm"]:
                    tgt_9 = (
                        gem[gem["icd10cm"].str.contains(tgt_10)]["icd9cm"]
                        .mode()
                        .iloc[0]
                    )
                    map_manual[code_10] = tgt_9
                    break

        def icd_convert(icd_version, icd_code):
            if icd_version == 9:
                return icd_code

            elif icd_code in map_cms:
                return map_cms[icd_code]

            elif icd_code in map_manual:
                return map_manual[icd_code]
            else:
                logger.warn("WRONG CODE: " + icd_code)

        dx["icd_code_converted"] = dx.apply(
            lambda x: icd_convert(x["icd_version"], x["icd_code"]), axis=1
        )
        return dx

    def infer_data_extension(self) -> str:
        if (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv.gz"))) == 22
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv.gz"))) == 9
        ):
            ext = ".csv.gz"
        elif (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv"))) == 22
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv"))) == 9
        ):
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext
