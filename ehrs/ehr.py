import sys
import os
import glob
import subprocess
import shutil
import logging

import datetime
import pandas as pd

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

    def build_cohorts(self, icustays, cached=False):
        if cached:
            loaded = self.load_from_cache(self.ehr_name + ".cohorts")
            if loaded:
                return self.cohorts

        logger.info(
            "Start to build cohorts for {}".format(self.ehr_name)
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
            loaded = self.load_from_cache(self.ehr_name + ".cohorts.labeled")
            if loaded:
                return self.cohorts

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

        return labeled_cohorts

    def save_to_cache(self, f, fname, use_pickle=False) -> None:
        if use_pickle:
            pass
        else:
            f.to_pickle(
                os.path.join(self.cache_dir, fname)
            )

    def load_from_cache(self, fname) -> bool:
        cached = os.path.join(self.cache_dir, fname)
        if os.path.exists(cached):
            self.cohorts = pd.read_pickle(cached)

            logger.info(
                "Loaded data from {}".format(len(self.cohorts), cached)
            )
            return True
        else:
            return False

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