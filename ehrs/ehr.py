import sys
import os
import subprocess
import logging
import os
import shutil

logger = logging.getLogger(__name__)


class EHR(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.data_dir = cfg.data

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

        os.makedirs(self.dest, exist_ok=True)

    def readmission_label(self, df):
        df.sort_values([self.second_key, self.icustay_key], inplace=True)

        if self.first_icu:
            df = df.groupby(self.second_key).first()

        df["readmission"] = 1
        # the last icustay for each HADM_ID means that they have no icu readmission
        df.loc[
            df.groupby(self.second_key)[self.icustay_key].idxmax(),
            "readmission",
        ] = 0

        return df

    def get_physionet_dataset(self, physionet_file_path, cache_dir):

        data_dir = self.cfg.data

        if data_dir is None or not os.path.exists(data_dir):
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            data_dir = os.path.join(cache_dir, "mimiciii")

            # Skip sanity check
            if os.path.exists(data_dir):
                logger.info("Loaded cached ehr data from {}.".format(data_dir))
            else:
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "It requires ~7GB drive spaces. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )

                username = input("Email or Username: ")
                subprocess.run(
                    [
                        "wget",
                        "-r",
                        "-N",
                        "-c",
                        "np",
                        "--user",
                        username,
                        "--ask-password",
                        f"https://physionet.org/files/{physionet_file_path}",
                        "-P",
                        cache_dir,
                    ]
                )

                if (
                    len(
                        os.listdir(
                            os.path.join(
                                cache_dir, f"physionet.org/files/{physionet_file_path}"
                            )
                        )
                    )
                    != 30
                ):
                    raise AssertionError(
                        "Access refused. Please log in with a credentialed user."
                    )

                os.rename(
                    os.path.join(
                        cache_dir, f"physionet.org/files/{physionet_file_path}"
                    ),
                    data_dir,
                )
                shutil.rmtree(os.path.join(cache_dir, "physionet.org"))
        else:
            logger.info("Loaded cached ehr data from {}.".format(data_dir))
        return data_dir

    def get_ccs(self, cache_dir):
        ccs_path = self.cfg.ccs
        if ccs_path is None or not os.path.exists(ccs_path):
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            ccs_path = os.path.join(cache_dir, "ccs_multi_dx_tool_2015.csv")

            if os.path.exists(ccs_path):
                logger.info("Loaded cached ccs file from {}".format(ccs_path))
            else:
                logger.info(
                    "`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet."
                )

                subprocess.run(
                    [
                        "wget",
                        "https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip",
                        "-P",
                        cache_dir,
                    ]
                )

                import zipfile

                with zipfile.ZipFile(
                    os.path.join(cache_dir, "Multi_Level_CCS_2015.zip"), "r"
                ) as zip_ref:
                    zip_ref.extractall(os.path.join(cache_dir, "tmp"))
                os.rename(
                    os.path.join(cache_dir, "tmp", "ccs_multi_dx_tool_2015.csv"),
                    ccs_path,
                )
                os.remove(os.path.join(cache_dir, "Multi_Level_CCS_2015.zip"))
                shutil.rmtree(os.path.join(cache_dir, "tmp"))

        return ccs_path

    def get_gem(self, cache_dir):
        gem_path = self.cfg.gem
        if gem_path is None or not os.path.exists(gem_path):
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            gem_path = os.path.join(cache_dir, "icd10cmtoicd9gem.csv")

            if os.path.exists(gem_path):
                logger.info("Loaded cached gem file from {}".format(gem_path))
            else:
                logger.info(
                    "`icd10cmtoicd9gem.csv` is not found so try to download from the internet."
                )

                subprocess.run(
                    [
                        "wget",
                        "https://data.nber.org/gem/icd10cmtoicd9gem.csv",
                        "-P",
                        cache_dir,
                    ]
                )

        return gem_path
