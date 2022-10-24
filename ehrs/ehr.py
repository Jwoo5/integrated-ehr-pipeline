import sys
import os

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
