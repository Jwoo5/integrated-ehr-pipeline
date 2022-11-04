import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import argparse
import os
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence


"""
Sample Dataset for Integrated-EHR-Pipeline
NOTE: The preprocessed token indices are stored as np.int16 type for efficienty,
        so overflow can be caused when vocab size > 32767.
 - ehr.cohort.labeled.index: dataframe pickle
 - ehr.data: .np file with np.int16 type, (num_total_events, 3, max_word_len)
    - second dimension
        - 0: input_ids
        - 1: type_ids
        - 2: dpe_ids
"""


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ehr",
        type=str,
        required=True,
        choices=["mimiciii", "mimiciv", "eicu"],
        help="Name of the EHR dataset",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the preprocessed data"
    )
    parser.add_argument(
        "--pred_target",
        type=str,
        required=True,
        choices=[
            "mortality",
            "readmission",
            "los_3day",
            "los_7day",
            "final_acuity",
            "imminent_discharge",
            "diagnosis",
        ],
        help="Prediction target",
    )
    parser.add_argument(
        "--max_word_len",
        type=int,
        default=128,
        help="Maximum token length for each event",
    )
    parser.add_argument(
        "--max_event_len",
        type=int,
        default=256,
        help="Maximum number of events to consider",
    )
    return parser


class EHRDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()

        self.args = args
        df_path = os.path.join(args.data, f"{args.ehr}.cohorts.labeled.index")
        self.df = pd.read_pickle(df_path)
        self.df = self.df[self.df["split"] == split]
        self.data_path = os.path.join(args.data, f"{args.ehr}.data.npy")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # NOTE: Warning occurs when converting np.int16 read-only array into tensor, but ignoreable
        row = self.df.iloc[idx]

        assert row["end"] - row["start"] <= self.args.max_event_len

        data = np.memmap(
            self.data_path,
            dtype=np.int16,
            shape=(row["end"] - row["start"], 3, self.args.max_word_len),
            mode="r",
            offset=row["start"] * 3 * 2 * self.args.max_word_len,  # np.int16 = 2 bytes
        )
        return {
            "input_ids": torch.IntTensor(data[:, 0, :]),
            "type_ids": torch.IntTensor(data[:, 1, :]),
            "dpe_ids": torch.IntTensor(data[:, 2, :]),
            "label": row[self.args.pred_target],
        }

    def collate_fn(self, out):
        ret = dict()
        if len(out) == 1:
            for k, v in out[0].items():
                if k == "label":
                    ret[k] = (
                        one_hot(torch.LongTensor([v]), self.num_classes)
                        .float()
                        .reshape((1, self.num_classes))
                    )
                else:
                    ret[k] = v.unsqueeze(0)
        else:
            for k, v in out[0].items():
                if k == "label":
                    ret[k] = one_hot(
                        torch.LongTensor([i[k] for i in out]), self.num_classes
                    ).float()
                else:
                    ret[k] = pad_sequence([i[k] for i in out], batch_first=True)
        return ret
