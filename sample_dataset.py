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
    - index: should be resetted by `df.reset_index()`
    - [hi_start:hi_end] is the range of indices for the patient's history
    - split: one of ['train', 'val', 'test']
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
        help="Maximum token length for each event for Hi",
    )
    parser.add_argument(
        "--max_event_len",
        type=int,
        default=256,
        help="Maximum number of events to consider for Hi",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=8192,
        help="Maximum sequence length for Flat",
    )

    return parser


class BaseEHRDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()

        self.args = args
        df_path = os.path.join(args.data, f"{args.ehr}.cohorts.labeled.index")
        self.df = pd.read_pickle(df_path)
        self.df = self.df[self.df["split"] == split]
        self.data_path = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError()

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


class HierarchicalEHRDataset(BaseEHRDataset):
    def __init__(self, args, split):
        super().__init__(args, split)
        self.data_path = os.path.join(args.data, f"{args.ehr}.hi.npy")

    def __getitem__(self, idx):
        # NOTE: Warning occurs when converting np.int16 read-only array into tensor, but ignoreable
        row = self.df.iloc[idx]

        assert row["hi_end"] - row["hi_start"] <= self.args.max_event_len

        data = np.memmap(
            self.data_path,
            dtype=np.int16,
            shape=(row["hi_end"] - row["hi_start"], 3, self.args.max_word_len),
            mode="r",
            offset=row["hi_start"] * 3 * 2 * self.args.max_word_len,
        )
        return {
            "input_ids": torch.IntTensor(data[:, 0, :]),
            "type_ids": torch.IntTensor(data[:, 1, :]),
            "dpe_ids": torch.IntTensor(data[:, 2, :]),
            "label": row[self.args.pred_target],
        }


class FlattenEHRDataset(BaseEHRDataset):
    def __init__(self, args, split):
        super().__init__(args, split)
        self.data_path = os.path.join(args.data, f"{args.ehr}.flat.npy")

    def __getitem__(self, idx):
        # NOTE: Warning occurs when converting np.int16 read-only array into tensor, but ignoreable
        row = self.df.iloc[idx]

        data = np.memmap(
            self.data_path,
            dtype=np.int16,
            shape=(3, self.args.max_seq_len),
            mode="r",
            offset=row.index * 3 * 2 * self.args.max_seq_len,
        )
        return {
            "input_ids": torch.IntTensor(data[0, :]),
            "type_ids": torch.IntTensor(data[1, :]),
            "dpe_ids": torch.IntTensor(data[2, :]),
            "label": row[self.args.pred_target],
        }
