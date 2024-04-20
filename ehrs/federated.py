import os
import pandas as pd
import sys
import logging
import argparse

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest", default="outputs", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--eicu_region", nargs="*", action="store", choices=["West", "South"], type=str, default=["West", "South"]
    )
    parser.add_argument(
        "--valid-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of data to use as validation and test set (between 0 and 0.5)",
    )
    parser.add_argument("--seed", default="42,43,44,45,46", type=str, metavar="N", help="random seed")
    return parser


def split_save_cohort(args, save_prefix, cohort, patient_key):
    for seed in args.seed:
        shuffled = cohort.groupby(patient_key)[patient_key].count().sample(frac=1, random_state=seed)
        cum_len = shuffled.cumsum()

        cohort.loc[cohort[patient_key].isin(
            shuffled[cum_len < int(len(shuffled)*args.valid_percent)].index), f'split_{seed}'] = 'test'
        cohort.loc[cohort[patient_key].isin(
            shuffled[(cum_len >= int(len(shuffled)*args.valid_percent)) 
            & (cum_len < int(len(shuffled)*2*args.valid_percent))].index), f'split_{seed}'] = 'valid'
        cohort.loc[cohort[patient_key].isin(
            shuffled[cum_len >= int(len(shuffled)*2*args.valid_percent)].index), f'split_{seed}'] = 'train'
    
    cohort.to_csv(os.path.join(args.dest, f"{save_prefix}_cohort.csv"), index=False)


def main(args):
    args.seed = [int(s) for s in args.seed.replace(' ','').split(",")]
    eicu_icustay_key = "patientunitstayid"
    eicu_patient_key = "uniquepid"
    eicu_hospital_key = "hospitalid"
    eicu_region_key = "region"
    mimiciii_icustay_key = "ICUSTAY_ID"
    mimiciii_dbsource = "DBSOURCE"
    mimiciii_patient_key = "SUBJECT_ID"

    cache_dir = os.path.expanduser("~/.cache/ehr")
    assert os.path.exists(cache_dir), "Cache path should exist and contain raw data files"

    # Create eICU Cohorts - Retrieve Top-N Hospitals from Original Cohort, and Divide into Separate Cohorts
    eicu = pd.read_csv(os.path.join(args.dest, "eicu_cohort.csv"))
    patient = pd.read_csv(os.path.join(cache_dir, 'eicu', 'patient.csv.gz'))
    hospital = pd.read_csv(os.path.join(cache_dir, 'eicu', 'hospital.csv.gz'))[['hospitalid', 'region']]

    split_cols = [col for col in eicu if col.startswith('split_')]
    eicu = eicu.drop(split_cols, axis=1)

    eicu = eicu.merge(patient[[eicu_icustay_key, eicu_hospital_key]], how='left', on=eicu_icustay_key)
    eicu = eicu.merge(hospital[[eicu_hospital_key, eicu_region_key]], how='left', on=eicu_hospital_key)
    
    for region in args.eicu_region:
        split_save_cohort(args, f"eicu_{region}", eicu[eicu['region']==region], eicu_patient_key)

    # Create MIMIC-III cohorts - Divide Cohorts Based on DBSOURCE
    mimiciii = pd.read_csv(os.path.join(args.dest, "mimiciii_cohort.csv"))
    icustays = pd.read_csv(os.path.join(cache_dir, 'mimiciii', 'ICUSTAYS.csv.gz'))

    split_cols = [col for col in mimiciii if col.startswith('split_')]
    mimiciii = mimiciii.drop(split_cols, axis=1)

    mimiciii = mimiciii.merge(icustays[[mimiciii_icustay_key, mimiciii_dbsource]], how='left', on=mimiciii_icustay_key)
    mimiciii_dbs = {"cv": "carevue", "mv": "metavision"}

    for i in mimiciii_dbs.keys():
        split_save_cohort(args, f"mimiciii_{i}", mimiciii[mimiciii[mimiciii_dbsource] == mimiciii_dbs[i]], mimiciii_patient_key)
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
