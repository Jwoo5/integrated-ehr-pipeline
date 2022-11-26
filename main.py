import os
import sys
import logging
import argparse
from pyspark.sql import SparkSession

from ehrs import EHR_REGISTRY

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="specific task from the whole pipeline."
        "if not set, run the whole steps.",
    )
    parser.add_argument(
        "--dest", default="outputs", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of data to use as validation and test set (between 0 and 0.5)",
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")

    # data
    parser.add_argument(
        "--ehr", type=str, required=True, choices=['mimiciii', 'mimiciv', 'eicu'],
        help="name of the ehr system to be processed."
    )
    parser.add_argument(
        "--data",
        metavar="DIR",
        default=None,
        help="directory containing data files of the given ehr (--ehr)."
                "if not given, try to download from the internet.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=None,
        help="extension for ehr data to look for. "
                "if not given, try to infer from --data"
    )
    parser.add_argument(
        "--ccs",
        type=str,
        default=None,
        help="path to `ccs_multi_dx_tool_2015.csv`"
        "if not given, try to download from the internet.",
    )
    parser.add_argument(
        "--gem",
        type=str,
        default=None,
        help="path to `icd10cmtoicd9gem.csv`"
        "if not given, try to download from the internet.",
    )

    parser.add_argument(
        "-c", "--cache",
        action="store_true",
        help="whether to load data from cache if exists"
    )


    # misc
    parser.add_argument(
        "--max_event_size", type=int, default=256, help="max event size to crop to"
    )
    parser.add_argument(
        "--min_event_size",
        type=int,
        default=5,
        help="min event size to skip small samples",
    )
    parser.add_argument(
        "--min_age", type=int, default=18, help="min age to skip too young patients"
    )
    parser.add_argument(
        "--max_age", type=int, default=None, help="max age to skip too old patients"
    )
    parser.add_argument(
        "--obs_size", type=int, default=12, help="observation window size by the hour"
    )
    parser.add_argument(
        "--gap_size", type=int, default=0, help="time gap window size by the hour"
    )
    parser.add_argument(
        "--pred_size", type=int, default=24, help="prediction window size by the hour"
    )
    parser.add_argument(
        "--long_term_pred_size", type=int, default=336, help="prediction window size by the hour (for long term mortality task)"
    )
    parser.add_argument(
        "--first_icu",
        action="store_true",
        help="whether to use only the first icu or not",
    )

    # tasks
    parser.add_argument(
        "--mortality", action='store_true',
        help="whether to include mortality task or not"
    )
    parser.add_argument(
        "--long_term_mortality", action='store_true',
        help="whether to include long term mortality task or not"
    )
    parser.add_argument(
        "--los_3day", action='store_true',
        help="whether to include 3-day los task or not"
    )
    parser.add_argument(
        "--los_7day", action='store_true',
        help="whether to include 7-day los task or not"
    )
    parser.add_argument(
        "--readmission", action='store_true',
        help="whether to include readmission task or not"
    )
    parser.add_argument(
        "--final_acuity", action='store_true',
        help="whether to include final acuity task or not"
    )
    parser.add_argument(
        "--imminent_discharge", action="store_true",
        help="whether to include imminent discharge task or not"
    )
    parser.add_argument(
        "--diagnosis", action='store_true',
        help="whether to include diagnosis task or not"
    )
    parser.add_argument(
        "--creatinine", action='store_true',
        help="whether to include creatinine task or not"
    )
    parser.add_argument(
        "--bilirubin", action='store_true',
        help="whether to include bilirubin task or not"
    )
    parser.add_argument(
        "--platelets", action='store_true',
        help="whether to include platelets task or not"
    )
    parser.add_argument(
        "--crp", action='store_true',
        help="whether to include c-reactive protein task or not"
    )


    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="chunk size to read large csv files",
    )
    parser.add_argument(
        '--bins', type=int, default=20,
        help='num buckets to bin time intervals by'
    )

    parser.add_argument(
        '--max_event_token_len', type=int, default=128,
        help='max token length for each event (Hierarchical)'
    )

    parser.add_argument(
        '--max_patient_token_len', type=int, default=8192,
        help='max token length for each patient (Flatten)'
    )

    parser.add_argument(
        '--data_sampling', action='store_true',
        help='whether to perform data sampling or not'
    )
    parser.add_argument(
        '--ds_seed', type=int, default=10,
        help='random seed for data sampling'
    )
    parser.add_argument(
        '--ds_overlap', type=int, default=5,
        help="overlap size for data sampling"
    )

    parser.add_argument(
        '--num_threads', type=int, default=8, help='number of threads to use'
    )
    return parser


def main(args):
    task = args.task

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    ehr = EHR_REGISTRY[args.ehr](args)
    spark = (
        SparkSession.builder.master(f"local[{args.num_threads}]")
        .config("spark.driver.memory", "100g")
        .config("spark.driver.maxResultSize", "10g")
        .config("spark.network.timeout", "100s")
        .appName("Main_Preprocess")
        .getOrCreate()
    )
    ehr.run_pipeline(spark)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
