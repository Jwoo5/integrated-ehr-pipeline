import argparse
import logging
import os
import sys

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
    parser.add_argument(
        "--seed", default="42", type=str, metavar="N", help="random seed"
    )

    # data
    parser.add_argument(
        "--ehr",
        type=str,
        required=True,
        choices=["mimiciii", "mimiciv", "eicu"],
        help="name of the ehr system to be processed.",
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
        "if not given, try to infer from --data",
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
        "-c",
        "--cache",
        action="store_true",
        help="whether to load data from cache if exists",
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
        "--obs_size", type=int, default=48, help="Restrict cohorts (ex. los>obs_size)"
    )
    parser.add_argument(
        "--pred_size",
        type=int,
        default=48,
        help="Prediction points from icu adm (ex. pred at pred_size)",
    )

    # tasks
    parser.add_argument(
        "--mortality",
        nargs="+",
        default=[1, 2, 3, 7, 14],
        help="whether to include n-day mortality tasks or not",
    )
    parser.add_argument(
        "--los",
        nargs="+",
        default=[7, 14],
        help="whether to include n-day los tasks or not",
    )
    parser.add_argument(
        "--readmission",
        action="store_true",
        help="whether to include readmission task or not",
    )
    parser.add_argument(
        "--diagnosis",
        action="store_true",
        help="whether to include diagnosis task or not",
    )
    parser.add_argument(
        "--creatinine",
        nargs="+",
        default=[1, 2, 3],
        help="whether to include n-day creatinine tasks or not",
    )
    parser.add_argument(
        "--platelets",
        nargs="+",
        default=[1, 2, 3],
        help="whether to include n-day platelets tasks or not",
    )
    parser.add_argument(
        "--wbc",
        nargs="+",
        default=[1, 2, 3],
        help="whether to include n-day wbc tasks or not",
    )
    parser.add_argument(
        "--hb",
        nargs="+",
        default=[1, 2, 3],
        help="whether to include n-day hb tasks or not",
    )
    parser.add_argument(
        "--bicarbonate",
        nargs="+",
        default=[1, 2, 3],
        help="whether to include n-day bicarbonate tasks or not",
    )
    parser.add_argument(
        "--sodium",
        nargs="+",
        default=[1, 2, 3],
        help="whether to include n-day sodium tasks or not",
    )

    parser.add_argument(
        "--max_event_token_len",
        type=int,
        default=128,
        help="max token length for each event (Hierarchical)",
    )

    parser.add_argument(
        "--max_patient_token_len",
        type=int,
        default=8192,
        help="max token length for each patient (Flatten)",
    )

    parser.add_argument(
        "--use_ed",
        action="store_true",
        help="Use ED data, Only supported for MIMIC-IV",
    )

    parser.add_argument(
        "--lab_only",
        action="store_true",
        help="Use only lab events as input",
    )

    parser.add_argument(
        "--num_threads", type=int, default=8, help="number of threads to use"
    )

    parser.add_argument(
        "--debug", action="store_true", help="whether to run in debug mode or not"
    )

    return parser


def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    ehr = EHR_REGISTRY[args.ehr](args)
    spark = (
        SparkSession.builder.master(f"local[{args.num_threads}]")
        .config("spark.driver.memory", "200g")
        .config("spark.driver.maxResultSize", "20g")
        .config("spark.network.timeout", "100s")
        .appName("Main_Preprocess")
        .getOrCreate()
    )
    ehr.run_pipeline(spark)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
