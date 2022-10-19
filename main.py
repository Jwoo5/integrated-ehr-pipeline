import logging

import argparse

logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument(
        '--task', type=str, default=None,
        help='specific task from the whole pipeline.'
            'if not set, run the whole steps.'
    )
    parser.add_argument(
        "--dest", default="outputs", type=str, metavar="DIR", help="output directory"
    )
    # data
    parser.add_argument(
        '--ehr', type=str, required=True,
        help='name of the ehr system to be processed.'
    )
    parser.add_argument(
        '--data', metavar='DIR', default=None,
        help='directory containing .csv files of --ehr.'
            'if not given, try to download from the internet.'
    )
    parser.add_argument(
        '--ccs', type=str, default=None,
        help='path to `ccs_multi_dx_tool_2015.csv`'
            'if not given, try to download from the internet.'
    )


    # misc
    parser.add_argument(
        '--max_event_size', type=int, default=None,
        help='max event size to crop to'
    )
    parser.add_argument(
        '--min_event_size', type=int, default=None,
        help='min event size to skip small samples'
    )
    parser.add_argument(
        '--min_age', type=int, default=18,
        help='min age to skip too young patients'
    )
    parser.add_argument(
        '--max_age', type=int, default=None,
        help='max age to skip too old patients'
    )
    parser.add_argument(
        '--obs_size', type=int, default=12,
        help='observation window size by the hour'
    )
    parser.add_argument(
        '--gap_size', type=int, default=6,
        help='time gap window size by the hour'
    )
    parser.add_argument(
        '--pred_size', type=int, default=48,
        help='prediction window size by the hour'
    )

def main(args):
    task = args.task

    if args.data is not None:


    if task is None:
        logger.info('...')

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)