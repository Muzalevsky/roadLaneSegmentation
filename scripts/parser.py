#!/usr/bin/env python3

import argparse
import os
import sys
from datetime import datetime

from lane_detection_hackathon import apolloscape


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="name of the dataset root folder", type=str)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    PROJECT_ROOT = os.path.dirname(os.getcwd())
    DATA_DIR = PROJECT_ROOT + os.path.sep + "data"
    RAW_DATASET_DIR = DATA_DIR + os.path.sep + args.dataset
    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"RAW_DATASET_DIR={RAW_DATASET_DIR}")

    if args.dataset == "apolloscape":
        ds_parser = apolloscape.ApolloScape()
        df = ds_parser.parse(RAW_DATASET_DIR)
        table_dir = DATA_DIR + os.sep + "datasets" + os.sep + args.dataset
        if not os.path.exists(table_dir):
            os.makedirs(table_dir)

        excel_path = table_dir + os.sep + f'raw_data_{datetime.today().strftime("%Y_%m_%d")}.xlsx'
        df.to_excel(excel_path)
        print(f"Parsed dataset <{args.dataset}>. Resulting file: {excel_path}")


if __name__ == "__main__":
    sys.exit(main())
