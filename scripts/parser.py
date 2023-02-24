#!/usr/bin/env python3


import logging
import os
import sys
from datetime import datetime

import click

from lane_detection_hackathon import apolloscape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parser.py")


@click.command()
@click.option("--dataset", "-d", help="name of the dataset root folder", type=str, required=True)
def main(dataset):
    PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DATASET_DIR = os.path.join(DATA_DIR, dataset)

    logger.info(f"PROJECT_ROOT={PROJECT_ROOT}")
    logger.info(f"RAW_DATASET_DIR={RAW_DATASET_DIR}")

    if dataset == "apolloscape":
        ds_parser = apolloscape.ApolloScape()
        df = ds_parser.parse(RAW_DATASET_DIR)
        table_dir = os.path.join(
            DATA_DIR, "datasets", dataset, datetime.today().strftime("%Y_%m_%d")
        )
        if not os.path.exists(table_dir):
            os.makedirs(table_dir)

        excel_path = os.path.join(table_dir, "raw_data.xlsx")
        df.to_excel(excel_path)
        logger.info(f"Parsed dataset <{dataset}>. Resulting file: {excel_path}")


if __name__ == "__main__":
    sys.exit(main())
