import logging
import os
import sys

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("splitter.py")


@click.command()
@click.option("--raw", help="path to the raw dataset description table", type=str, required=True)
def main(raw):

    raw_dir = os.path.abspath(os.path.dirname(raw))
    print(raw_dir)

    raw_df = pd.read_excel(raw, index_col=0)

    RANDOM_SEED = 42
    unique_folders = np.unique(raw_df["folders"])

    logger.info(f"Unique folders: {unique_folders.shape[0]}")

    train_folders, test_and_valid_folders = train_test_split(
        unique_folders, test_size=0.3, random_state=RANDOM_SEED
    )
    test_folders, valid_folders = train_test_split(
        test_and_valid_folders, test_size=0.5, random_state=RANDOM_SEED
    )

    logger.info("Folders:")
    logger.info(f"\tTrain: \t{train_folders}")
    logger.info(f"\tTest: \t{test_folders}")
    logger.info(f"\tValid: \t{valid_folders}")

    df_train = raw_df[raw_df["folders"].isin(train_folders)]
    df_test = raw_df[raw_df["folders"].isin(test_folders)]
    df_valid = raw_df[raw_df["folders"].isin(valid_folders)]

    logger.info("Instances:")
    logger.info(
        f"\tTrain: \t{df_train.shape[0]} \t({df_train.shape[0] / raw_df.shape[0] * 100:.2f}%)"
    )
    logger.info(f"\tTest: \t{df_test.shape[0]} \t({df_test.shape[0] / raw_df.shape[0] * 100:.2f}%)")
    logger.info(
        f"\tValid: \t{df_valid.shape[0]} \t({df_valid.shape[0] / raw_df.shape[0] * 100:.2f}%)"
    )

    df_train.to_excel(os.path.join(raw_dir, "train.xlsx"))
    df_test.to_excel(os.path.join(raw_dir, "test.xlsx"))
    df_valid.to_excel(os.path.join(raw_dir, "valid.xlsx"))


if __name__ == "__main__":
    sys.exit(main())
