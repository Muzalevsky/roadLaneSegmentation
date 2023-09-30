import logging
import os
import sys
from datetime import datetime

import click

from road_lane_segmentation import apolloscape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


@click.command()
@click.option("--input", "-i", help="dataset root folder path", type=str, required=True)
@click.option("--output", "-o", help="output folder path", type=str, required=True)
@click.option("--cell_size", "-c", help="cell size of the image in [px]", type=int, required=True)
def main(input, output, cell_size):
    if apolloscape.ApolloScape.is_in(input):
        parser = apolloscape.ApolloScape
    else:
        raise ValueError(f"Unsupported dataset <{input}>.")

    images_dirname = os.path.join(output, "images")
    os.makedirs(images_dirname, exist_ok=True)

    masks_dirname = os.path.join(output, "masks")
    os.makedirs(masks_dirname, exist_ok=True)

    excel_dirname = os.path.join(output, datetime.today().strftime("%Y_%m_%d"))
    os.makedirs(excel_dirname, exist_ok=True)

    logger.info(f"RAW_DATASET_DIR={input}")
    logger.info(f"OUTPUT_DIR={output}")

    ds_parser = parser(input, output, images_dirname, masks_dirname, excel_dirname)
    ds_parser.parse(cell_size)
    logger.info(f"Parsed dataset <{input}>.")


if __name__ == "__main__":
    sys.exit(main())
