from typing import Optional

import glob
import logging
import multiprocessing as mpr
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .baseparser import BaseParser
from .patch_extractor import ImageBlockReader
from .utils import fs

SampleInfo = namedtuple("SampleInfo", field_names=["img_fpath", "mask_fpath", "folder_idx"])


class ApolloScape(BaseParser):
    dataset_folder_name = "apolloscape"

    def __init__(
        self,
        dpath: str,
        res_dir: str,
        img_dir: str,
        mask_dir: str,
        excel_dir: str,
        n_jobs: Optional[int] = -1,
    ):
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._root_dpath = dpath
        self._res_dir = res_dir
        self._img_dir = img_dir
        self._mask_dir = mask_dir
        self._excel_dir = excel_dir

        self._n_jobs = n_jobs
        if self._n_jobs < 0:
            self._n_jobs = mpr.cpu_count() - 1
        self._logger.info(f"{self._n_jobs} jobs are configured.")

    @staticmethod
    def is_in(dpath: str):
        return ApolloScape.dataset_folder_name in dpath

    @staticmethod
    def process_image(
        res_dir: str,
        cell_images_dir: str,
        cell_mask_dir: str,
        cell_size_px: int,
        image_fpath: str,
        mask_fpath: str,
        folder_idx: int,
        logger: logging.Logger,
    ):

        imagepath = image_fpath
        maskpath = mask_fpath

        img_name, img_extension = os.path.splitext(os.path.basename(imagepath))
        mask_name, mask_extension = os.path.splitext(os.path.basename(maskpath))

        img = fs.read_image(imagepath)
        mask = fs.read_image(maskpath)

        if img.shape[:2] != mask.shape[:2]:
            logger.warning(f"Different image sizes <{img.shape[:2]} != {mask.shape[:2]}>.")
            return

        block_reader = ImageBlockReader([0, 0, 0])
        img_tiles = block_reader.read_blocks(img, cell_size_px)
        mask_tiles = block_reader.read_blocks(mask, cell_size_px)

        img_cell_dirpath = os.path.join(cell_images_dir, img_name)
        mask_cell_dirpath = os.path.join(cell_mask_dir, mask_name)
        os.makedirs(img_cell_dirpath, exist_ok=True)
        os.makedirs(mask_cell_dirpath, exist_ok=True)

        cells = []

        cells_geometry = block_reader.get_cells_geometry(img.shape[1], img.shape[0], cell_size_px)
        fs.save_json(os.path.join(img_cell_dirpath, "geometry.json"), cells_geometry)
        fs.save_json(os.path.join(mask_cell_dirpath, "geometry.json"), cells_geometry)

        for index, (img_tile, mask_tile) in enumerate(zip(img_tiles, mask_tiles)):
            img_cell_fname = f"{index}" + img_extension
            mask_cell_fname = f"{index}" + mask_extension

            img_cell_fpath = os.path.join(img_cell_dirpath, img_cell_fname)
            mask_cell_fpath = os.path.join(mask_cell_dirpath, mask_cell_fname)

            fs.write_image(img_cell_fpath, img_tile)
            fs.write_image(mask_cell_fpath, mask_tile)

            img_cell_rel_path = img_cell_fpath.replace(res_dir, "").lstrip(os.sep)
            mask_cell_rel_path = mask_cell_fpath.replace(res_dir, "").lstrip(os.sep)

            cells.append([img_cell_rel_path, mask_cell_rel_path, folder_idx])

        return cells

    def get_dataset_info(self) -> pd.DataFrame:
        image_infos, unique_folders = [], []
        for image_fpath in glob.iglob(os.path.join(self._root_dpath, "**/*.jpg"), recursive=True):
            mask_dirpath = os.path.dirname(image_fpath).replace("ColorImage", "Label")
            image_basename = os.path.splitext(os.path.basename(image_fpath))[0]
            mask_fpath = os.path.join(mask_dirpath, image_basename + "_bin.png")

            if mask_dirpath not in unique_folders:
                unique_folders.append(mask_dirpath)

            if os.path.exists(mask_fpath):
                img_info = SampleInfo(
                    img_fpath=os.path.abspath(image_fpath),
                    mask_fpath=os.path.abspath(mask_fpath),
                    folder_idx=unique_folders.index(mask_dirpath),
                )
                image_infos.append(img_info)

        df = pd.DataFrame(
            image_infos, columns=[BaseParser.src_key, BaseParser.target_key, BaseParser.folder_key]
        )
        return df

    def parse(self, cell_size_px: int):
        image_infos = self.get_dataset_info()

        RANDOM_SEED = 42
        unique_folders = np.unique(image_infos[self.folder_key])

        self._logger.info(f"Unique folders: {unique_folders.shape[0]}")

        train_folders, test_and_valid_folders = train_test_split(
            unique_folders, test_size=0.3, random_state=RANDOM_SEED
        )
        test_folders, valid_folders = train_test_split(
            test_and_valid_folders, test_size=0.5, random_state=RANDOM_SEED
        )

        df_train_and_valid = image_infos[
            image_infos[self.folder_key].isin(
                np.concatenate((train_folders, valid_folders), axis=0)
            )
        ]

        output_test = []
        for row in image_infos[image_infos[self.folder_key].isin(test_folders)].values:
            src_img_parts = row[0].split(ApolloScape.dataset_folder_name)
            src_img_rel_path = ApolloScape.dataset_folder_name + src_img_parts[1]
            src_mask_parts = row[1].split(ApolloScape.dataset_folder_name)
            src_mask_rel_path = ApolloScape.dataset_folder_name + src_mask_parts[1]
            output_test.append({self.src_key: src_img_rel_path, self.target_key: src_mask_rel_path})

        df_test = pd.DataFrame(output_test, columns=[self.src_key, self.target_key])
        df_test.to_excel(os.path.join(self._excel_dir, "test.xlsx"))

        fut_results = []
        cell_infos = []
        with ProcessPoolExecutor(max_workers=self._n_jobs) as ex:
            for index, row in df_train_and_valid.iterrows():
                fut_result = ex.submit(
                    self.process_image,
                    res_dir=self._res_dir,
                    cell_images_dir=self._img_dir,
                    cell_mask_dir=self._mask_dir,
                    cell_size_px=cell_size_px,
                    image_fpath=row[self.src_key],
                    mask_fpath=row[self.target_key],
                    folder_idx=row[self.folder_key],
                    logger=self._logger,
                )
                fut_results.append(fut_result)

            stream = tqdm(
                as_completed(fut_results), total=len(fut_results), desc="Image Info Processing"
            )
            for fut_result in stream:
                cell_info = fut_result.result()

                if cell_info is None:
                    continue

                cell_infos += cell_info

        df_parsed = pd.DataFrame(
            cell_infos, columns=[self.src_key, self.target_key, self.folder_key]
        )

        train_folders, valid_folders = train_test_split(
            np.unique(df_parsed[self.folder_key]), test_size=0.15, random_state=RANDOM_SEED
        )

        df_train = df_parsed[df_parsed[self.folder_key].isin(train_folders)]
        df_valid = df_parsed[df_parsed[self.folder_key].isin(valid_folders)]

        self._logger.info("Instances:")
        self._logger.info(f"\tTrain: \t{df_train.shape[0]}")
        self._logger.info(f"\tTest: \t{df_test.shape[0]}")
        self._logger.info(f"\tValid: \t{df_valid.shape[0]}")

        df_train.to_excel(os.path.join(self._excel_dir, "train.xlsx"))
        df_valid.to_excel(os.path.join(self._excel_dir, "valid.xlsx"))
