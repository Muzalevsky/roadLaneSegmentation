from typing import Optional

import glob
import logging
import multiprocessing as mpr
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
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
        imageinfo: SampleInfo,
        logger: logging.Logger,
    ):

        imagepath = imageinfo.img_fpath
        maskpath = imageinfo.mask_fpath
        folder = imageinfo.folder_idx

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

        cells = []
        for index, (img_tile, mask_tile) in enumerate(zip(img_tiles, mask_tiles)):
            img_cell_fname = img_name + f"_{index}" + img_extension
            mask_cell_fname = mask_name + f"_{index}" + mask_extension

            img_cell_fpath = os.path.join(cell_images_dir, img_cell_fname)
            mask_cell_fpath = os.path.join(cell_mask_dir, mask_cell_fname)

            fs.write_image(img_cell_fpath, img_tile)
            fs.write_image(mask_cell_fpath, mask_tile)

            img_cell_rel_path = img_cell_fpath.replace(res_dir, "").lstrip(os.sep)
            mask_cell_rel_path = mask_cell_fpath.replace(res_dir, "").lstrip(os.sep)

            cells.append([img_cell_rel_path, mask_cell_rel_path, folder])

        return cells

    def get_dataset_info(self) -> list[SampleInfo]:
        image_infos, unique_folders = [], []
        for image_fpath in glob.iglob(os.path.join(self._root_dpath, "**/*.jpg"), recursive=True):
            mask_dirpath = os.path.dirname(image_fpath).replace("ColorImage", "Label")
            image_basename = os.path.splitext(os.path.basename(image_fpath))[0]
            mask_fpath = os.path.join(mask_dirpath, image_basename + "_bin.png")

            if mask_dirpath not in unique_folders:
                unique_folders.append(mask_dirpath)

            if os.path.exists(mask_fpath):
                img_info = SampleInfo(
                    img_fpath=image_fpath,
                    mask_fpath=mask_fpath,
                    folder_idx=unique_folders.index(mask_dirpath),
                )
                image_infos.append(img_info)

        return image_infos

    def parse(self, cell_size_px: int):
        image_infos = self.get_dataset_info()

        fut_results = []
        cell_infos = []
        with ProcessPoolExecutor(max_workers=self._n_jobs) as ex:
            for img_info in image_infos:
                fut_result = ex.submit(
                    self.process_image,
                    res_dir=self._res_dir,
                    cell_images_dir=self._img_dir,
                    cell_mask_dir=self._mask_dir,
                    cell_size_px=cell_size_px,
                    imageinfo=img_info,
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

        df = pd.DataFrame(cell_infos, columns=[self.src_key, self.target_key, self.folder_key])
        df.to_excel(os.path.join(self._excel_dir, "raw_data.xlsx"))
