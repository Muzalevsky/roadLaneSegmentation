from abc import ABC, abstractmethod
from enum import Enum


class DatasetName(Enum):
    Apolloscape = "apolloscape"


class BaseParser(ABC):
    src_key = "src"
    target_key = "tgt"
    folder_key = "folder"

    output_img_key = "images"
    output_msk_key = "labels"

    @abstractmethod
    def parse(self, dataset_path: str):
        """Parse raw dataset and create df with mapping [source file; mask file]

        Parameters
        ----------
        dataset_path : str
            path to the dataset root folder
        """
