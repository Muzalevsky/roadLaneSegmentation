import glob
import os
import os.path as osp

import pandas as pd
from tqdm import tqdm

from .baseparser import BaseParser


class ApolloScape(BaseParser):
    def parse(self, dataset_path: str):
        root = dataset_path
        ds_name = os.path.basename(root)
        ds_root = os.path.dirname(root)
        src_files = []
        tgt_files = []
        folder_indices = []

        unique_folders = []

        for filename in tqdm(glob.iglob(root + os.sep + "**/*.jpg", recursive=True)):
            color_image_rel_path = filename.replace(root, ds_name)
            clear_fname = os.path.basename(color_image_rel_path)
            color_image_basename = os.path.splitext(clear_fname)[0]

            color_image_folderpath = os.path.dirname(color_image_rel_path)
            label_folderpath = color_image_folderpath.replace("ColorImage", "Label")
            label_candidate = os.path.join(label_folderpath, color_image_basename + "_bin.png")

            if label_folderpath not in unique_folders:
                unique_folders.append(label_folderpath)

            if osp.exists(os.path.join(ds_root, label_candidate)):
                src_files.append(color_image_rel_path)
                tgt_files.append(label_candidate)
                folder_indices.append(unique_folders.index(label_folderpath))

        return pd.DataFrame(
            {self.src_key: src_files, self.target_key: tgt_files, self.folder_key: folder_indices}
        )
