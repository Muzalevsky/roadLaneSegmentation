import glob
import os
import os.path as osp

import pandas as pd


class ApolloScape:
    def parse(self, dataset_path: str):
        root = dataset_path
        ds_name = os.path.basename(root)
        ds_root = os.path.dirname(root)
        src_files = []
        tgt_files = []

        for filename in glob.iglob(root + os.sep + "**/*.jpg", recursive=True):
            color_image_rel_path = filename.replace(root, ds_name)
            clear_fname = os.path.basename(color_image_rel_path)
            color_image_basename = os.path.splitext(clear_fname)[0]

            color_image_folderpath = os.path.dirname(color_image_rel_path)
            label_folderpath = color_image_folderpath.replace("ColorImage", "Label")
            label_candidate = os.path.join(label_folderpath, color_image_basename + "_bin.png")

            if osp.exists(os.path.join(ds_root, label_candidate)):
                src_files.append(color_image_rel_path)
                tgt_files.append(label_candidate)

        return pd.DataFrame({"src": src_files, "tgt": tgt_files})
