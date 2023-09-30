import hashlib
import json
import pathlib

import checksumdir
import pandas as pd


def dict_hash(cfg: dict) -> str:
    cfg_str = json.dumps(cfg)

    hash_str = hashlib.md5(cfg_str.encode("utf-8")).hexdigest()

    return hash_str


def file_hash(fpath: str) -> str:
    hash_str = hashlib.md5(pathlib.Path(fpath).read_bytes()).hexdigest()

    return hash_str


def dir_hash(dpath: str) -> str:
    hash_str = checksumdir.dirhash(dpath)

    return hash_str


def dataframe_hash(df: pd.DataFrame) -> str:
    rec = df.to_records(index=False).tostring()
    hash_str = hashlib.md5(rec).hexdigest()

    return hash_str
