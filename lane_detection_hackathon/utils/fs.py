import types

import json

import cv2
import imageio
import numpy as np
import yaml
from tqdm import tqdm

from .image import is_gray


def read_image(fpath: str, gray_scale: bool = False) -> np.ndarray:
    flag = None
    if gray_scale:
        flag = cv2.IMREAD_GRAYSCALE

    img = cv2.imread(fpath, flags=flag)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def write_image(fpath: str, img: np.array):
    if not is_gray(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(fpath, img)


def read_yaml(fpath: str) -> dict:
    with open(fpath) as fp:
        data = yaml.safe_load(fp)
    return data


def save_yaml(fpath: str, data: dict):
    with open(fpath, "w") as fp:
        yaml.safe_dump(data, fp)


def read_json(fpath: str) -> dict:
    with open(fpath) as fp:
        data = json.load(fp)
    return data[0]


def save_json(fpath: str, data: dict):
    with open(fpath, "w") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


class VideoProcessor:
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self._stream.close()


class VideoReader(VideoProcessor):
    fps_key = "fps"
    size_key = "size"
    nframe_key = "nframes"

    def __init__(self, fpath: str):
        self._stream = imageio.get_reader(fpath, "ffmpeg")
        self._meta_data = self._stream.get_meta_data()

    @property
    def fps(self) -> float:
        return self._meta_data[self.fps_key]

    @property
    def size(self) -> tuple[int, int]:
        return self._meta_data[self.size_key]

    @property
    def frame_number(self) -> int:
        return self._meta_data[self.nframe_key]

    @property
    def duration_s(self) -> float:
        frame_n = int(self._stream.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_count = self._stream.get(cv2.CAP_PROP_FPS)
        return frame_n / fps_count

    def get_frames(self):
        for frame in self._stream.iter_data():
            frame = np.asarray(frame)
            yield frame


class VideoWriter(VideoProcessor):
    """Class implementation for video conversion from frames."""

    def __init__(self, fpath: str, fps: int = 50, verbose: bool = False):
        self._stream = imageio.get_writer(fpath, fps=int(fps))
        self._verbose = verbose

    def write_frame(self, frame: types.Image):
        """Write single frame."""

        self._stream.append_data(frame)

    def write(self, frames: list[types.Image]):
        """Write list of frames."""

        stream = frames
        if self._verbose:
            stream = tqdm(frames, desc="Frames processing")

        for frame in stream:
            self.write_frame(frame)
