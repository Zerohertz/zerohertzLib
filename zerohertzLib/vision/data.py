"""
MIT License

Copyright (c) 2023 Hyogeun Oh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import os
from glob import glob
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from zerohertzLib.util import Json, JsonDir


class ImageLoader:
    """경로와 image의 수를 지정하여 경로 내 image를 return하는 class

    Args:
        path (``Optional[str]``): Image들이 존재하는 경로
        cnt (``Optional[int]``): 호출 시 return 할 image의 수

    Attributes:
        image_paths (``List[str]``): 지정한 경로 내 image들의 경로

    Methods:
        __len__:
            Returns:
                ``int``: ``cnt`` 에 해당하는 image들의 수

        __getitem__:
            Args:
                idx (``int``): 입력 index

            Returns:
                ``Union[NDArray[np.uint8], List[NDArray[np.uint8]]]``: ``cnt`` 에 따라 단일 image 또는 image들의 list

    Examples:
        >>> il = zz.vision.ImageLoader()
        >>> len(il)
        27
        >>> type(il[0])
        <class 'numpy.ndarray'>
        >>> il = zz.vision.ImageLoader(cnt=4)
        >>> len(il)
        7
        >>> type(il[0])
        <class 'list'>
        >>> len(il[0])
        4
        >>> type(il[0][0])
        <class 'numpy.ndarray'>
    """

    def __init__(self, path: Optional[str] = "./", cnt: Optional[int] = 1) -> None:
        ext = (
            "jpg",
            "JPG",
            "jpeg",
            "JPEG",
            "png",
            "PNG",
            "tif",
            "TIF",
            "tiff",
            "TIFF",
        )
        self.cnt = cnt
        self.image_paths = []
        for ext_ in ext:
            self.image_paths += glob(os.path.join(path, f"*.{ext_}"))

    def __len__(self) -> int:
        return math.ceil(len(self.image_paths) / self.cnt)

    def __getitem__(
        self, idx: int
    ) -> Union[NDArray[np.uint8], List[NDArray[np.uint8]]]:
        if self.cnt == 1:
            return cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        return [
            cv2.imread(path, cv2.IMREAD_UNCHANGED)
            for path in self.image_paths[self.cnt * idx : self.cnt * (idx + 1)]
        ]


class JsonImageLoader:
    """JSON file을 통해 image와 JSON file 내 정보를 불러오는 class

    Args:
        data_path (``str``): 목표 data가 존재하는 directory 경로
        json_path (``str``): 목표 JSON file이 존재하는 directory 경로
        json_key (``str``): ``data_path`` 에서 data의 file 이름을 나타내는 key 값

    Attributes:
        json (``zerohertzLib.util.JsonDir``): JSON file들을 읽어 data 구축 시 활용

    Methods:
        __len__:
            Returns:
                ``int``: 읽어온 JSON file들의 수

        __getitem__:
            읽어온 JSON file들을 list와 같이 indexing 후 해당하는 image return

            Args:
                idx (``int``): 입력 index

            Returns:
                ``Tuple[NDArray[np.uint8], zerohertzLib.util.Json]``: Image와 JSON 내 정보

    Examples:
        >>> jil = zz.vision.JsonImageLoader(data_path, json_path, json_key)
        100%|█████████████| 17248/17248 [00:04<00:00, 3581.22it/s]
        >>> img, js = jil[10]
        >>> img.shape
        (600, 800, 3)
        >>> js.tree()
        └─ info
            └─ name
            └─ date_created
        ...
    """

    def __init__(
        self,
        data_path: str,
        json_path: str,
        json_key: str,
    ) -> None:
        self.data_path = data_path
        self.json_path = json_path
        self.json = JsonDir(json_path)
        self.json_key = self.json._get_key(json_key)

    def __len__(self) -> int:
        return len(self.json)

    def __getitem__(self, idx: int) -> Tuple[NDArray[np.uint8], Json]:
        data_name = self.json[idx].get(self.json_key)
        img = cv2.imread(os.path.join(self.data_path, data_name), cv2.IMREAD_UNCHANGED)
        return img, self.json[idx]
