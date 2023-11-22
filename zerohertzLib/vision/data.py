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

import os
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from zerohertzLib.util import Json, JsonDir


class JsonImageLoader:
    """JSON file을 통해 image와 JSON file 내 정보를 불러오는 class

    Args:
        data_path (``str``): 목표 data가 존재하는 directory 경로
        json_path (``str``): 목표 JSON file이 존재하는 directory 경로
        json_key (``str``): ``data_path`` 에서 data의 file 이름을 나타내는 key 값

    Attributes:
        json (``zerohertzLib.util.JsonDir``): JSON file들을 읽어 data 구축 시 활용

    Methods:
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

    def __getitem__(self, idx: int) -> Tuple[NDArray[np.uint8], Json]:
        data_name = self.json[idx].get(self.json_key)
        img = cv2.imread(os.path.join(self.data_path, data_name), cv2.IMREAD_UNCHANGED)
        return img, self.json[idx]
