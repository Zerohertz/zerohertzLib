import os
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

import zerohertzLib as zz


class JsonImageLoader:
    """JSON file을 통해 image와 JSON file 내 정보를 불러오는 class

    Args:
        dataPath (``str``): 목표 data가 존재하는 directory 경로
        jsonPath (``str``): 목표 JSON file이 존재하는 directory 경로
        jsonKey (``str``): ``dataPath`` 에서 data의 file 이름을 나타내는 key 값

    Attributes:
        gt (``zerohertzLib.util.JsonDir``): JSON file들을 읽어 data 구축 시 활용

    Methods:
        __getitem__:
            읽어온 JSON file들을 list와 같이 indexing 후 해당하는 image return

            Args:
                idx (``int``): 입력 index

            Returns:
                ``Tuple[NDArray[np.uint8], zerohertzLib.util.Json]``: Image와 JSON 내 정보

    Examples:
        >>> jil = zz.vision.JsonImageLoader(dataPath, jsonPath, jsonKey)
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
        dataPath: str,
        jsonPath: str,
        jsonKey: str,
    ) -> None:
        self.dataPath = dataPath
        self.jsonPath = jsonPath
        self.gt = zz.util.JsonDir(jsonPath)
        self.jsonKey = self.gt._getKey(jsonKey)

    def __getitem__(self, idx: int) -> Tuple[NDArray[np.uint8], zz.util.Json]:
        dataName = self.gt[idx].get(self.jsonKey)
        img = cv2.imread(os.path.join(self.dataPath, dataName), cv2.IMREAD_UNCHANGED)
        return img, self.gt[idx]
