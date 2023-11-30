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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import DTypeLike, NDArray
from tqdm import tqdm

from zerohertzLib.util import Json, JsonDir, rmtree, write_json

from .convert import poly2mask
from .visual import bbox, masks


def _get_image_paths(path):
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
    image_paths = []
    for ext_ in ext:
        image_paths += glob(os.path.join(path, f"*.{ext_}"))
    return image_paths


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
        self.cnt = cnt
        self.image_paths = _get_image_paths(path)

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


class YoloLoader:
    """YOLO format의 dataset을 읽고 시각화하는 class

    Args:
        data_path (``str``): Image가 존재하는 directory 경로
        txt_path (``str``): YOLO format의 ``.txt`` 가 존재하는 directory 경로
        poly (``Optional[bool]``): ``.txt`` file의 format (``False``: detection, ``True``: segmentation)
        absolute (``Optional[bool]``): ``.txt`` file의 절대 좌표계 여부 (``False``: relative coordinates, ``True``: absolute coordinates)
        vis_path (``Optional[str]``): 시각화 image들이 저장될 경로
        class_color (``Optional[Dict[Union[int, str], Tuple[int]]]``): 시각화 결과에 적용될 class에 따른 색상

    Methods:
        __len__:
            Returns:
                ``int``: 읽어온 image file들의 수

        __getitem__:
            Index에 따른 image와 ``.txt`` file에 대한 정보 return (``vis_path`` 와 ``class_color`` 입력 시 시각화 image ``vis_path`` 에 저장)

            Args:
                idx (``int``): 입력 index

            Returns:
                ``Tuple[NDArray[np.uint8], List[int], List[NDArray[DTypeLike]]]``: 읽어온 image와 그에 따른 ``class_list`` 및 ``bbox`` 혹은 ``poly``

    Examples:
        >>> data_path = ".../images"
        >>> txt_path = ".../labels"
        >>> class_color = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
        >>> yololoader = YoloLoader(data_path, txt_path, poly=True, absolute=False, vis_path="tmp", class_color=class_color)
        >>> image, class_list, objects = yololoader[0]
        >>> type(image)
        <class 'numpy.ndarray'>
        >>> class_list
        [1, 1]
        >>> len(objects)
        2
    """

    def __init__(
        self,
        data_path: str,
        txt_path: str,
        poly: Optional[bool] = True,
        absolute: Optional[bool] = False,
        vis_path: Optional[str] = None,
        class_color: Optional[Dict[Union[int, str], Tuple[int]]] = None,
    ) -> None:
        self.data_paths = _get_image_paths(data_path)
        self.txt_path = txt_path
        self.poly = poly
        self.absolute = absolute
        self.vis_path = vis_path
        if vis_path is not None:
            if class_color is None:
                raise ValueError(
                    "Visualization requires the 'class_color' variable to be specified"
                )
            rmtree(vis_path)
            self.class_color = class_color

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[NDArray[np.uint8], List[int], List[NDArray[DTypeLike]]]:
        data_path = self.data_paths[idx]
        data_file_name = data_path.split("/")[-1]
        txt_path = os.path.join(
            self.txt_path, ".".join(data_file_name.split(".")[:-1]) + ".txt"
        )
        image = cv2.imread(data_path)
        try:
            class_list, objects = self._convert(txt_path, image)
        except FileNotFoundError:
            return None, None, None
        if self.vis_path is not None:
            self._visualization(data_file_name, image, class_list, objects)
        return image, class_list, objects

    def _convert(
        self, txt_path: str, image: NDArray[np.uint8]
    ) -> Tuple[List[int], List[NDArray[DTypeLike]]]:
        class_list = []
        objects = []
        with open(txt_path, "r", encoding="utf-8") as file:
            data_lines = file.readlines()
        for data_line in data_lines:
            data_str = data_line.strip().split(" ")
            class_list.append(int(data_str[0]))
            if self.poly:
                obj = np.array(list(map(float, data_str[1:]))).reshape(-1, 2)
                if not self.absolute:
                    obj *= image.shape[:2][::-1]
            else:
                obj = np.array(list(map(float, data_str[1:])))
                if not self.absolute:
                    obj *= image.shape[:2][::-1] * 2
            objects.append(obj)
        return class_list, objects

    def _visualization(
        self,
        file_name: str,
        image: NDArray[np.uint8],
        class_list: List[int],
        objects: List[NDArray[DTypeLike]],
    ) -> None:
        mks = np.zeros((len(objects), *image.shape[:2]), bool)
        if self.poly:
            for idx, poly in enumerate(objects):
                mks[idx] = poly2mask(poly, image.shape[:2])
            image = masks(
                image, mks, class_list=class_list, class_color=self.class_color
            )
        else:
            for idx, (cls, box) in enumerate(zip(class_list, objects)):
                image = bbox(image, box, self.class_color[cls])
        cv2.imwrite(os.path.join(self.vis_path, file_name), image)


class LabelStudio:
    """Label Studio에 mount된 data를 불러오기 위한 JSON file 생성 class

    Note:
        아래와 같이 환경 변수가 설정된 Label Studio image를 사용하고 ``/home/user`` 에 ``image`` directory가 mount 되어야 ``LabelStudio`` class로 생성된 JSON file을 적용할 수 있다.

        .. code-block:: docker

            FROM heartexlabs/label-studio

            ENV LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
            ENV LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/user

        .. code-block:: bash

            docker run --name label-studio -p 8080:8080 -v ${PWD}/files:/home/user label-studio

        ``Projects`` → ``{PROJECT_NAME}`` → ``Settings`` → ``Cloud Storage`` → ``Add Source Storage`` 클릭 후 아래와 같이 정보를 기재하고 ``Sync Storage`` 를 누른다.

        + Storage Type: ``Local files``
        + Absolute local path: ``/home/user/image``
        + File Filter Regex: ``^.*\.(jpe?g|png|tiff?)$``
        + Treat every bucket object as a source file: ``True``

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/286834432-462ad17b-d78f-4490-909b-a54debdf719a.png
            :alt: Label Studio Setup
            :align: center
            :width: 400px

        Sync 이후 ``LabelStudio`` class로 생성된 JSON file을 Label Studio에 import하면 아래와 같이 setup 할 수 있다.

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/286842007-1fb780eb-7275-4569-a662-a4545be2e348.png
            :alt: Label Studio Setup
            :align: center
            :width: 400px

    Args:
        data_path (``str``): Image가 존재하는 directory 경로 (``${PWD}/files/image``)
        data_function (``Optional[Callable[[str], Dict[str, Any]]]``): Label Studio에서 사용할 수 있는 ``data`` 항목 추가 함수 (예시 참고)
        ingress (``Optional[str]``): Label Studio의 URL

    Methods:
        __len__:
            Returns:
                ``int``: 읽어온 image file들의 수

        __getitem__:
            Index에 따른 image file 이름과 JSON file에 포함될 dictionary return

            Args:
                idx (``int``): 입력 index

            Returns:
                ``Tuple[str, Dict[str, Dict[str, str]]]``: Image file의 이름과 생성된 ``data`` dictionary
    """

    def __init__(
        self,
        data_path: str,
        data_function: Optional[Callable[[str], Dict[str, Any]]] = None,
        ingress: Optional[str] = "",
    ) -> None:
        self.data_path = data_path
        self.data_paths = _get_image_paths(data_path)
        self.data_function = data_function
        self.ingress = ingress

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[str, Dict[str, Dict[str, str]]]:
        file_name = self.data_paths[idx].split("/")[-1]
        return file_name, {
            "data": {"image": f"{self.ingress}/data/local-files/?d=image/{file_name}"}
        }

    def make(self) -> None:
        """Label Studio에 mount된 data를 불러오기 위한 JSON file 생성

        Returns:
            ``None``: ``{data_path}.json`` 에 결과 저장

        Examples:
            Default:

                >>> ls = LabelStudio(data_path)
                >>> ls[0]
                ('0000008316.jpg', {'data': {'image': '/data/local-files/?d=image/0000008316.jpg'}})
                >>> ls[1]
                ('0000006339.jpg', {'data': {'image': '/data/local-files/?d=image/0000006339.jpg'}})
                >>> ls.make()
                100%|█████████████| 15445/15445 [00:00<00:00, 65420.84it/s]

                .. code-block:: json

                    [
                        {
                            "data": {
                                "image": "/data/local-files/?d=image/0000008316.jpg"
                            }
                        },
                        {
                            "data": {
                                "...": "..."
                            }
                        },
                    ]

            With ``data_function``:

                .. code-block:: python

                    def data_function(file_name):
                        return data_store[file_name]

                >>> ls = LabelStudio(data_path, ingress="https://test.zerohertz.xyz")
                >>> ls.make()
                100%|█████████████| 15445/15445 [00:00<00:00, 65420.84it/s]

                .. code-block:: json

                    [
                        {
                            "data": {
                                "image": "/data/local-files/?d=image/0000008316.jpg"
                                "Label": "...",
                                "patient_id": "...",
                                "file_name": "...",
                                "외래/입원": "...",
                                "성별": "...",
                                "나이(세)": "...",
                                "나이(개월)": "...",
                                "나이(세월)": "...",
                                "나이대": "...",
                                "...": "...",
                            }
                        },
                        {
                            "data": {
                                "...": "..."
                            }
                        },
                    ]

            With Ingress:

                >>> ls = LabelStudio(data_path, ingress="https://test.zerohertz.xyz")
                >>> ls.make()
                100%|█████████████| 15445/15445 [00:00<00:00, 65420.84it/s]

                .. code-block:: json

                    [
                        {
                            "data": {
                                "image": "https://test.zerohertz.xyz/data/local-files/?d=image/0000008316.jpg"
                            }
                        },
                        {
                            "data": {
                                "...": "..."
                            }
                        },
                    ]
        """
        json_data = []
        for file_name, data in tqdm(self):
            if "aug" in file_name:
                continue
            if self.data_function is not None:
                data["data"].update(self.data_function(file_name))
            json_data.append(data)
        write_json(json_data, self.data_path)
