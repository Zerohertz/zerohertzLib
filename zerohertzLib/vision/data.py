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
import shutil
import urllib.parse
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import DTypeLike, NDArray
from tqdm import tqdm

from zerohertzLib.util import Json, JsonDir, rmtree, write_json

from .convert import poly2cwh, poly2mask, poly2xyxy, xyxy2poly
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
    """Label Studio 관련 data를 handling하는 class

    Args:
        data_path (``str``): Image들이 존재하는 directory 경로
        json_path (``Optional[str]``): Label Studio에서 다른 format으로 변환할 시 사용될 annotation 정보가 담긴 JSON file

    Methods:
        __len__:
            Returns:
                ``int``: 읽어온 image file 혹은 annotation들의 수

        __getitem__:
            Args:
                idx (``int``): 입력 index

            Returns:
                ``Union[Tuple[str, Dict[str, Dict[str, str]]], Tuple[List[str], Dict[str, List[Any]]]]``: Index에 따른 image file 이름 또는 경로와 JSON file에 포함될 dictionary 또는 annotation 정보

    Examples:
        Without ``json_path``:

            >>> ls = zz.vision.LabelStudio(data_path)
            >>> ls[0]
            ('0000007864.png', {'data': {'image': 'data/local-files/?d=image/0000007864.png'}})
            >>> ls[1]
            ('0000008658.png', {'data': {'image': 'data/local-files/?d=image/0000008658.png'}})

        With ``json_path``:

            Bbox:

                >>> ls = zz.vision.LabelStudio(data_path, json_path)
                >>> ls[0]
                >>> ls[0]
                ('/PATH/TO/IMAGE', {'labels': ['label1', ...], 'polys': [array([0.39471694, 0.30683403, 0.03749811, 0.0167364 ]), ...], 'whs': [(1660, 2349), ...]})
                >>> ls[1]
                ('/PATH/TO/IMAGE', {'labels': ['label2', ...], 'polys': [array([0.29239837, 0.30149896, 0.04013469, 0.02736506]), ...], 'whs': [(1655, 2324), ...]})
                >>> ls.labels
                {'label1', 'label2'}
                >>> ls.type
                'rectanglelabels'

            Poly:

                >>> ls = zz.vision.LabelStudio(data_path, json_path)
                >>> ls[0]
                ('/PATH/TO/IMAGE', {'labels': ['label1', ...], 'polys': [array([[0.4531892 , 0.32880674], ..., [0.46119428, 0.32580483]]), ...], 'whs': [(3024, 4032), ...]})
                >>> ls[1]
                ('/PATH/TO/IMAGE', {'labels': ['label2', ...], 'polys': [array([[0.31973699, 0.14660367], ..., [0.29032053, 0.1484422 ]]), ...], 'whs': [(3024, 4032), ...]})
                >>> ls.labels
                {'label1', 'label2'}
                >>> ls.type
                'polygonlabels'
    """

    def __init__(
        self,
        data_path: str,
        json_path: Optional[str] = None,
    ) -> None:
        self.annotations = None
        if json_path is None:
            self.data_paths = _get_image_paths(data_path)
        else:
            self.annotations = Json(json_path)
            self.type = self.annotations[0]["annotations"][0]["result"][0]["type"]
        self.data_path = data_path
        self.labels = set()

    def __len__(self) -> int:
        if self.annotations is None:
            return len(self.data_paths)
        return len(self.annotations)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[str, Dict[str, Dict[str, str]]], Tuple[List[str], Dict[str, List[Any]]]
    ]:
        if self.annotations is None:
            file_name = self.data_paths[idx].split("/")[-1]
            return file_name, {
                "data": {"image": f"data/local-files/?d=image/{file_name}"}
            }
        file_name = self.annotations[idx]["data"]["image"].split("/")[-1]
        file_name = urllib.parse.unquote(file_name)
        if len(file_name) > 8 and "-" == file_name[8]:
            file_name = "-".join(file_name.split("-")[1:])
        file_path = os.path.join(self.data_path, file_name)
        if len(self.annotations[idx]["annotations"]) > 1:
            raise ValueError("The 'annotations' are plural")
        if self.type == "rectanglelabels":
            return file_path, self._dict2cwh(
                self.annotations[idx]["annotations"][0]["result"]
            )
        if self.type == "polygonlabels":
            return file_path, self._dict2poly(
                self.annotations[idx]["annotations"][0]["result"]
            )
        raise ValueError(f"Unknown annotation type: {self.type}")

    def _dict2cwh(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels, polys, whs = [], [], []
        for result in results:
            width, height = result["original_width"], result["original_height"]
            box_cwh = (
                np.array(
                    [
                        result["value"]["x"],
                        result["value"]["y"],
                        result["value"]["width"],
                        result["value"]["height"],
                    ]
                )
                / 100
            )
            if len(result["value"]["rectanglelabels"]) > 1:
                raise ValueError("The 'rectanglelabels' are plural")
            label = result["value"]["rectanglelabels"][0]
            labels.append(label)
            self.labels.add(label)
            polys.append(box_cwh)
            whs.append((width, height))
        return {"labels": labels, "polys": polys, "whs": whs}

    def _dict2poly(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels, polys, whs = [], [], []
        for result in results:
            width, height = result["original_width"], result["original_height"]
            box_poly = np.array(result["value"]["points"]) / 100
            if len(result["value"]["polygonlabels"]) > 1:
                raise ValueError("The 'polygonlabels' are plural")
            label = result["value"]["polygonlabels"][0]
            labels.append(label)
            self.labels.add(label)
            polys.append(box_poly)
            whs.append((width, height))
        return {"labels": labels, "polys": polys, "whs": whs}

    def json(
        self,
        data_function: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> None:
        """Label Studio에 mount된 data를 불러오기 위한 JSON file 생성

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
            + Absolute local path: ``/home/user/image`` (``data_path``: ``${PWD}/files/image``)
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
            data_function (``Optional[Callable[[str], Dict[str, Any]]]``): Label Studio에서 사용할 수 있는 ``data`` 항목 추가 함수 (예시 참고)

        Returns:
            ``None``: ``{data_path}.json`` 에 결과 저장

        Examples:
            Default:

                >>> ls = zz.vision.LabelStudio(data_path)
                >>> ls.json()
                100%|█████████████| 476/476 [00:00<00:00, 259993.32it/s

                .. code-block:: json

                    [
                        {
                            "data": {
                                "image": "data/local-files/?d=image/0000007864.png"
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

                >>> ls = zz.vision.LabelStudio(data_path)
                >>> ls.json(data_function)
                100%|█████████████| 476/476 [00:00<00:00, 78794.25it/s]

                .. code-block:: json

                    [
                        {
                            "data": {
                                "image": "data/local-files/?d=image/0000007864.png",
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
        """
        json_data = []
        for file_name, data in tqdm(self):
            if "aug" in file_name:
                continue
            if data_function is not None:
                data["data"].update(data_function(file_name))
            json_data.append(data)
        write_json(json_data, self.data_path)

    def yolo(self, target_path: str, label: Optional[List[str]] = None) -> None:
        """Label Studio로 annotation한 JSON data를 YOLO format으로 변환

        Args:
            target_path (``str``): YOLO format data가 저장될 경로
            label (``Optional[List[str]]``): Label Studio에서 사용한 label을 정수로 변환하는 list (index 사용)

        Returns:
            ``None``: ``{target_path}/images`` 및 ``{target_path}/labels`` 에 image와 `.txt` file 저장

        Examples:
            >>> ls = zz.vision.LabelStudio(data_path, json_path)
            >>> ls.yolo(target_path)
            >>> label = ["label1", "label2"]
            >>> ls.yolo(target_path, label)
        """
        if label is None:
            label = []
        os.makedirs(os.path.join(target_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_path, "labels"), exist_ok=True)
        for file_path, result in self:
            img_file_name = file_path.split("/")[-1]
            txt_file_name = ".".join(img_file_name.split(".")[:-1]) + ".txt"
            converted_gt = []
            for lab, poly in zip(result["labels"], result["polys"]):
                if self.type == "rectanglelabels":
                    box_cwh = poly
                elif self.type == "polygonlabels":
                    box_cwh = poly2cwh(poly)
                else:
                    raise ValueError(f"Unknown annotation type: {self.type}")
                if lab not in label:
                    label.append(lab)
                converted_gt.append(
                    f"{label.index(lab)} " + " ".join(map(str, box_cwh)) + "\n"
                )
            try:
                shutil.copy(
                    file_path, os.path.join(target_path, "images", img_file_name)
                )
                with open(
                    os.path.join(target_path, "labels", txt_file_name),
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.writelines(converted_gt)
            except FileNotFoundError:
                print(f"'{file_path}' is not found")

    def labelme(self, target_path: str, label: Optional[Dict[str, Any]] = None) -> None:
        """Label Studio로 annotation한 JSON data를 LabelMe format으로 변환

        Args:
            target_path (``str``): LabelMe format data가 저장될 경로
            label (``Optional[Dict[str, Any]]``): Label Studio에서 사용한 label을 변경하는 dictionary

        Returns:
            ``None``: ``{target_path}/images`` 및 ``{target_path}/labels`` 에 image와 JSON file 저장

        Examples:
            >>> ls = zz.vision.LabelStudio(data_path, json_path)
            >>> ls.labelme(target_path)
            >>> label = {"label1": "lab1", "label2": "lab2"}
            >>> ls.labelme(target_path, label)
        """
        if label is None:
            label = {}
        os.makedirs(os.path.join(target_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_path, "labels"), exist_ok=True)
        for file_path, result in self:
            img_file_name = file_path.split("/")[-1]
            json_file_name = ".".join(img_file_name.split(".")[:-1])
            converted_gt = []
            for lab, poly, wh in zip(result["labels"], result["polys"], result["whs"]):
                if self.type == "rectanglelabels":
                    box_xyxy = poly * (wh * 2)
                    box_xyxy[2:] += box_xyxy[:2]
                    box_poly = xyxy2poly(box_xyxy)
                elif self.type == "polygonlabels":
                    box_poly = poly * wh
                else:
                    raise ValueError(f"Unknown annotation type: {self.type}")
                converted_gt.append(
                    {
                        "label": label.get(lab, lab),
                        "points": box_poly.tolist(),
                        "shape_type": "polygon",
                    }
                )
            try:
                shutil.copy(
                    file_path, os.path.join(target_path, "images", img_file_name)
                )
                write_json(
                    {"shapes": converted_gt},
                    os.path.join(target_path, "labels", json_file_name),
                )
            except FileNotFoundError:
                print(f"'{file_path}' is not found")

    def classification(
        self,
        target_path: str,
        label: Optional[Dict[str, Any]] = None,
        rand: Optional[int] = 0,
        shrink: Optional[bool] = True,
        aug: Optional[int] = 1,
    ) -> None:
        """Label Studio로 annotation한 JSON data를 classification format으로 변환

        Args:
            target_path (``str``): Classification format data가 저장될 경로
            label (``Optional[Dict[str, Any]]``): Label Studio에서 사용한 label을 변경하는 dictionary
            rand (``Optional[int]``): Image crop 시 random 범위 추가
            shrink (``Optional[bool]``): ``rand`` 에 의한 crop 시 image의 수축 여부
            aug (``Optional[int]``): 한 annotation 당 저장할 image의 수

        Returns:
            ``None``: ``{target_path}/{label}/{img_file_name}_{idx}_{i}.{img_file_ext}`` 에 image 저장 (``idx``: annotation의 index, ``i``: ``rand`` 의 index)

        Examples:
            >>> ls = zz.vision.LabelStudio(data_path, json_path)
            >>> ls.classification(target_path)
            >>> label = {"label1": "lab1", "label2": "lab2"}
            >>> ls.classification(target_path, label, rand=10, aug=10, shrink=False)
        """
        if label is None:
            label = {}
        for file_path, result in self:
            img = cv2.imread(file_path)
            if img is None:
                print(f"'{file_path}' is not found")
                continue
            img_file = file_path.split("/")[-1].split(".")
            img_file_name = ".".join(img_file[:-1])
            img_file_ext = img_file[-1]
            for idx, (lab, poly, wh) in enumerate(
                zip(result["labels"], result["polys"], result["whs"])
            ):
                if self.type == "rectanglelabels":
                    box_xyxy = poly * (wh * 2)
                    box_xyxy[2:] += box_xyxy[:2]
                elif self.type == "polygonlabels":
                    box_poly = poly * wh
                    box_xyxy = poly2xyxy(box_poly)
                else:
                    raise ValueError(f"Unknown annotation type: {self.type}")
                os.makedirs(
                    os.path.join(target_path, label.get(lab, lab)), exist_ok=True
                )
                for i in range(aug):
                    bias = (2 * rand * (np.random.rand(4) - 0.5)).astype(np.int32)
                    if not shrink:
                        bias[:2] = -abs(bias[:2])
                        bias[2:] = abs(bias[2:])
                    x_0, y_0, x_1, y_1 = box_xyxy.astype(np.int32) + bias
                    try:
                        cv2.imwrite(
                            os.path.join(
                                target_path,
                                label.get(lab, lab),
                                f"{img_file_name}_{idx}_{i}.{img_file_ext}",
                            ),
                            img[y_0:y_1, x_0:x_1, :],
                        )
                    except cv2.error:
                        print(
                            f"Impossible crop ('x_0': {x_0}, 'y_0': {y_0}, 'x_1': {x_1}, 'y_1': {y_1})"
                        )
