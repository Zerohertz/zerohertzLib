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
import multiprocessing as mp
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import DTypeLike, NDArray
from tqdm import tqdm

from zerohertzLib.util import Json, JsonDir, rmtree, write_json

from .convert import poly2mask
from .data import _get_image_paths
from .visual import bbox, mask


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
                ``Union[Tuple[str, NDArray[np.uint8]], Tuple[List[str], List[NDArray[np.uint8]]]``: ``cnt`` 에 따른 file 경로 및 image 값

    Examples:
        >>> il = zz.vision.ImageLoader()
        >>> len(il)
        510
        >>> il[0][0]
        './1.2.410.200001.1.9999.1.20220513101953581.1.1.jpg'
        >>> il[0][1].shape
        (480, 640, 3)
        >>> il = zz.vision.ImageLoader(cnt=4)
        >>> len(il)
        128
        >>> il[0][0]
        ['./1.2.410.200001.1.9999.1.20220513101953581.1.1.jpg', '...', '...', '...']
        >>> il[0][1][0].shape
        (480, 640, 3)
        >>> len(il[0][0])
        4
        >>> len(il[0][1])
        4
    """

    def __init__(self, path: Optional[str] = "./", cnt: Optional[int] = 1) -> None:
        self.cnt = cnt
        self.image_paths = _get_image_paths(path)
        self.image_paths.sort()

    def __len__(self) -> int:
        return math.ceil(len(self.image_paths) / self.cnt)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[str, NDArray[np.uint8]], Tuple[List[str], List[NDArray[np.uint8]]]
    ]:
        if self.cnt == 1:
            return (
                self.image_paths[idx],
                cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED),
            )
        return (
            self.image_paths[self.cnt * idx : self.cnt * (idx + 1)],
            [
                cv2.imread(path, cv2.IMREAD_UNCHANGED)
                for path in self.image_paths[self.cnt * idx : self.cnt * (idx + 1)]
            ],
        )


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
        data_path (``Optional[str]``): Image가 존재하는 directory 경로
        txt_path (``Optional[str]``): YOLO format의 ``.txt`` 가 존재하는 directory 경로
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
        >>> yolo = zz.vision.YoloLoader(data_path, txt_path, poly=True, absolute=False, vis_path="tmp", class_color=class_color)
        >>> image, class_list, objects = yolo[0]
        >>> type(image)
        <class 'numpy.ndarray'>
        >>> class_list
        [1, 1]
        >>> len(objects)
        2
    """

    def __init__(
        self,
        data_path: Optional[str] = "images",
        txt_path: Optional[str] = "labels",
        poly: Optional[bool] = False,
        absolute: Optional[bool] = False,
        vis_path: Optional[str] = None,
        class_color: Optional[Dict[Union[int, str], Tuple[int]]] = None,
    ) -> None:
        self.data_path = data_path
        self.data_paths = _get_image_paths(self.data_path)
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
        img = cv2.imread(data_path)
        try:
            class_list, objects = self._convert(txt_path, img)
        except FileNotFoundError:
            print(f"'{data_file_name}' is not found")
            return None, None, None
        if self.vis_path is not None:
            self._visualization(data_file_name, img, class_list, objects)
        return img, class_list, objects

    def _convert(
        self, txt_path: str, img: NDArray[np.uint8]
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
                    obj *= img.shape[:2][::-1]
            else:
                obj = np.array(list(map(float, data_str[1:])))
                if not self.absolute:
                    obj *= img.shape[:2][::-1] * 2
            objects.append(obj)
        return class_list, objects

    def _visualization(
        self,
        file_name: str,
        img: NDArray[np.uint8],
        class_list: List[int],
        objects: List[NDArray[DTypeLike]],
    ) -> None:
        if self.poly:
            mks = np.zeros((len(objects), *img.shape[:2]), bool)
            for idx, poly in enumerate(objects):
                mks[idx] = poly2mask(poly, img.shape[:2])
            img = mask(img, mks, class_list=class_list, class_color=self.class_color)
        else:
            for cls, box in zip(class_list, objects):
                img = bbox(img, box, self.class_color[cls])
        cv2.imwrite(os.path.join(self.vis_path, file_name), img)

    def _value(
        self,
        img: NDArray[np.uint8],
        obj: NDArray[DTypeLike],
        labels: List[str],
        cls: int,
    ):
        original_height, original_width = img.shape[:2]
        obj *= 100
        if self.poly:
            obj /= (original_width, original_height)
            return {
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    "points": obj.tolist(),
                    "closed": True,
                    "polygonlabels": [labels[cls]],
                },
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels",
                "origin": "manual",
            }
        obj[:2] -= obj[2:] / 2
        obj /= (original_width, original_height) * 2
        return {
            "original_width": original_width,
            "original_height": original_height,
            "image_rotation": 0,
            "value": {
                "x": obj[0],
                "y": obj[1],
                "width": obj[2],
                "height": obj[3],
                "rectanglelabels": [labels[cls]],
            },
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "origin": "manual",
        }

    def _annotation(self, args: List[Union[int, str, List[str]]]) -> Dict[str, Any]:
        idx, directory, labels = args
        img, class_list, objects = self[idx]
        data_path = self.data_paths[idx]
        data_file_name = data_path.split("/")[-1]
        annotation = {
            "data": {"image": f"data/local-files/?d={directory}/{data_file_name}"}
        }
        result_data = []
        for cls, obj in zip(class_list, objects):
            result_data.append(self._value(img, obj, labels, cls))
        annotation["annotations"] = [{"result": result_data}]
        return annotation

    def labelstudio(
        self,
        directory: Optional[str] = "image",
        labels: Optional[List[str]] = None,
        mp_num: Optional[int] = 0,
    ) -> None:
        """
        YOLO format의 data를 Label Studio에서 확인 및 수정할 수 있게 변환

        Args:
            directory (``Optional[str]``): Label Studio 내 ``/home/user/{directory}`` 의 이름
            labels (``Optional[List[str]]``): YOLO format의 ``.txt`` 상에서 index에 따른 label의 이름
            mp_num (``Optional[int]``): 병렬 처리에 사용될 process의 수 (``0``: 직렬 처리)

        Returns:
            ``None``: ``{path}.json`` 으로 결과 저장

        Examples
            >>> yolo.labelstudio("images", mp_num=10, labels=["t1", "t2", "t3", "t4"])
        """
        if labels is None:
            labels = [str(i) for i in range(100)]
        json_data = []
        if mp_num == 0:
            for idx in range(len(self)):
                json_data.append(self._annotation([idx, directory, labels]))
        else:
            args = [[idx, directory, labels] for idx in range(len(self))]
            with mp.Pool(processes=mp_num) as pool:
                annotations = pool.map(self._annotation, args)
            for annotation in annotations:
                json_data.append(annotation)
        write_json(json_data, self.data_path)


class CocoLoader:
    """COCO format의 dataset을 읽고 시각화하는 class

    Args:
        data_path (``str``): Image 및 annotation이 존재하는 directory 경로
        vis_path (``Optional[str]``): 시각화 image들이 저장될 경로
        class_color (``Optional[Dict[Union[int, str], Tuple[int]]]``): 시각화 결과에 적용될 class에 따른 색상

    Methods:
        __len__:
            Returns:
                ``int``: 읽어온 image file들의 수

        __call__:
            Index에 따른 image와 annotation에 대한 정보 return (``vis_path`` 와 ``class_color`` 입력 시 시각화 image ``vis_path`` 에 저장)

            Args:
                idx (``int``): 입력 index
                read (``Optional[bool]``): Image 읽음 여부
                int_class (``Optional[bool]``): 출력될 class의 type 지정

            Returns:
                ``Tuple[Union[str, NDArray[np.uint8]], List[Union[int, str]], NDArray[DTypeLike], List[NDArray[DTypeLike]]]``: Image 경로 혹은 읽어온 image와 그에 따른 ``class_list``, ``bboxes``, ``polys``

        __getitem__:
            Index에 따른 image와 annotation에 대한 정보 return (``vis_path`` 와 ``class_color`` 입력 시 시각화 image ``vis_path`` 에 저장)

            Args:
                idx (``int``): 입력 index

            Returns:
                ``Tuple[NDArray[np.uint8], List[str], NDArray[DTypeLike], List[NDArray[DTypeLike]]]``: 읽어온 image와 그에 따른 ``class_list``, ``bboxes``, ``polys``

    Examples:
        >>> data_path = "train"
        >>> class_color = {"label1": (0, 255, 0), "label2": (255, 0, 0)}
        >>> coco = zz.vision.CocoLoader(data_path, vis_path="tmp", class_color=class_color)
        >>> image, class_list, bboxes, polys = coco(0, False, True)
        >>> type(image)
        <class 'str'>
        >>> image
        '{IMAGE_PATH}.jpg'
        >>> class_list
        [0, 1]
        >>> type(bboxes)
        <class 'numpy.ndarray'>
        >>> bboxes.shape
        (2, 4)
        >>> image, class_list, bboxes = coco[0]
        >>> type(image)
        <class 'numpy.ndarray'>
        >>> class_list
        ['label1', 'label2']
        >>> type(bboxes)
        <class 'numpy.ndarray'>
        >>> bboxes.shape
        (2, 4)
    """

    def __init__(
        self,
        data_path: str,
        vis_path: Optional[str] = None,
        class_color: Optional[Dict[Union[int, str], Tuple[int]]] = None,
    ) -> None:
        self.data_path = data_path
        data = Json(f"{data_path}.json")
        self.images = data["images"]
        self.annotations = data["annotations"]
        self.images.sort(key=lambda x: x["id"])
        self.annotations.sort(key=lambda x: x["image_id"])
        self.image2annotation = defaultdict(list)
        for idx, annotation in enumerate(self.annotations):
            self.image2annotation[annotation["image_id"]].append(idx)
        self.classes = {}
        for idx, cls in enumerate(data["categories"]):
            self.classes[cls["id"]] = (idx, cls["name"])
        self.vis_path = vis_path
        if vis_path is not None:
            if class_color is None:
                raise ValueError(
                    "Visualization requires the 'class_color' variable to be specified"
                )
            rmtree(vis_path)
            self.class_color = class_color

    def __len__(self) -> int:
        return len(self.images)

    def __call__(
        self, idx: int, read: Optional[bool] = False, int_class: Optional[bool] = False
    ) -> Tuple[
        Union[str, NDArray[np.uint8]],
        List[Union[int, str]],
        NDArray[DTypeLike],
        List[NDArray[DTypeLike]],
    ]:
        img_path = os.path.join(
            self.data_path, os.path.basename(self.images[idx]["file_name"])
        )
        if read:
            img = cv2.imread(img_path)
        else:
            img = img_path
        class_list = []
        bboxes = []
        polys = []
        for idx_ in self.image2annotation[self.images[idx]["id"]]:
            annotation = self.annotations[idx_]
            if int_class:
                class_list.append(self.classes[annotation["category_id"]][0])
            else:
                class_list.append(self.classes[annotation["category_id"]][1])
            bboxes.append(
                [
                    annotation["bbox"][0] + annotation["bbox"][2] / 2,
                    annotation["bbox"][1] + annotation["bbox"][3] / 2,
                    annotation["bbox"][2],
                    annotation["bbox"][3],
                ]
            )
            if "segmentation" in annotation.keys():
                polys.append(np.array(annotation["segmentation"][0]).reshape(-1, 2))
        bboxes = np.array(bboxes)
        return img, class_list, bboxes, polys

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        NDArray[np.uint8], List[str], NDArray[DTypeLike], List[NDArray[DTypeLike]]
    ]:
        img, class_list, bboxes, polys = self(idx, read=True)
        if self.vis_path is not None:
            self._visualization(
                os.path.basename(self.images[idx]["file_name"]),
                img,
                class_list,
                bboxes,
                polys,
            )
        return img, class_list, bboxes, polys

    def _visualization(
        self,
        file_name: str,
        img: NDArray[np.uint8],
        class_list: List[str],
        bboxes: NDArray[DTypeLike],
        polys: List[NDArray[DTypeLike]],
    ) -> None:
        for cls, box in zip(class_list, bboxes):
            img = bbox(img, box, self.class_color[cls])
        if polys:
            mks = np.zeros((len(polys), *img.shape[:2]), bool)
            for idx, poly in enumerate(polys):
                mks[idx] = poly2mask(poly, img.shape[:2])
            img = mask(img, mks, class_list=class_list, class_color=self.class_color)
        cv2.imwrite(os.path.join(self.vis_path, file_name), img)

    def yolo(
        self,
        target_path: str,
        label: Optional[List[str]] = None,
        poly: Optional[bool] = False,
    ) -> None:
        """COCO format을 YOLO format으로 변환

        Args:
            target_path (``str``): YOLO format data가 저장될 경로
            label (``Optional[List[str]]``): COCO에서 사용한 label을 정수로 변환하는 list (index 사용)
            poly (``Optional[bool]``): Segmentation format 유무

        Returns:
            ``None``: ``{target_path}/images`` 및 ``{target_path}/labels`` 에 image와 `.txt` file 저장

        Examples:
            >>> coco = zz.vision.CocoLoader(data_path)
            >>> coco.yolo(target_path)
            100%|█████████████| 476/476 [00:00<00:00, 78794.25it/s]
            >>> label = ["label1", "label2"]
            >>> cooc.yolo(target_path, label)
            100%|█████████████| 476/476 [00:00<00:00, 78794.25it/s]
        """
        rmtree(os.path.join(target_path, "images"))
        rmtree(os.path.join(target_path, "labels"))
        for idx in tqdm(range(len(self))):
            img_path, class_list, bboxes, polys = self(
                idx, read=False, int_class=label is None
            )
            converted_gt = []
            if poly:
                for cls, poly_ in zip(class_list, polys):
                    poly_ /= (self.images[idx]["width"], self.images[idx]["height"])
                    if label:
                        cls = label.index(cls)
                    converted_gt.append(
                        f"{cls} " + " ".join(map(str, poly_.reshape(-1)))
                    )
            else:
                for cls, box in zip(class_list, bboxes):
                    box /= (self.images[idx]["width"], self.images[idx]["height"]) * 2
                    if label:
                        cls = label.index(cls)
                    converted_gt.append(f"{cls} " + " ".join(map(str, box)))
            img_file_name = os.path.basename(img_path)
            txt_file_name = ".".join(img_file_name.split(".")[:-1]) + ".txt"
            try:
                shutil.copy(
                    img_path, os.path.join(target_path, "images", img_file_name)
                )
                with open(
                    os.path.join(target_path, "labels", txt_file_name),
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.writelines("\n".join(converted_gt))
            except FileNotFoundError:
                print(f"'{img_path}' is not found")
