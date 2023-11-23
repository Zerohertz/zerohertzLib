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
import shutil
from typing import Optional

from tqdm import tqdm

from .json import Json, JsonDir


class MakeData:
    """JSON file 내 값에 따라 data를 구축하는 함수

    Args:
        start_data_path (``str``): 목표 data가 존재하는 directory 경로
        start_json_path (``str``): 목표 JSON file이 존재하는 directory 경로
        json_key (``str``): ``dataPath`` 에서 data의 file 이름을 나타내는 key 값
        target_path (``str``): Data 구축 경로
        end_data_dir (``Optional[str]``): 구축될 data file들의 directory 이름
        end_json_dir (``Optional[str]``): 구축될 JSON file들의 directory 이름

    Attributes:
        json (``zerohertzLib.util.JsonDir``): JSON file들을 읽어 data 구축 시 활용
        end_data_path (``str``): ``{target_path}/{end_data_dir}``
        end_json_path (``str``): ``{target_path}/{end_json_dir}``
    """

    def __init__(
        self,
        start_data_path: str,
        start_json_path: str,
        json_key: str,
        target_path: str,
        end_data_dir: Optional[str] = "data",
        end_json_dir: Optional[str] = "json",
    ) -> None:
        self.start_data_path = start_data_path
        self.start_json_path = start_json_path
        self.json = JsonDir(start_json_path)
        self.json_key = self.json._get_key(json_key)
        self.target_path = target_path
        self.end_data_path = os.path.abspath(
            os.path.join(self.target_path, end_data_dir)
        )
        self.end_json_path = os.path.abspath(
            os.path.join(self.target_path, end_json_dir)
        )

    def condition(self, json_instance: Json) -> bool:
        """Data 구축 시 filtering 될 조건

        Args:
            json_instance (``zerohertzLib.util.Json``): ``Json`` instance의 정보

        Returns:
            ``bool``: Data 포함 여부

        아래와 같이 상속을 통해 조건을 설정할 수 있다.

        Examples:
            Condition:

            .. code-block:: python

                class MakeDataCar(zz.util.MakeData):
                    def condition(self, json_instance):
                        key = json_instance._get_key("supercategory_name")
                        category = json_instance._get_value(key)
                        return category == "CityCar" or category == "Mid-size car"

            Condition & Make Data:

            .. code-block:: python

                class MakeDataCarDamage(zz.util.MakeData):
                    def condition(self, json_instance):
                        annotations = json_instance.get("annotations")
                        return (annotations[0]["color"] in ["White", "Black"]) and (
                            json_instance.get("supercategory_name") == "CityCar"
                        )
                    def make_data(self, json_instance, data_name):
                        img = cv2.imread(os.path.join(self.start_data_path, data_name))
                        for i, ant in enumerate(json_instance["annotations"]):
                            label = ant["damage"]
                            if not label is None:
                                poly = ant["segmentation"]
                                poly = np.array(poly[0][0])
                                tmp = zz.vision.cutout(img, poly)
                                h, w, _ = tmp.shape
                                if 100 <= h <= 300 and 100 <= w <= 300:
                                    fileName = ".".join(data_name.split(".")[:-1]) + f"_{i}"
                                    xm, ym = poly[:, 0].min(), poly[:, 1].min()
                                    poly -= (xm, ym)
                                    cv2.imwrite(
                                        os.path.join(
                                            self.end_data_path,
                                            f"{fileName}.png",
                                        ),
                                        tmp,
                                    )
                                    zz.util.write_json(
                                        {"name": f"{fileName}.png", "poly": poly.tolist()},
                                        os.path.join(self.end_json_path, fileName),
                                    )

            Make Data:

            .. code-block:: python

                class MakeDataCarAugment(zz.util.MakeData):
                    def make_data(self, json_instance, data_name):
                        img = cv2.imread(
                            random.choice(glob("*"))
                        )
                        target = cv2.imread(
                            os.path.join(self.start_data_path, data_name), cv2.IMREAD_UNCHANGED
                        )
                        H, W, _ = img.shape
                        h, w, _ = target.shape
                        x, y = random.randrange(100, W - w - 100), random.randrange(100, H - h - 100)
                        box = [x, y, x + w, y + h]
                        img = zz.vision.paste(img, target, box, False, False)
                        fileName = ".".join(data_name.split(".")[:-1])
                        cv2.imwrite(
                            os.path.join(
                                self.end_data_path,
                                f"{fileName}.png",
                            ),
                            img,
                        )
        """
        return True

    def make_data(self, json_instance: Json, data_name: str) -> None:
        """Data 구축 방법 정의

        Args:
            json_instance (``zerohertzLib.util.Json``): ``Json`` instance의 정보
            data_name (``str``): ``json_key`` 에 의해 출력된 data의 이름

        Returns:
            ``None``: ``end_data_path``, ``end_json_path`` 와 본 함수를 통해 data 구축
        """
        try:
            shutil.copy(
                os.path.join(self.start_data_path, data_name),
                os.path.join(self.end_data_path, data_name),
            )
            shutil.copy(
                os.path.join(self.start_json_path, json_instance.name),
                os.path.join(self.end_json_path, json_instance.name),
            )
        except FileNotFoundError:
            print("Missing:\t", os.path.join(self.start_data_path, data_name))

    def make(self) -> None:
        """Data 구축 실행

        .. warning::

            실행 시 ``target_path`` 삭제 후 구축 진행

        Examples:
            >>> md = MakeData(start_data_path, start_json_path, json_key, target_path)
            >>> md.make()
            100%|█████████████| 403559/403559 [00:54<00:00, 7369.96it/s]
            ====================================================================================================
            DATA PATH:       /.../data
            JSON PATH:       /.../json
            ====================================================================================================
            100%|█████████████| 403559/403559 [01:04<00:00, 6292.39it/s]
        """
        try:
            shutil.rmtree(self.target_path)
        except FileNotFoundError:
            pass
        print("=" * 100)
        print("DATA PATH:\t", self.end_data_path)
        print("JSON PATH:\t", self.end_json_path)
        print("=" * 100)
        os.makedirs(self.end_data_path, exist_ok=True)
        os.makedirs(self.end_json_path, exist_ok=True)
        for json_instance in tqdm(self.json):
            data_name = json_instance._get_value(self.json_key)
            if self.condition(json_instance):
                self.make_data(json_instance, data_name)
