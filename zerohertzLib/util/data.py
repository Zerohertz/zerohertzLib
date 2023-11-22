import os
import shutil
from typing import Optional

from tqdm import tqdm

from .json import Json, JsonDir


class MakeData:
    """json 파일 내 값에 따라 data를 구축하는 함수

    Args:
        startDataPath (``str``): 목표 data가 존재하는 directory 경로
        startJsonPath (``str``): 목표 json 파일이 존재하는 directory 경로
        jsonKey (``str``): ``dataPath`` 에서 data의 파일명을 나타내는 key 값
        targetPath (``str``): Data 구축 경로
        endDataDir (``Optional[str]``): 구축될 data 파일들의 directory 이름
        endJsonDir (``Optional[str]``): 구축될 json 파일들의 directory 이름

    Attributes:
        gt (``zerohertzLib.util.JsonDir``): json 파일들을 읽어 data 구축 시 활용
        endDataPath (``str``): ``{targetPath}/{endDataDir}``
        endJsonPath (``str``): ``{targetPath}/{endJsonDir}``
    """

    def __init__(
        self,
        startDataPath: str,
        startJsonPath: str,
        jsonKey: str,
        targetPath: str,
        endDataDir: Optional[str] = "data",
        endJsonDir: Optional[str] = "json",
    ) -> None:
        self.startDataPath = startDataPath
        self.startJsonPath = startJsonPath
        self.gt = JsonDir(startJsonPath)
        self.jsonKey = self.gt._getKey(jsonKey)
        self.targetPath = targetPath
        self.endDataPath = os.path.abspath(os.path.join(self.targetPath, endDataDir))
        self.endJsonPath = os.path.abspath(os.path.join(self.targetPath, endJsonDir))

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
                        key = json_instance._getKey("supercategory_name")
                        category = json_instance._getValue(key)
                        return category == "CityCar" or category == "Mid-size car"

            Condition & Make Data:

            .. code-block:: python

                class MakeDataCarDamage(zz.util.MakeData):
                    def condition(self, json_instance):
                        annotations = json_instance.get("annotations")
                        return (annotations[0]["color"] in ["White", "Black"]) and (
                            json_instance.get("supercategory_name") == "CityCar"
                        )
                    def makeData(self, json_instance, dataName):
                        img = cv2.imread(os.path.join(self.startDataPath, dataName))
                        for i, ant in enumerate(json_instance["annotations"]):
                            label = ant["damage"]
                            if not label is None:
                                poly = ant["segmentation"]
                                poly = np.array(poly[0][0])
                                tmp = zz.vision.cutout(img, poly)
                                h, w, _ = tmp.shape
                                if 100 <= h <= 300 and 100 <= w <= 300:
                                    fileName = ".".join(dataName.split(".")[:-1]) + f"_{i}"
                                    xm, ym = poly[:, 0].min(), poly[:, 1].min()
                                    poly -= (xm, ym)
                                    cv2.imwrite(
                                        os.path.join(
                                            self.endDataPath,
                                            f"{fileName}.png",
                                        ),
                                        tmp,
                                    )
                                    zz.util.write_json(
                                        {"name": f"{fileName}.png", "poly": poly.tolist()},
                                        os.path.join(self.endJsonPath, fileName),
                                    )
        """
        return True

    def makeData(self, json_instance: Json, dataName: str) -> None:
        """Data 구축 방법 정의

        Args:
            json_instance (``zerohertzLib.util.Json``): ```Json`` instance의 정보
            dataName (``str``): ``jsonKey`` 에 의해 출력된 data의 이름

        Returns:
            ``None``: ``endDataPath`` 와 ``endJsonPath`` 와 본 함수를 통해 data 구축
        """
        try:
            shutil.copy(
                os.path.join(self.startDataPath, dataName),
                os.path.join(self.endDataPath, dataName),
            )
            shutil.copy(
                os.path.join(self.startJsonPath, json_instance.name),
                os.path.join(self.endJsonPath, json_instance.name),
            )
        except:
            print("Missing:\t", os.path.join(self.startDataPath, dataName))

    def make(self) -> None:
        """Data 구축 실행

        .. warning::

            실행 시 ``targetPath`` 삭제 후 구축 진행

        Examples:
            Condition:

            >>> mdc = MakeDataCar(startDataPath, startJsonPath, "file_name", targetPath)
            >>> mdc.make()
            100%|█████████████| 403559/403559 [00:54<00:00, 7369.96it/s]
            ====================================================================================================
            DATA PATH:       /.../data
            JSON PATH:       /.../json
            ====================================================================================================
            100%|█████████████| 403559/403559 [01:04<00:00, 6292.39it/s]

            Condition & Make Data:

            >>> mdcd = MakeDataCarDamage(startDataPath, startJsonPath, "file_name", targetPath)
            >>> mdcd.make()
            100%|█████████████| 50445/50445 [00:08<00:00, 6227.74it/s]
            ====================================================================================================
            DATA PATH:       /.../data
            JSON PATH:       /.../json
            ====================================================================================================
            100%|█████████████| 50445/50445 [14:56<00:00, 56.26it/s]
        """
        try:
            shutil.rmtree(self.targetPath)
        except:
            pass
        print("=" * 100)
        print("DATA PATH:\t", self.endDataPath)
        print("JSON PATH:\t", self.endJsonPath)
        print("=" * 100)
        os.makedirs(self.endDataPath, exist_ok=True)
        os.makedirs(self.endJsonPath, exist_ok=True)
        for json_instance in tqdm(self.gt):
            dataName = json_instance._getValue(self.jsonKey)
            if self.condition(json_instance):
                self.makeData(json_instance, dataName)
