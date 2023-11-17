import os
import shutil

from tqdm import tqdm

from .json import Json, JsonDir


class MakeData:
    """json 파일 내 값에 따라 data를 구축하는 함수

    Args:
        dataPath (``str``): 목표 data가 존재하는 directory 경로
        jsonPath (``str``): 목표 json 파일이 존재하는 directory 경로
        jsonKey (``str``): ``dataPath`` 에서 data의 파일명을 나타내는 key 값
        targetPath (``str``): Data 구축 경로

    Attributes:
        gt (``zerohertzLib.util.JsonDir``): json 파일들을 읽어 data 구축 시 활용
        daPath (``str``): ``{targetPath}/data``
        gtPath (``str``): ``{targetPath}/json``
    """

    def __init__(
        self,
        dataPath: str,
        jsonPath: str,
        jsonKey: str,
        targetPath: str,
    ) -> None:
        self.dataPath = dataPath
        self.jsonPath = jsonPath
        self.gt = JsonDir(jsonPath)
        self.jsonKey = self.gt._getKey(jsonKey)
        self.targetPath = targetPath
        self.daPath = os.path.abspath(os.path.join(self.targetPath, "data"))
        self.gtPath = os.path.abspath(os.path.join(self.targetPath, "json"))

    def condition(self, json_instance: Json) -> bool:
        """Data 구축 시 filtering 될 조건

        Args:
            json_instance (``zerohertzLib.util.Json``): ``Json`` instance의 정보를 통해 구축할 data에 포함시킬지 여부 결정

        Returns:
            ``bool``: Data 포함 여부

        아래와 같이 상속을 통해 조건을 설정할 수 있다.

        Examples:
            .. code-block:: python

                class MakeDataCar(MakeData):
                    def condition(self, json_instance):
                        key = json_instance._getKey("supercategory_name")
                        category = json_instance._getValue(key)
                        return category == "CityCar" or category == "Mid-size car"
        """
        return True

    def make(self) -> None:
        """Data 구축 실행

        .. warning::

            실행 시 ``targetPath`` 삭제 후 구축 진행

        Examples:
            >>> mdc = MakeDataCar(dataPath, jsonPath, jsonKey, targetPath)
            >>> mdc.make()
            100%|█████████████| 403559/403559 [00:54<00:00, 7369.96it/s]
            ====================================================================================================
            GT PATH:         /.../data
            DATA PATH:       /.../json
            ====================================================================================================
            100%|█████████████| 403559/403559 [01:04<00:00, 6292.39it/s]
        """
        try:
            shutil.rmtree(self.targetPath)
        except:
            pass
        print("=" * 100)
        print("GT PATH:\t", self.daPath)
        print("DATA PATH:\t", self.gtPath)
        print("=" * 100)
        os.makedirs(self.daPath, exist_ok=True)
        os.makedirs(self.gtPath, exist_ok=True)
        for json_instance in tqdm(self.gt):
            dataName = json_instance._getValue(self.jsonKey)
            if self.condition(json_instance):
                try:
                    shutil.copy(
                        os.path.join(self.dataPath, dataName),
                        os.path.join(self.daPath, dataName),
                    )
                    shutil.copy(
                        os.path.join(self.jsonPath, json_instance.name),
                        os.path.join(self.gtPath, json_instance.name),
                    )
                except:
                    print("Missing:\t", os.path.join(self.dataPath, dataName))
