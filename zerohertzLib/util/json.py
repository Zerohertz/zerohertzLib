import json
import os
from glob import glob
from typing import Any, Dict, List, Optional, Set, Union

from tqdm import tqdm


class Json:
    """json 형식 파일을 읽고 사용하기 위한 클래스

    객체 생성 시 ``path`` 를 입력하지 않을 시 현재 경로에 존재하는 json 파일을 읽고 ``path`` 를 경로로 입력하면 해당 경로에 존재하는 json 파일을 읽는다.

    Args:
        path (``Optional[str]``): json 파일의 경로

    Attributes:
        name (``str``): json 파일명
        keys (``List[str]``): 직렬화된 json의 key 값들

    Methods:
        __getitem__:
            읽어온 json 파일에 key 값 입력

            Args:
                key (``str``): 읽어온 json 파일에서 불러올 key 값

            Returns:
                ``Any``: Key에 따른 value 값

        _getKey:
            Key의 경로를 찾아주는 메서드

            Args:
                key (``str``): 읽어온 json 파일에서 불러올 key 값 (깊이 무관)

            Returns:
                ``str``: ``/`` 으로 깊이를 표시한 key 값

        _getValue:
            ``Json._getKey`` 로 생성된 key 값을 입력 받아 value return

            Args:
                key (``str``): ``Json._getKey`` 로 생성된 key 값

            Returns:
                ``Any``: Key에 따른 value

    Examples:
        >>> js = zz.util.Json()
        >>> js["title"]
        '[v0.2.3] Release'
        >>> key = js._getKey("language")
        >>> key
        'head/repo/language'
        >>> js._getValue(key)
        'Python'
        >>> js.name
        '65.json'
        >>> js.keys
        ['url', 'id', ..., 'user', 'user/login', 'user/id', ...]
    """

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = glob("*.json")[0]
        elif not path.endswith(".json"):
            path = glob(f"{path}/*.json")[0]
        self.name = path.replace(os.path.dirname(path), "").replace("/", "")
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.keys = []
        self.map = []

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __getKeys(self, data: Any, key: str, cnt: int):
        if isinstance(data, dict):
            for k, v in data.items():
                self.map.append(" " * 4 * cnt + "└─ " + str(k))
                if key is None:
                    self.keys.append(f"{k}")
                    self.__getKeys(v, f"{k}", cnt + 1)
                else:
                    self.keys.append(f"{key}/{k}")
                    self.__getKeys(v, f"{key}/{k}", cnt + 1)

    def _getKeys(self, key: str = None, cnt=0):
        if self.keys == [] and self.map == []:
            self.__getKeys(self.data, key, cnt)
        return self.keys

    def _getKey(self, key: str) -> str:
        keys = self._getKeys()
        if key in keys:
            return key
        if not "/" in key:
            for k in keys:
                if k.endswith(key):
                    key = k
                    break
        return key

    def _getValue(self, key: str) -> Any:
        value = self.data
        keys = key.split("/")
        for key in keys:
            value = value.get(key)
        return value

    def get(self, key: str) -> Any:
        """``Json._getKey`` 로 생성된 key 값을 입력 받아 value return

        Args:
            key (``str``): 읽어온 json 파일에서 불러올 key 값 (깊이 무관)

        Returns:
            ``Any``: Key에 따른 value

        Examples:
            >>> js["title"]
            '[v0.2.3] Release'
            >>> js.get("title")
            '[v0.2.3] Release'
            >>> js["language"]
            Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "/home/zerohertz/anaconda3/lib/python3.8/site-packages/zerohertzLib/util/json.py", line 65, in __getitem__
                return self.data[key]
            KeyError: 'language'
            >>> js.get("language")
            'Python'
        """
        return self._getValue(self._getKey(key))

    def tree(self) -> None:
        """json의 구조를 출력하는 메서드

        Examples:
            >>> js.tree()
            └─ url
            └─ id
            ...
            └─ user
                └─ login
                └─ id
            ...
        """
        self._getKeys()
        print("\n".join(self.map))


class JsonDir:
    """입력된 경로에 존재하는 json 형식 파일들을 읽고 사용하기 위한 클래스

    객체 생성 시 ``path`` 를 입력하지 않을 시 현재 경로에 존재하는 json 파일들을 읽는다.

    Args:
        path (``Optional[str]``): json 파일의 경로

    Attributes:
        name (``str``): 읽어온 json 파일명들
        data (``Dict[str, zerohertzLib.util.Json]``): 파일명에 따른 `Json` 객체 배열

    Methods:
        __len__:
            Returns:
                ``int``: 읽어온 json 파일들의 수

        __getitem__:
            읽어온 json 파일들을 list와 같이 indexing

            Args:
                idx (``int``): 입력 index

            Returns:
                ``zerohertzLib.util.Json``: Index에 따른 ``Json`` instance

        _getKey:
            Key의 경로를 찾아주는 메서드

            Args:
                key (``str``): 읽어온 json 파일에서 불러올 key 값 (깊이 무관)

            Returns:
                ``str``: ``/`` 으로 깊이를 표시한 key 값

    Examples:
        >>> jsd = zz.util.JsonDir()
        100%|█████████████| 5/5 [00:00<00:00, 3640.26it/s]
        >>> len(jsd)
        5
        >>> jsd[0]
        <zerohertzLib.util.json.Json object at 0x7f2562b83d00>
        >>> jsd[0]["title"]
        '[v0.2.3] Release'
        >>> jsd._getKey("language")
        'head/repo/language'
    """

    def __init__(self, path: Optional[str] = "") -> None:
        if path.endswith(".json"):
            raise Exception("Error: path = '*.json'")
        self.data = {}
        self.name = []
        for j in tqdm(glob(os.path.join(path, "*.json"))):
            name = j.replace(path, "").replace("/", "")
            self.data[name] = Json(j)
            self.name.append(name)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Json:
        return self.data[self.name[idx]]

    def _getKey(self, key: str) -> str:
        return self[0]._getKey(key)

    def tree(self) -> None:
        """json의 구조를 출력하는 메서드

        Examples:
            >>> jsd.tree()
            └─ url
            └─ id
            ...
            └─ user
                └─ login
                └─ id
            ...
        """
        self[0].tree()

    def unique(self, key: str) -> Set[str]:
        """읽어온 json 데이터들의 유일한 값을 return하는 메서드

        Args:
            key (``str``): 읽어온 json 파일에서 불러올 key 값

        Returns:
            ``Set[str]``: Key에 따른 유일한 값들의 집합

        Examples:
            >>> jsd.unique("label")
            {'Zerohertz:docs', 'Zerohertz:dev-v0.2.3', 'Zerohertz:docs-v0.2.2'}
            >>> jsd.unique("sha")
            {'dfd53a0bfc73221dbe96d5e44a49c524d5a8596b', 'bc33235424e89cbbf23434b2a824ea068d167c7d', '97f52f9b81ba885fe69b9726632e580f5cba94be', '768c7711f94af0be00cd55e0ce7b892465cfa64a', '97e103788359f0361f4ec0e138a14218f28eddd4'}
        """
        key = self._getKey(key)
        uq = set()
        for json_instance in self:
            data = json_instance._getValue(key)
            uq.add(str(data))
        return uq


def write_json(data: Union[Dict[Any, Any], List[Dict[Any, Any]]], path: str) -> str:
    """JSON (JavaScript Object Notation)를 작성하는 함수

    Args:
        data (``Dict[Any, Any]``): 입력 데이터 (header 포함 무관)
        path (``str``): 출력될 json 파일의 경로 및 파일명

    Returns:
        ``str``: 파일의 절대 경로

    Examples:
        >>> zz.util.write_json([{"id": "4169", "전투력": 4209, "정보": ["아무", "거나"]}]*100, "zerohertzLib/star_craft")
        '/.../star_craft.json'
        [
            {
                "id": "4169",
                "전투력": 4209,
                "정보": [
                    "아무",
                    "거나"
                ]
            },
            {
                "id": "4169",
                "전투력": 4209,
                "정보": [
                    "아무",
                    "거나"
                ]
            },
        ...
    """
    with open(f"{path}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return os.path.abspath(f"{path}.json")
