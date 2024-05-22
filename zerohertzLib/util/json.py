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
from glob import glob
from typing import Any, Dict, List, Optional, Set, Union

import orjson
from tqdm import tqdm


class Json:
    """JSON 형식 file을 읽고 사용하기 위한 class

    객체 생성 시 ``path`` 를 입력하지 않을 시 현재 경로에 존재하는 JSON file을 읽고 ``path`` 를 경로로 입력하면 해당 경로에 존재하는 JSON file을 읽는다.

    Args:
        path (``Optional[str]``): JSON file의 경로

    Attributes:
        name (``str``): JSON file 이름
        keys (``List[str]``): 직렬화된 JSON의 key 값들

    Methods:
        __len__:
            Returns:
                ``int``: 읽어온 JSON file의 길이

        __getitem__:
            읽어온 JSON file에 key 값 입력

            Args:
                key (``Union[int, str]``): 읽어온 JSON file에서 불러올 key 값

            Returns:
                ``Any``: Key에 따른 value 값

        _get_key:
            Key의 경로를 찾아주는 method

            Args:
                key (``str``): 읽어온 JSON file에서 불러올 key 값 (깊이 무관)

            Returns:
                ``str``: ``/`` 으로 깊이를 표시한 key 값

        _get_value:
            ``Json._get_key`` 로 생성된 key 값을 입력 받아 value return

            Args:
                key (``str``): ``Json._get_key`` 로 생성된 key 값

            Returns:
                ``Any``: Key에 따른 value

    Examples:
        >>> js = zz.util.Json()
        >>> js["title"]
        '[v0.2.3] Release'
        >>> key = js._get_key("color")
        >>> key
        'labels/LIST/color'
        >>> js._get_value(key)
        'd73a4a'
        >>> js.name
        '65.json'
        >>> js.keys
        ['url', 'id', ..., 'assignees/LIST/login', ..., 'active_lock_reason']
    """

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = glob("*.json")[0]
        elif not path.endswith(".json"):
            path = glob(f"{path}/*.json")[0]
        self.name = path.replace(os.path.dirname(path), "").replace("/", "")
        with open(path, "r", encoding="utf-8") as file:
            self.data = orjson.loads(file.read())
        self.keys = []
        self.map = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: Union[int, str]) -> Any:
        return self.data[key]

    def __get_keys(
        self, data: Any, key: Optional[str] = "", front: Optional[str] = ""
    ) -> None:
        if isinstance(data, dict):
            for idx, (key_, val_) in enumerate(data.items()):
                if idx + 1 == len(data):
                    self.map.append(front + "└── " + str(key_))
                    front_ = " "
                else:
                    self.map.append(front + "├── " + str(key_))
                    front_ = "│"
                if key:
                    self.keys.append(f"{key}/{key_}")
                    self.__get_keys(val_, f"{key}/{key_}", front + f"{front_}   ")
                else:
                    self.keys.append(f"{key_}")
                    self.__get_keys(val_, f"{key_}", front + f"{front_}   ")
        elif isinstance(data, list):
            self.map.append(front + "└── " + "LIST")
            if data:
                if key:
                    self.__get_keys(data[0], f"{key}/LIST", front + "    ")
                else:
                    self.__get_keys(data[0], "LIST", front + "    ")

    def _get_keys(self) -> List[str]:
        if not self.keys and not self.map:
            self.__get_keys(self.data)
        return self.keys

    def _get_key(self, key: str) -> str:
        keys = self._get_keys()
        if key in keys:
            return key
        if "/" not in key:
            for key_ in keys:
                if key_.endswith(key):
                    key = key_
                    break
        return key

    def _get_value(self, key: str) -> Any:
        value = self.data
        for key_ in key.split("/"):
            if key_ == "LIST":
                value = value[0]
            else:
                value = value.get(key_)
        return value

    def get(self, key: str) -> Any:
        """``Json._get_key`` 로 생성된 key 값을 입력 받아 value return

        Args:
            key (``str``): 읽어온 JSON file에서 불러올 key 값 (깊이 무관)

        Returns:
            ``Any``: Key에 따른 value

        Examples:
            >>> js["title"]
            '[v0.2.3] Release'
            >>> js.get("title")
            '[v0.2.3] Release'
            >>> js["color"]
            Traceback (most recent call last):
              File "<stdin>", line 1, in <module>
              File "/home/zerohertz/Zerohertz/zerohertzLib/zerohertzLib/util/json.py", line 107, in __getitem__
            KeyError: 'color'
            >>> js.get("color")
            'd73a4a'
        """
        return self._get_value(self._get_key(key))

    def tree(self) -> None:
        """JSON의 구조를 출력하는 method

        Examples:
            >>> js.tree()
            ├── url
            ...
            ├── user
            │   ├── login
            ...
            │   └── site_admin
            ├── body
            ...
            ├── assignee
            │   ├── login
            ...
            │   └── site_admin
            ├── assignees
            │   └── LIST
            │       ├── login
            ...
            │       └── site_admin
            ...
            └── active_lock_reason
        """
        self._get_keys()
        print("\n".join(self.map))


class JsonDir:
    """입력된 경로에 존재하는 JSON 형식 file들을 읽고 사용하기 위한 class

    객체 생성 시 ``path`` 를 입력하지 않을 시 현재 경로에 존재하는 JSON file들을 읽는다.

    Args:
        path (``Optional[str]``): JSON file의 경로

    Attributes:
        name (``str``): 읽어온 JSON file의 이름들
        data (``Dict[str, zerohertzLib.util.Json]``): File 이름에 따른 `Json` 객체 배열

    Methods:
        __len__:
            Returns:
                ``int``: 읽어온 JSON file들의 수

        __getitem__:
            읽어온 JSON file들을 list와 같이 indexing

            Args:
                idx (``int``): 입력 index

            Returns:
                ``zerohertzLib.util.Json``: Index에 따른 ``Json`` instance

        _get_key:
            Key의 경로를 찾아주는 method

            Args:
                key (``str``): 읽어온 JSON file에서 불러올 key 값 (깊이 무관)

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
        >>> jsd._get_key("color")
        'labels/LIST/color'
    """

    def __init__(self, path: Optional[str] = "") -> None:
        if path.endswith(".json"):
            raise ValueError("'path' ends with '*.json'")
        self.data = {}
        self.name = []
        for json_path in tqdm(glob(os.path.join(path, "*.json"))):
            name = json_path.replace(path, "").replace("/", "")
            self.data[name] = Json(json_path)
            self.name.append(name)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Json:
        return self.data[self.name[idx]]

    def _get_key(self, key: str) -> str:
        return self[0]._get_key(key)

    def tree(self) -> None:
        """JSON의 구조를 출력하는 method

        Examples:
            >>> jsd.tree()
            ├── url
            ...
            ├── user
            │   ├── login
            ...
            │   └── site_admin
            ├── body
            ...
            ├── assignee
            │   ├── login
            ...
            │   └── site_admin
            ├── assignees
            │   └── LIST
            │       ├── login
            ...
            │       └── site_admin
            ...
            └── active_lock_reason
        """
        self[0].tree()

    def unique(self, key: str) -> Set[str]:
        """읽어온 JSON data들의 유일한 값을 return하는 method

        Args:
            key (``str``): 읽어온 JSON file에서 불러올 key 값

        Returns:
            ``Set[str]``: Key에 따른 유일한 값들의 집합

        Examples:
            >>> jsd.unique("label")
            {'Zerohertz:docs', 'Zerohertz:dev-v0.2.3', 'Zerohertz:docs-v0.2.2'}
            >>> jsd.unique("sha")
            {'dfd53a0bfc73221dbe96d5e44a49c524d5a8596b', 'bc33235424e89cbbf23434b2a824ea068d167c7d', '97f52f9b81ba885fe69b9726632e580f5cba94be', '768c7711f94af0be00cd55e0ce7b892465cfa64a', '97e103788359f0361f4ec0e138a14218f28eddd4'}
        """
        key = self._get_key(key)
        uniq = set()
        for json_instance in self:
            data = json_instance._get_value(key)
            uniq.add(str(data))
        return uniq


def write_json(data: Union[Dict[Any, Any], List[Dict[Any, Any]]], path: str) -> str:
    """JSON (JavaScript Object Notation)을 작성하는 함수

    Args:
        data (``Union[Dict[Any, Any], List[Dict[Any, Any]]]``): 입력 data (header 포함 무관)
        path (``str``): 출력될 JSON file의 경로 및 이름

    Returns:
        ``str``: File의 절대 경로

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
    with open(f"{path}.json", "wb") as file:
        file.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    return os.path.abspath(f"{path}.json")
