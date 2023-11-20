import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union


def read_csv(
    path: str, header: Optional[bool] = True
) -> Dict[Union[int, str], List[str]]:
    """CSV (Comma-Separated Values) 혹은 TSV (Tab-Separated Values)를 작성하는 함수

    Args:
        path (``str``): 입력될 CSV 혹은 TSV 경로 및 파일명
        header (``Optional[bool]``): Header의 존재 유무

    Returns:
        ``Dict[str, List[str]]``: Header의 값을 기반으로 column에 따라 `List` 로 구성

    Note:
        Header가 존재하지 않는 경우 `0` 부터 차례대로 key 값 정의

    Examples:
        >>> zz.util.read_csv("star_craft.csv")
        defaultdict(<class 'list'>, {'id': ['5hi9', 'gor2', 'gk03'], '종족': ['프로토스', '테란', '저그'], '점수': ['1248', '2309', '291']})
        >>> zz.util.read_csv("star_craft.tsv")
        defaultdict(<class 'list'>, {'id': ['5hi9', 'gor2', 'gk03'], '종족': ['프로토스', '테란', '저그'], '점수': ['1248', '2309', '291']})
        >>> zz.util.read_csv("star_craft.csv", header=False)
        defaultdict(<class 'list'>, {0: ['id', '5hi9', 'gor2', 'gk03'], 1: ['종족', '프로토스', '테란', '저그'], 2: ['점수', '1248', '2309', '291']})
        >>> zz.util.read_csv("star_craft.tsv", header=False)
        defaultdict(<class 'list'>, {0: ['id', '5hi9', 'gor2', 'gk03'], 1: ['종족', '프로토스', '테란', '저그'], 2: ['점수', '1248', '2309', '291']})
    """
    data = defaultdict(list)
    keys = []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.readlines()
    if path.endswith(".csv"):
        delimiter = ","
    elif path.endswith(".tsv"):
        delimiter = "\t"
    else:
        raise Exception("File is not CSV or TSV")
    if header:
        r = raw[0]
        for key in r.strip().split(delimiter):
            keys.append(key)
        raw = raw[1:]
    else:
        for key in range(len(raw[0].strip().split(delimiter))):
            keys.append(key)
    for r in raw:
        for k, d in zip(keys, r.strip().split(delimiter)):
            data[k].append(d)
    return data


def write_csv(data: List[List[Any]], path: str, tsv: Optional[bool] = False) -> str:
    """CSV (Comma-Separated Values) 혹은 TSV (Tab-Separated Values)를 작성하는 함수

    Args:
        data (``List[List[Any]]``): 입력 데이터 (header 포함 무관)
        path (``str``): 출력될 CSV 혹은 TSV 경로 및 파일명
        tsv (``Optional[bool]``): TSV 작성 여부

    Returns:
        ``str``: 파일의 절대 경로

    Examples:
        >>> zz.util.write_csv([["id", "종족", "점수"], ["5hi9", "프로토스", 1248], ["gor2", "테란", 2309], ["gk03", "저그", 291]], "zerohertzLib/star_craft")
        '/.../star_craft.csv'
        >>> zz.util.write_csv([["id", "종족", "점수"], ["5hi9", "프로토스", 1248], ["gor2", "테란", 2309], ["gk03", "저그", 291]], "zerohertzLib/star_craft", True)
        '/.../star_craft.tsv'
    """
    if tsv:
        with open(f"{path}.tsv", "w", encoding="utf-8") as f:
            for d in data:
                f.writelines("\t".join(list(map(str, d))) + "\n")
        return os.path.abspath(f"{path}.tsv")
    else:
        with open(f"{path}.csv", "w", encoding="utf-8") as f:
            for d in data:
                f.writelines(",".join(list(map(str, d))) + "\n")
        return os.path.abspath(f"{path}.csv")
