import json
import os
from typing import Any, Dict, List, Optional, Union


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
