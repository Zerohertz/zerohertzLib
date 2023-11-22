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

import random
from typing import List, Optional, Tuple, Union

import seaborn as sns
from matplotlib import pyplot as plt


def _save(title: str, dpi: int) -> None:
    title = title.lower().replace(" ", "_").replace("/", "-")
    plt.savefig(
        f"{title}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close("all")


def color(
    cnt: Optional[int] = 1, rand: Optional[bool] = False, uint8: Optional[bool] = False
) -> Union[Tuple[float], List[int], List[Tuple[float]], List[List[int]]]:
    """색 추출 함수

    Args:
        cnt (``Optional[int]``): 추출할 색의 수
        rand (``Optional[bool]``): Random 추출 여부
        uint8 (``Optional[bool]``): 출력 색상의 type

    Returns:
        ``Union[Tuple[float], List[Tuple[List]]]``: 단일 색 또는 list로 구성된 여러 색

    Examples:
        >>> zz.plot.color()
        (0.9710194877714075, 0.4645444048369612, 0.21958695134807432)
        >>> zz.plot.color(4)
        [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701), (0.5920891529639701, 0.6418467016378244, 0.1935069134991043), (0.21044753832183283, 0.6773105080456748, 0.6433941168468681), (0.6423044349219739, 0.5497680051256467, 0.9582651433656727)]
        >>> zz.plot.color(4, True)
        [(0.22420518847992715, 0.6551391052055489, 0.8272616286387289), (0.9677975592919913, 0.44127456009157356, 0.5358103155058701), (0.21125140522513897, 0.6760830215342485, 0.6556099802889619), (0.9590000285927794, 0.36894286394742526, 0.9138608732554839)]
        >>> zz.plot.color(uint8=True)
        [53, 172, 167]
        >>> zz.plot.color(4, uint8=True)
        [[246, 112, 136], [150, 163, 49], [53, 172, 164], [163, 140, 244]]
        >>> zz.plot.color(4, True, True)
        [[247, 117, 79], [73, 160, 244], [54, 171, 176], [110, 172, 49]]
    """
    if cnt == 1:
        if uint8:
            return [
                int(i * 255)
                for i in random.choice(sns.color_palette("husl", n_colors=100))
            ]
        return random.choice(sns.color_palette("husl", n_colors=100))
    if rand:
        if uint8:
            return [
                [int(j * 255) for j in i]
                for i in random.choices(sns.color_palette("husl", n_colors=100), k=cnt)
            ]
        return random.choices(sns.color_palette("husl", n_colors=100), k=cnt)
    if uint8:
        return [
            [int(j * 255) for j in i] for i in sns.color_palette("husl", n_colors=cnt)
        ]
    return sns.color_palette("husl", n_colors=cnt)
