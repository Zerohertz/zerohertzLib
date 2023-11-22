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

from cmath import exp, pi
from typing import List, Optional


def fft(sig: List[complex], inv: Optional[bool] = False) -> List[complex]:
    """FFT (Fast Fourier Transform)를 수행하기 위한 함수

    Args:
        sig (``List[complex]``): 입력 신호 (복소수 list)
        inv (``Optional[bool]``): 변환 방향을 지정 (``False``: 정방향, ``True``: 역방향)

    Returns:
        ``List[complex]``: 변환된 결과 (복소수 list)

    Examples:
        >>> zz.algorithm.fft([1, 0, 0, 0])
        [(1+0j), (1+0j), (1+0j), (1+0j)]
        >>> zz.algorithm.fft([1+0j, 1+0j, 1+0j, 1+0j], True)
        [(4+0j), 0j, 0j, 0j]
    """
    length = len(sig)
    if length == 1:
        return sig
    if not inv:
        sig_even = fft(sig[0::2], 0)
        sig_odd = fft(sig[1::2], 0)
        weight = [exp(2j * pi * i / length) for i in range(length // 2)]
        return [sig_even[i] + weight[i] * sig_odd[i] for i in range(length // 2)] + [
            sig_even[i] - weight[i] * sig_odd[i] for i in range(length // 2)
        ]
    sig_even = fft(sig[0::2], 1)
    sig_odd = fft(sig[1::2], 1)
    weight = [exp(-2j * pi * i / length) for i in range(length // 2)]
    return [sig_even[i] + weight[i] * sig_odd[i] for i in range(length // 2)] + [
        sig_even[i] - weight[i] * sig_odd[i] for i in range(length // 2)
    ]
