# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from cmath import exp, pi


def fft(sig: list[complex], inv: bool = False) -> list[complex]:
    """FFT (Fast Fourier Transform)를 수행하기 위한 함수

    Args:
        sig: 입력 신호 (복소수 list)
        inv: 변환 방향을 지정 (`False`: 정방향, `True`: 역방향)

    Returns:
        변환된 결과 (복소수 list)

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
        sig_even = fft(sig[0::2], False)
        sig_odd = fft(sig[1::2], False)
        weight = [exp(2j * pi * i / length) for i in range(length // 2)]
        return [sig_even[i] + weight[i] * sig_odd[i] for i in range(length // 2)] + [
            sig_even[i] - weight[i] * sig_odd[i] for i in range(length // 2)
        ]
    sig_even = fft(sig[0::2], True)
    sig_odd = fft(sig[1::2], True)
    weight = [exp(-2j * pi * i / length) for i in range(length // 2)]
    return [sig_even[i] + weight[i] * sig_odd[i] for i in range(length // 2)] + [
        sig_even[i] - weight[i] * sig_odd[i] for i in range(length // 2)
    ]
