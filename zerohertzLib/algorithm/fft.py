from cmath import exp, pi
from typing import List


def fft(sig: List[complex], inv: int) -> List[complex]:
    """FFT (Fast Fourier Transform)를 수행하기 위한 함수

    Args:
        sig (``List[complex]``): 입력 신호 (복소수 리스트)
        inv (``int``): 변환 방향을 지정 (0: 정방향, 1: 역방향)

    Returns:
        ``List[complex]``: 변환된 결과 (복소수 리스트)

    Examples:
        >>> zz.algorithm.fft([1, 0, 0, 0], 0)
        [(1+0j), (1+0j), (1+0j), (1+0j)]
        >>> zz.algorithm.fft([1+0j, 1+0j, 1+0j, 1+0j], 1)
        [(4+0j), 0j, 0j, 0j]
    """
    N = len(sig)
    if N == 1:
        return sig
    if inv == 0:
        sig_even = fft(sig[0::2], 0)
        sig_odd = fft(sig[1::2], 0)
        W = [exp(2j * pi * i / N) for i in range(N // 2)]
        return [sig_even[i] + W[i] * sig_odd[i] for i in range(N // 2)] + [
            sig_even[i] - W[i] * sig_odd[i] for i in range(N // 2)
        ]
    elif inv == 1:
        sig_even = fft(sig[0::2], 1)
        sig_odd = fft(sig[1::2], 1)
        W = [exp(-2j * pi * i / N) for i in range(N // 2)]
        return [sig_even[i] + W[i] * sig_odd[i] for i in range(N // 2)] + [
            sig_even[i] - W[i] * sig_odd[i] for i in range(N // 2)
        ]
