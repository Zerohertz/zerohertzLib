import numpy as np
from numpy.typing import DTypeLike, NDArray

from zerohertzLib.mlops import BaseTritonPythonModel


class TritonPythonModel(BaseTritonPythonModel):
    def _inference(
        self, images: NDArray[DTypeLike], boxes: NDArray[DTypeLike]
    ) -> tuple[NDArray[DTypeLike]]:
        return (
            np.random.randint(0, 1000, size=(boxes.shape[0], 4)),
            np.random.uniform(0.0, 1.0, size=(boxes.shape[0])),
            np.random.randint(0, 10, size=(boxes.shape[0])),
        )
