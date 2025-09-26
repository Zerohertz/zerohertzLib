import sys
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike, NDArray

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
# import zerohertzLib as zz
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
