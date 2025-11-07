import numpy as np
from numpy.typing import DTypeLike, NDArray

import zerohertzLib as zz


class TritonPythonModel(zz.mlops.BaseTritonPythonModel):
    def _inference(
        self,
        images: NDArray[DTypeLike],
        boxes: NDArray[DTypeLike],
        texts: NDArray[DTypeLike],
        checks: NDArray[DTypeLike],
    ) -> tuple[list[NDArray[DTypeLike]]]:
        return (
            np.random.randint(0, 1000, size=(boxes.shape[0], 4)),
            np.random.uniform(0.0, 1.0, size=(boxes.shape[0])),
            np.array(["test"] * boxes.shape[0], dtype=object),
            np.random.randint(0, 2, size=(boxes.shape[0]), dtype=bool),
        )
