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
        batch_size = images.shape[0]
        _boxes, _scores, _texts, _checks, batch_index = [], [], [], [], []
        for batch in range(batch_size):
            objects = np.random.randint(1, 10)
            _boxes.append(np.random.randint(0, 1000, size=(objects, 4)))
            _scores.append(np.random.uniform(0.0, 1.0, size=(objects)))
            _texts.append(np.array(["test"] * objects, dtype=object))
            _checks.append(np.random.randint(0, 2, size=(objects), dtype=bool))
            batch_index.append(np.array([batch] * objects))
        return _boxes, _scores, _texts, _checks, batch_index
