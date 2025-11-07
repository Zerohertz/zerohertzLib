import numpy as np
from numpy.typing import DTypeLike, NDArray

import zerohertzLib as zz


class TritonPythonModel(zz.mlops.BaseTritonPythonModel):
    def _inference(
        self, images: NDArray[DTypeLike], boxes: NDArray[DTypeLike]
    ) -> tuple[list[NDArray[DTypeLike]]]:
        batch_size = images.shape[0]
        _boxes, scores, texts, batch_index = [], [], [], []
        for batch in range(batch_size):
            objects = np.random.randint(1, 10)
            _boxes.append(np.random.randint(0, 1000, size=(objects, 4)))
            scores.append(np.random.uniform(0.0, 1.0, size=(objects)))
            texts.append(np.array(["test" * len(objects)], dtype=np.object_))
            batch_index.append(np.array([batch] * objects))
        return _boxes, scores, texts, batch_index
