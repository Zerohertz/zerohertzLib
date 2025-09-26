# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import DTypeLike, NDArray

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass


class BaseTritonPythonModel(ABC):
    """Triton Inference Server에서 Python backend 사용을 위한 class

    Note:
        Abstract Base Class: Model의 추론을 수행하는 abstract method `_inference` 정의 후 사용

    Examples:
        `model.py`:
            ```python
            class TritonPythonModel(zz.mlops.BaseTritonPythonModel):
                def initialize(self, args: dict[str, str]) -> None:
                    super().initialize(args)
                    self.model = Model(cfg)

                def _inference(input) -> tuple[Any]:
                    return self.model(input)
            ```

        Normal Logs (Without Batching):
            ```
            2025-09-25 16:06:51.904 | INFO     | zerohertzLib.mlops.triton:initialize:* - Initialize: {
                "name": "...",
                "platform": "",
                "backend": "python",
                "runtime": "",
                "version_policy": {
                    "latest": {
                        "num_versions": 1
                    }
                },
                "max_batch_size": 0,
            ...
            2025-09-25 16:22:48.226 | INFO     | zerohertzLib.mlops.triton:execute:* - Called
            2025-09-25 16:22:48.234 | DEBUG    | zerohertzLib.mlops.triton:_get_inputs:* - inputs: images=(2078, 1470, 3)
            2025-09-25 16:22:48.234 | INFO     | zerohertzLib.mlops.triton:execute:* - Inference start
            2025-09-25 16:22:49.026 | INFO     | zerohertzLib.mlops.triton:execute:* - Inference completed (0.79s)
            2025-09-25 16:22:49.026 | DEBUG    | zerohertzLib.mlops.triton:_set_outputs:* - outputs: boxes=(12, 4), scores=(12,), labels=(12,)
            ```

        Normal Logs (With Batching):
            ```
            2025-09-25 16:24:34.224 | INFO     | zerohertzLib.mlops.triton:execute:* - Called
            2025-09-25 16:24:34.232 | DEBUG    | zerohertzLib.mlops.triton:_get_inputs:* - inputs: images=(9, 1000, 1000, 3)
            2025-09-25 16:24:34.232 | INFO     | zerohertzLib.mlops.triton:execute:* - Inference start
            2025-09-25 16:24:34.486 | INFO     | zerohertzLib.mlops.triton:execute:* - Inference completed (0.25s)
            2025-09-25 16:24:34.486 | DEBUG    | zerohertzLib.mlops.triton:_set_outputs:* - outputs (0 ~ 3): boxes=(6, 4), (6, 4), (6, 4), scores=(6,), (6,), (6,), labels=(6,), (6,), (6,), batch_index=(6,), (6,), (6,)
            2025-09-25 16:24:34.486 | DEBUG    | zerohertzLib.mlops.triton:_set_outputs:* - outputs (3 ~ 6): boxes=(6, 4), (6, 4), (6, 4), scores=(6,), (6,), (6,), labels=(6,), (6,), (6,), batch_index=(6,), (6,), (6,)
            2025-09-25 16:24:34.486 | DEBUG    | zerohertzLib.mlops.triton:_set_outputs:* - outputs (6 ~ 9): boxes=(6, 4), (6, 4), (6, 4), scores=(6,), (6,), (6,), labels=(6,), (6,), (6,), batch_index=(6,), (6,), (6,)
            ```

        Error Logs:
            ```
            2025-09-25 16:26:32.004 | ERROR    | zerohertzLib.mlops.triton:execute:* - zerohertzLib!
            Traceback (most recent call last):
            > File "/usr/local/lib/python3.10/dist-packages/zerohertzLib/mlops/triton.py", line 371, in execute
                outputs = self._inference(**inputs)
                        |    |            -> {'images': array([[[ 38,  38,  38],
                        |    |                       [ 37,  37,  37],
                        |    |                       [ 37,  37,  37],
                        |    |                       ...,
                        |    |                       [255, 255, 255],
                        |    |                ...
                        |    -> <function TritonPythonModel._inference at 0x7f106f48f400>
                        -> <1.model.TritonPythonModel object at 0x7f121fa1f010>

            File "/models/docling_layout_old_static/1/model.py", line 34, in _inference
                raise Exception("zerohertzLib!")
            Exception: zerohertzLib!
            ```
    """

    def initialize(self, args: dict[str, str]) -> None:
        """Triton Inference Server 시작 시 수행되는 method

        Args:
            args: `config.pbtxt` 에 포함된 model의 정보
        """
        self.cfg = json.loads(args["model_config"])
        logger.info(f"Initialize: {json.dumps(self.cfg, indent=4)}")
        self.device = "cpu"
        device = args.get("model_instance_device_id", None)
        if device is not None:
            self.device = f"cuda:{device}"
        self.max_batch_size = self.cfg.get("max_batch_size", 0)

    def execute(self, requests: list[Any]) -> list[Any]:
        """Triton Inference Server 호출 시 수행되는 method

        Args:
            requests: Client에서 전송된 model inputs

        Returns:
            Client에 응답할 model의 추론 결과
        """
        logger.info("Called")
        try:
            inputs, batch_index = self._get_inputs(requests=requests)
            logger.info("Inference start")
            start = time.time()
            outputs = self._inference(**inputs)
            end = time.time()
            logger.info(f"Inference completed ({end - start:.2f}s)")
            if not isinstance(outputs, tuple):
                outputs = tuple([outputs])
            responses = self._set_outputs(outputs=outputs, batch_index=batch_index)
        except Exception as exc:
            logger.exception(exc)
            responses = [
                pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(exc)
                )
                for _ in requests
            ]
        return responses

    def _get_inputs(
        self, requests: list[Any]
    ) -> tuple[dict[str, NDArray[DTypeLike]], list[int]]:
        batch_index = [0]
        _inputs = defaultdict(list)
        for request in requests:
            for index, cfg_input in enumerate(self.cfg["input"]):
                value = pb_utils.get_input_tensor_by_name(
                    request, cfg_input["name"]
                ).as_numpy()
                if index == 0 and 0 < self.max_batch_size:
                    batch_index.append(batch_index[-1] + value.shape[0])
                _inputs[cfg_input["name"]].append(value)
        inputs = {}
        for key, value in _inputs.items():
            inputs[key] = np.concatenate(value, axis=0)
        logger.debug(
            "inputs: "
            + ", ".join([f"{key}={value.shape}" for key, value in inputs.items()])
        )
        return inputs, batch_index

    def _set_outputs(self, outputs: tuple[Any], batch_index: list[int]) -> list[Any]:
        responses = []
        if 0 < self.max_batch_size:
            for index in range(len(batch_index) - 1):
                batch_tensors = defaultdict(list)
                for batch in range(batch_index[index], batch_index[index + 1]):
                    for cfg_output, value in zip(self.cfg["output"], outputs):
                        _value = value[batch]
                        if cfg_output["name"] == "batch_index":
                            _value -= batch_index[index]
                        batch_tensors[cfg_output["name"]].append(_value)
                output_tensors = []
                for cfg_output in self.cfg["output"]:
                    value = np.concatenate(batch_tensors[cfg_output["name"]], axis=0)
                    output_tensors.append(
                        pb_utils.Tensor(
                            cfg_output["name"],
                            value.astype(
                                pb_utils.triton_string_to_numpy(cfg_output["data_type"])
                            ),
                        )
                    )
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=output_tensors)
                )
                logger.debug(
                    f"outputs ({batch_index[index]} ~ {batch_index[index + 1]}): "
                    + ", ".join(
                        [
                            f"{key}="
                            + ", ".join([f"{_value.shape}" for _value in value])
                            for key, value in batch_tensors.items()
                        ]
                    )
                )
            return responses
        output_tensors = []
        for cfg_output, value in zip(self.cfg["output"], outputs):
            output_tensors.append(
                pb_utils.Tensor(
                    cfg_output["name"],
                    value.astype(
                        pb_utils.triton_string_to_numpy(cfg_output["data_type"])
                    ),
                )
            )
        responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        logger.debug(
            "outputs: "
            + ", ".join(
                [
                    f"""{key["name"]}={value.shape}"""
                    for key, value in zip(self.cfg["output"], outputs)
                ]
            )
        )
        return responses

    @abstractmethod
    def _inference(self, **inputs: NDArray[DTypeLike]) -> Any | tuple[Any]:
        """
        Model 추론을 수행하는 private method (상속을 통한 재정의 필수)

        Args:
            inputs: Model 추론 시 사용될 입력 (`config.pbtxt` 의 입력에 따라 입력 결정)

        Returns:
            Model의 추론 결과
        """
        pass

    def finalize(self) -> None:
        """Triton Inference Server 종료 시 수행되는 method"""
        logger.info("Finalize")
