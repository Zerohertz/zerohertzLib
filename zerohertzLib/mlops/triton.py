# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
import tritonclient.grpc as grpcclient
from loguru import logger
from numpy.typing import DTypeLike, NDArray
from prettytable import PrettyTable
from tritonclient.utils import triton_to_np_dtype

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass


class TritonClientURL(grpcclient.InferenceServerClient):
    """외부에서 실행되는 triton inference server의 호출을 위한 class

    Args:
        url: 호출할 triton inference server의 URL
        port: triton inference server의 gRPC 통신 port 번호
        verbose: Verbose 출력 여부

    Examples:
        >>> tc = zz.mlops.TritonClientURL("localhost")
        >>> tc("YOLO", np.zeros((1, 3, 640, 640)))
        {'output0': array([[[3.90108061e+00, 3.51982164e+00, 7.49971962e+00, ...,
        2.21481919e-03, 1.17585063e-03, 1.36753917e-03]]], dtype=float32)}
    """

    def __init__(self, url: str, port: int = 8001, verbose: bool = False) -> None:
        self.url = f"{url}:{port}"
        super().__init__(url=self.url, verbose=verbose)
        self.configs = {}
        self.models = []
        for model in self.get_model_repository_index(as_json=True)["models"]:
            self.models.append(model["name"])
        self.emoji = {
            "LOADING": "🚀",
            "READY": "✅",
            "UNLOADING": "🛌",
            "UNAVAILABLE": "💤",
        }

    def __call__(
        self,
        model: int | str,
        *args: NDArray[DTypeLike],
        renew: bool = False,
    ) -> dict[str, NDArray[DTypeLike]]:
        """
        Model 호출 수행

        Args:
            model: 호출할 model의 이름 및 ID
            *args: Model 호출 시 사용될 입력
            renew: 각 모델의 상태 조회 시 갱신 여부

        Returns:
            호출된 model의 결과
        """
        if isinstance(model, int):
            model = self.models[model]
        self._update_configs(model, renew)
        inputs = self.configs[model]["config"]["input"]
        outputs = self.configs[model]["config"]["output"]
        assert len(inputs) == len(args)
        triton_inputs = []
        for input_info, arg in zip(inputs, args):
            triton_inputs.append(
                self._set_input(
                    input_info,
                    arg,
                    self.configs[model]["config"].get("max_batch_size", None),
                )
            )
        triton_outputs = []
        for output in outputs:
            triton_outputs.append(grpcclient.InferRequestedOutput(output["name"]))
        response = self.infer(
            model_name=model, inputs=triton_inputs, outputs=triton_outputs
        )
        response.get_response()
        triton_results = {}
        for output in outputs:
            triton_results[output["name"]] = response.as_numpy(output["name"])
        return triton_results

    def _update_configs(self, model: str, renew: bool) -> None:
        if renew or model not in self.configs:
            self.configs[model] = self.get_model_config(model, as_json=True)

    def _set_input(
        self,
        input_info: dict[str, list[int]],
        value: NDArray[DTypeLike],
        max_batch_size: int | None,
    ) -> grpcclient._infer_input.InferInput:
        if "dims" in input_info.keys():
            if max_batch_size is None:
                if len(input_info["dims"]) != len(value.shape):
                    logger.warning(
                        f"""Expected dimension length of input ({len(input_info["dims"])}) does not match the input dimension length ({len(value.shape)}) [input dimension: {value.shape}]""",
                    )
            elif len(input_info["dims"]) + 1 != len(value.shape):
                logger.warning(
                    f"""Expected dimension length of input ({len(input_info["dims"]) + 1}) does not match the input dimension length ({len(value.shape)}) [input dimension: {value.shape}]""",
                )
        value = value.astype(triton_to_np_dtype(input_info["data_type"][5:]))
        return grpcclient.InferInput(
            input_info["name"],
            value.shape,
            input_info["data_type"][5:],
        ).set_data_from_numpy(value)

    def status(
        self,
        renew: bool = False,
        sortby: str = "STATE",
        reverse: bool = False,
    ) -> None:
        """Triton Inferece Server의 상태를 확인하는 function

        Args:
            renew: 각 모델의 상태 조회 시 갱신 여부
            sortby: 정렬 기준
            reverse: 정렬 역순 여부

        Examples:
            >>> tc.status()

            ![Status GIF](../../../assets/mlops/TritonClientURL.status.gif)
        """
        table = PrettyTable(
            ["STATE", "ID", "MODEL", "VERSION", "BACKEND", "INPUT", "OUTPUT"],
            title=f"Triton Inference Server Status [{self.url}]",
        )
        for model in self.get_model_repository_index(as_json=True)["models"]:
            if model["name"] not in self.models:
                self.models.append(model["name"])
            state = model.get("state", "UNAVAILABLE")
            if state in ["LOADING", "UNAVAILABLE"]:
                _input, _output = ["-"], ["-"]
                backend = "-"
            else:
                self._update_configs(model["name"], renew)
                _input, _output = [], []
                for inputs in self.configs[model["name"]]["config"]["input"]:
                    _input.append(
                        f"""{inputs["name"]} [{inputs["data_type"][5:]}: ({", ".join(inputs["dims"])})]"""
                    )
                for outputs in self.configs[model["name"]]["config"]["output"]:
                    _output.append(
                        f"""{outputs["name"]} [{outputs["data_type"][5:]}: ({", ".join(outputs["dims"])})]"""
                    )
                backend = self.configs[model["name"]]["config"].get("backend", "-")
            table.add_row(
                [
                    self.emoji[state],
                    self.models.index(model["name"]),
                    model["name"],
                    model.get("version", "-"),
                    backend,
                    "\n".join(_input),
                    "\n".join(_output),
                ]
            )
        if sortby:
            table.sortby = sortby
        table.reversesort = reverse
        logger.info(f"\n{table}")

    def load_model(
        self,
        model_name: int | str,
        headers: str | None = None,
        config: str | None = None,
        files: str | None = None,
        client_timeout: float | None = None,
    ) -> None:
        """Triton Inference Server 내 model을 load하는 function

        Args:
            model_name: Load할 model의 이름 또는 ID
            headers: Request 전송 시 포함할 추가 HTTP header
            config: Model load 시 사용될 config
            files: Model load 시 override model directory에서 사용할 file
            client_timeout: 초 단위의 timeout

        Examples:
            >>> tc.load_model(0)
            >>> tc.load_model("MODEL_NAME")
        """
        if isinstance(model_name, int):
            model_name = self.models[model_name]
        super().load_model(model_name, headers, config, files, client_timeout)

    def unload_model(
        self,
        model_name: int | str,
        headers: str | None = None,
        unload_dependents: bool = False,
        client_timeout: float | None = None,
    ) -> None:
        """Triton Inference Server 내 model을 unload하는 function

        Args:
            model_name: Unload할 model의 이름 또는 ID
            headers: Request 전송 시 포함할 추가 HTTP header
            unload_dependents: Model unload 시 dependents의 unload 여부
            client_timeout: 초 단위의 timeout

        Examples:
            >>> tc.unload_model(0)
            >>> tc.unload_model("MODEL_NAME")
        """
        if isinstance(model_name, int):
            model_name = self.models[model_name]
        super().unload_model(model_name, headers, unload_dependents, client_timeout)


class TritonClientK8s(TritonClientURL):
    """Kubernetes에서 실행되는 triton inference server의 호출을 위한 class

    Args:
        svc_name: 호출할 triton inference server의 Kubernetes service의 이름
        namespace: 호출할 triton inference server의 namespace
        port: triton inference server의 gRPC 통신 port 번호
        verbose: Verbose 출력 여부

    Examples:
        Kubernetes:
        ```bash
        $ kubectl get svc -n yolo
        NAME                          TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
        fastapi-svc                   ClusterIP   10.106.72.126   <none>        80/TCP     90s
        triton-inference-server-svc   ClusterIP   10.96.28.172    <none>        8001/TCP   90s
        $ docker exec -it ${API_CONTAINER} bash
        ```
        Python:
        ```python
        >>> tc = zz.mlops.TritonClientK8s("triton-inference-server-svc", "yolo")
        >>> tc("YOLO", np.zeros((1, 3, 640, 640)))
        {'output0': array([[[3.90108061e+00, 3.51982164e+00, 7.49971962e+00, ...,
        2.21481919e-03, 1.17585063e-03, 1.36753917e-03]]], dtype=float32)}
        ```
    """

    def __init__(
        self,
        svc_name: str,
        namespace: str,
        port: int = 8001,
        verbose: bool = False,
    ) -> None:
        super().__init__(f"{svc_name}.{namespace}", port, verbose)


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
