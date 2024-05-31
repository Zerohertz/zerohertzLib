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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import tritonclient.grpc as grpcclient
from numpy.typing import DTypeLike, NDArray
from prettytable import PrettyTable
from tritonclient.utils import triton_to_np_dtype

from zerohertzLib.logging import Logger

try:
    import json
    import traceback

    import triton_python_backend_utils as pb_utils
except ImportError:
    pass


class TritonClientURL(grpcclient.InferenceServerClient):
    """외부에서 실행되는 triton inference server의 호출을 위한 class

    Args:
        url (``str``): 호출할 triton inference server의 URL
        port (``Optional[int]``): triton inference server의 gRPC 통신 port 번호
        verbose (``Optional[bool]``): Verbose 출력 여부

    Methods:
        __call__:
            Model 호출 수행

            Args:
                model (``Union[int, str]``): 호출할 model의 이름 및 ID
                *args (``NDArray[DTypeLike]``): Model 호출 시 사용될 입력
                renew: (``Optional[bool]``): 각 모델의 상태 조회 시 갱신 여부

            Returns:
                ``Dict[str, NDArray[DTypeLike]]``: 호출된 model의 결과

    Examples:
        >>> tc = zz.mlops.TritonClientURL("localhost")
        >>> tc("YOLO", np.zeros((1, 3, 640, 640)))
        {'output0': array([[[3.90108061e+00, 3.51982164e+00, 7.49971962e+00, ...,
        2.21481919e-03, 1.17585063e-03, 1.36753917e-03]]], dtype=float32)}
    """

    def __init__(
        self, url: str, port: Optional[int] = 8001, verbose: Optional[bool] = False
    ) -> None:
        self.url = f"{url}:{port}"
        super().__init__(url=self.url, verbose=verbose)
        self.configs = {}
        self.models = []
        for model in self.get_model_repository_index(as_json=True)["models"]:
            self.models.append(model["name"])
        self.logger = Logger("TritonClient", logger_level=20)
        self.emoji = {
            "LOADING": "🚀",
            "READY": "✅",
            "UNLOADING": "🛌",
            "UNAVAILABLE": "💤",
        }

    def __call__(
        self,
        model: Union[int, str],
        *args: NDArray[DTypeLike],
        renew: Optional[bool] = False,
    ) -> Dict[str, NDArray[DTypeLike]]:
        if isinstance(model, int):
            model = self.models[model]
        self._update_configs(model, renew)
        inputs = self.configs[model]["config"]["input"]
        outputs = self.configs[model]["config"]["output"]
        assert len(inputs) == len(args)
        triton_inputs = []
        for input_info, arg in zip(inputs, args):
            triton_inputs.append(self._set_input(input_info, arg))
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
        self, input_info: Dict[str, List[int]], value: NDArray[DTypeLike]
    ) -> grpcclient._infer_input.InferInput:
        if "dims" in input_info.keys() and len(input_info["dims"]) != len(value.shape):
            self.logger.warning(
                "Expected dimension length of input (%d) does not match the input dimension length (%d) [input dimension: %s]",
                len(input_info["dims"]),
                len(value.shape),
                value.shape,
            )
        value = value.astype(triton_to_np_dtype(input_info["data_type"][5:]))
        return grpcclient.InferInput(
            input_info["name"],
            value.shape,
            input_info["data_type"][5:],
        ).set_data_from_numpy(value)

    def status(
        self,
        renew: Optional[bool] = False,
        sortby: Optional[str] = "STATE",
        reverse: Optional[bool] = False,
    ) -> None:
        """Triton Inferece Server의 상태를 확인하는 함수

        Args:
            renew: (``Optional[bool]``): 각 모델의 상태 조회 시 갱신 여부
            sortby (``Optional[str]``): 정렬 기준
            reverse (``Optional[bool]``): 정렬 역순 여부

        Examples:
            >>> tc.status()

            .. image:: _static/examples/static/mlops.TritonClientURL.status.gif
                :align: center
                :width: 700px
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
        self.logger.info("\n%s", str(table))

    def load_model(
        self,
        model_name: Union[int, str],
        headers: Optional[Dict] = None,
        config: Optional[str] = None,
        files: Optional[Dict] = None,
        client_timeout: Optional[float] = None,
    ) -> None:
        """Triton Inference Server 내 model을 load하는 함수

        Args:
            model_name (``Union[int, str]``): Load할 model의 이름 또는 ID
            headers (``Optional[Dict]``): Request 전송 시 포함할 추가 HTTP header
            config (``Optional[str]``): Model load 시 사용될 config
            config (``Optional[Dict]``): Model load 시 override model directory에서 사용할 file
            client_timeout (``Optional[float]``): 초 단위의 timeout

        Examples:
            >>> tc.load_model(0)
            >>> tc.load_model("MODEL_NAME")
        """
        if isinstance(model_name, int):
            model_name = self.models[model_name]
        super().load_model(model_name, headers, config, files, client_timeout)

    def unload_model(
        self,
        model_name: Union[int, str],
        headers: Optional[Dict] = None,
        unload_dependents: Optional[bool] = False,
        client_timeout: Optional[float] = None,
    ) -> None:
        """Triton Inference Server 내 model을 unload하는 함수

        Args:
            model_name (``Union[int, str]``): Unload할 model의 이름 또는 ID
            headers (``Optional[Dict]``): Request 전송 시 포함할 추가 HTTP header
            unload_dependents (``Optional[bool]``): Model unload 시 dependents의 unload 여부
            client_timeout (``Optional[float]``): 초 단위의 timeout

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
        svc_name (``str``): 호출할 triton inference server의 Kubernetes service의 이름
        namespace (``str``): 호출할 triton inference server의 namespace
        port (``Optional[int]``): triton inference server의 gRPC 통신 port 번호
        verbose (``Optional[bool]``): Verbose 출력 여부

    Methods:
        __call__:
            Model 호출 수행

            Args:
                model (``Union[int, str]``): 호출할 model의 이름 또는 ID
                *args (``NDArray[DTypeLike]``): Model 호출 시 사용될 입력
                renew: (``Optional[bool]``): 각 모델의 상태 조회 시 갱신 여부

            Returns:
                ``Dict[str, NDArray[DTypeLike]]``: 호출된 model의 결과

    Examples:
        Kubernetes:
            >>> kubectl get svc -n yolo
            NAME                          TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
            fastapi-svc                   ClusterIP   10.106.72.126   <none>        80/TCP     90s
            triton-inference-server-svc   ClusterIP   10.96.28.172    <none>        8001/TCP   90s
            >>> docker exec -it ${API_CONTAINER} bash

        Python:
            >>> tc = zz.mlops.TritonClientK8s("triton-inference-server-svc", "yolo")
            >>> tc("YOLO", np.zeros((1, 3, 640, 640)))
            {'output0': array([[[3.90108061e+00, 3.51982164e+00, 7.49971962e+00, ...,
            2.21481919e-03, 1.17585063e-03, 1.36753917e-03]]], dtype=float32)}
    """

    def __init__(
        self,
        svc_name: str,
        namespace: str,
        port: Optional[int] = 8001,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__(f"{svc_name}.{namespace}", port, verbose)


class BaseTritonPythonModel(ABC):
    """Triton Inference Server에서 Python backend 사용을 위한 class

    Note:
        Abstract Base Class: Model의 추론을 수행하는 abstract method ``_inference`` 정의 후 사용

    Hint:
        Logger의 색상 적용을 위해 아래와 같은 환경 변수 정의 필요

        .. code-block:: yaml

            spec:
              template:
                spec:
                  containers:
                    - name: ${NAME}
                      ...
                      env:
                        - name: "FORCE_COLOR"
                        value: "1"
                      ...

    Attributes:
        logger (``zerohertzLib.logging.Logger``): Triton Inference Server 내 log를 출력하기 위한 instance

    Methods:
        _inference:
            Model 추론을 수행하는 private method (상속을 통한 재정의 필수)

            Args:
                inputs (``NDArray[DTypeLike]``): Model 추론 시 사용될 입력 (``config.pbtxt`` 의 입력에 따라 입력 결정)

            Returns:
                ``Union[NDArray[DTypeLike], Tuple[NDArray[DTypeLike]]]``: Model의 추론 결과

    Examples:
        ``model.py``:
            .. code-block:: python

                class TritonPythonModel(zz.mlops.BaseTritonPythonModel):
                    def initialize(self, args):
                        super().initialize(args, 10)
                        self.model = Model(cfg)

                    def _inference(self, input_image):
                        return self.model(input_image)

        Normal Logs:
            .. code-block:: apl

                [04/04/24 00:00:00] INFO     [MODEL] Initialize                        triton.py:*
                [04/04/24 00:00:00] INFO     [MODEL] Called                            triton.py:*
                                    DEBUG    [MODEL] inputs: (3, 3, 3)                 triton.py:*
                                    INFO     [MODEL] Inference start                   triton.py:*
                                    DEBUG    [MODEL] outputs: (10,) (20,)              triton.py:*
                                    INFO     [MODEL] Inference completed               triton.py:*

        Error Logs:
            .. code-block:: apl

                [04/04/24 00:00:00] INFO     [MODEL] Called                            triton.py:*
                                    INFO     [MODEL] Inference start                   triton.py:*
                                    CRITICAL [MODEL] Hello, World!                     triton.py:*
                                            ====================================================================================================
                                            Traceback (most recent call last):
                                            File "/usr/local/lib/python3.8/dist-packages/zerohertzLib/mlops/triton.py", line *, in execute
                                                outputs = self._inference(*inputs)
                                            File "/models/model/*/model.py", line *, in _inference
                                                raise Exception("Hello, World!")
                                            Exception: Hello, World!
                                            ====================================================================================================
    """

    def initialize(self, args: Dict[str, Any], level: Optional[int] = 20) -> None:
        """Triton Inference Server 시작 시 수행되는 method

        Args:
            args (``Dict[str, Any]``): ``config.pbtxt`` 에 포함된 model의 정보
            level (``Optional[int]``): Logger의 level
        """
        self.cfg = json.loads(args["model_config"])
        self.logger = Logger(
            self.cfg["name"].upper(),
            170,
            file_name=self.cfg["name"],
            logger_level=level,
        )
        self.logger.info("Initialize")

    def execute(self, requests: List[Any]) -> List[Any]:
        """Triton Inference Server 호출 시 수행되는 method

        Args:
            requests (``List[pb_utils.InferenceRequest]``): Client에서 전송된 model inputs

        Returns:
            ``List[pb_utils.InferenceResponse]``: Client에 응답할 model의 추론 결과
        """
        responses = []
        for request in requests:
            try:
                self.logger.info("Called")
                inputs = self._get_inputs(request)
                self.logger.debug(
                    "inputs: %s", " ".join([str(input_.shape) for input_ in inputs])
                )
                self.logger.info("Inference start")
                outputs = self._inference(*inputs)
                if not isinstance(outputs, tuple):
                    outputs = tuple([outputs])
                self.logger.debug(
                    "outputs: %s", " ".join([str(output.shape) for output in outputs])
                )
                self.logger.info("Inference completed")
                response = self._set_outputs(outputs)
                responses.append(response)
            except Exception as error:
                message = (
                    str(error)
                    + "\n"
                    + "=" * 100
                    + "\n"
                    + str(traceback.format_exc())
                    + "=" * 100
                )
                self.logger.critical(message)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(message)
                    )
                )
        return responses

    def _get_inputs(self, request: Any) -> List[NDArray[DTypeLike]]:
        inputs = []
        for input_ in self.cfg["input"]:
            inputs.append(
                pb_utils.get_input_tensor_by_name(request, input_["name"]).as_numpy()
            )
        return inputs

    def _set_outputs(self, outputs: Tuple[NDArray[DTypeLike]]) -> Any:
        output_tensors = []
        for output, value in zip(self.cfg["output"], outputs):
            output_tensors.append(
                pb_utils.Tensor(
                    output["name"],
                    value.astype(pb_utils.triton_string_to_numpy(output["data_type"])),
                )
            )
        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    @abstractmethod
    def _inference(
        self, *inputs: NDArray[DTypeLike]
    ) -> Union[NDArray[DTypeLike], Tuple[NDArray[DTypeLike]]]:
        return inputs

    def finalize(self) -> None:
        """Triton Inference Server 종료 시 수행되는 method"""
        self.logger.info("Finalize")
