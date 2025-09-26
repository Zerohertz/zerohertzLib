# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import tritonclient.grpc as grpcclient
from loguru import logger
from numpy.typing import DTypeLike, NDArray
from prettytable import PrettyTable
from tritonclient.utils import triton_to_np_dtype


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
        max_batch_size = self.configs[model]["config"].get("max_batch_size", None)
        assert len(inputs) == len(args)
        triton_inputs = []
        for input_info, arg in zip(inputs, args):
            triton_inputs.append(self._set_input(input_info, arg, max_batch_size))
        triton_outputs = []
        for output in outputs:
            triton_outputs.append(grpcclient.InferRequestedOutput(output["name"]))
        response = self.infer(
            model_name=model, inputs=triton_inputs, outputs=triton_outputs
        )
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
