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

from typing import Dict, List, Optional

import tritonclient.grpc as grpcclient
from numpy.typing import DTypeLike, NDArray
from tritonclient.utils import triton_to_np_dtype


class TritonClientURL:
    """외부에서 실행되는 triton inference server의 호출을 위한 class

    Args:
        url (``str``): 호출할 triton inference server의 URL
        model_name(``str``): 호출할 triton inference server 내 model의 이름
        port (``Optional[int]``): triton inference server의 GRPC 통신 port 번호

    Attributes:
        inputs (``List[Dict[str, Any]]``): 지정된 model의 입력
        outputs (``List[Dict[str, Any]]``): 지정된 model의 출력

    Methods:
        __call__:
            Model 호출 수행

            Args:
                *args (``List[NDArray[DTypeLike]]``): Model 호출 시 사용될 입력 (``self.inputs``)

            Returns:
                ``Dict[str, NDArray[DTypeLike]]``: 호출된 model의 결과

    Examples:
        >>> tc = zz.mlops.TritonClientURL("localhost", "YOLO")
        >>> tc.inputs
        [{'name': 'images', 'data_type': 'TYPE_FP32', 'dims': ['1', '3', '640', '640']}]
        >>> tc.outputs
        [{'name': 'output0', 'data_type': 'TYPE_FP32', 'dims': ['1', '25200', '85']}]
        >>> tc(np.zeros((1, 3, 640, 640)))
        {'output0': array([[[3.90108061e+00, 3.51982164e+00, 7.49971962e+00, ...,
        2.21481919e-03, 1.17585063e-03, 1.36753917e-03]]], dtype=float32)}
    """

    def __init__(self, url: str, model_name: str, port: Optional[int] = 8001) -> None:
        self.server_url = f"{url}:{port}"
        self.model_name = model_name
        self.triton_client = grpcclient.InferenceServerClient(
            url=self.server_url, verbose=False
        )
        self.info = self.triton_client.get_model_config(model_name, as_json=True)
        assert self.info["config"]["name"] == model_name
        self.inputs = self.info["config"]["input"]
        self.outputs = self.info["config"]["output"]

    def __call__(self, *args) -> Dict[str, NDArray[DTypeLike]]:
        assert len(self.inputs) == len(args)
        triton_inputs = []
        for input_info, arg in zip(self.inputs, args):
            triton_inputs.append(self._set_input(input_info, arg))
        triton_outputs = []
        for output in self.outputs:
            triton_outputs.append(grpcclient.InferRequestedOutput(output["name"]))
        response = self.triton_client.infer(
            model_name=self.model_name, inputs=triton_inputs, outputs=triton_outputs
        )
        response.get_response()
        triton_results = {}
        for output in self.outputs:
            triton_results[output["name"]] = response.as_numpy(output["name"])
        return triton_results

    def _set_input(self, input_info: Dict[str, List[int]], value: NDArray[DTypeLike]):
        assert len(input_info["dims"]) == len(value.shape)
        value = value.astype(triton_to_np_dtype(input_info["data_type"][5:]))
        return grpcclient.InferInput(
            input_info["name"],
            value.shape,
            input_info["data_type"][5:],
        ).set_data_from_numpy(value)


class TritonClientK8s(TritonClientURL):
    """Kubernetes에서 실행되는 triton inference server의 호출을 위한 class

    Args:
        svc_name (``str``): 호출할 triton inference server의 Kubernetes service의 이름
        namespace (``str``): 호출할 triton inference server의 namespace
        model_name(``str``): 호출할 triton inference server 내 model의 이름
        port (``Optional[int]``): triton inference server의 GRPC 통신 port 번호

    Attributes:
        inputs (``List[Dict[str, Any]]``): 지정된 model의 입력
        outputs (``List[Dict[str, Any]]``): 지정된 model의 출력

    Methods:
        __call__:
            Model 호출 수행

            Args:
                *args (``List[NDArray[DTypeLike]]``): Model 호출 시 사용될 입력 (``self.inputs``)

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

        >>> tc = zz.mlops.TritonClientK8s("triton-inference-server-svc", "yolo", "YOLO")
        >>> tc.inputs
        [{'name': 'images', 'data_type': 'TYPE_FP32', 'dims': ['1', '3', '640', '640']}]
        >>> tc.outputs
        [{'name': 'output0', 'data_type': 'TYPE_FP32', 'dims': ['1', '25200', '85']}]
        >>> tc(np.zeros((1, 3, 640, 640)))
        {'output0': array([[[3.90108061e+00, 3.51982164e+00, 7.49971962e+00, ...,
        2.21481919e-03, 1.17585063e-03, 1.36753917e-03]]], dtype=float32)}
    """

    def __init__(
        self, svc_name: str, namespace: str, model_name: str, port: Optional[int] = 8001
    ) -> None:
        super().__init__(f"{svc_name}.{namespace}", model_name, port)
