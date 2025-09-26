import numpy as np
import pytest

import zerohertzLib as zz

TRITON_INFERENCE_SERVER_URL = "localhost"
TRITON_INFERENCE_SERVER_PORT = 8001


@pytest.fixture(scope="module")
def triton_client() -> zz.mlops.TritonClientURL:
    client = zz.mlops.TritonClientURL(
        TRITON_INFERENCE_SERVER_URL, port=TRITON_INFERENCE_SERVER_PORT
    )

    client.status()
    client.load_model(client.models.index("without_batching"))
    client.load_model(client.models.index("dynamic_batching"))

    yield client

    client.status(renew=True, sortby="ID", reverse=True)
    client.unload_model(0)
    client.unload_model(1)


@pytest.fixture(scope="module")
def test_data() -> tuple[np.ndarray, np.ndarray]:
    images = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    boxes = np.random.rand(10, 4).astype(np.float32)
    return images, boxes


@pytest.fixture(scope="module")
def test_data_empty_boxes() -> tuple[np.ndarray, np.ndarray]:
    images = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    boxes = np.empty((0, 4), dtype=np.float32)
    return images, boxes


class TestTritonClientURL:
    def test_server_connection(self, triton_client: zz.mlops.TritonClientURL) -> None:
        assert triton_client.is_server_ready()
        assert isinstance(triton_client.models, list)
        assert len(triton_client.models) > 0

    def test_model_availability(self, triton_client: zz.mlops.TritonClientURL) -> None:
        assert "without_batching" in triton_client.models
        assert "dynamic_batching" in triton_client.models

    def test_status_display(self, triton_client: zz.mlops.TritonClientURL) -> None:
        triton_client.status()
        triton_client.status(renew=True)
        triton_client.status(sortby="MODEL", reverse=True)

    def test_model_by_index(
        self,
        triton_client: zz.mlops.TritonClientURL,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        images, boxes = test_data
        result = triton_client(
            triton_client.models.index("without_batching"), images, boxes
        )
        assert isinstance(result, dict)

    def test_without_batching_inference(
        self,
        triton_client: zz.mlops.TritonClientURL,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        images, boxes = test_data
        result = triton_client("without_batching", images, boxes)

        assert isinstance(result, dict)
        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result

        assert result["boxes"].dtype == np.float32
        assert result["scores"].dtype == np.float32
        assert result["labels"].dtype == np.int64

        assert result["boxes"].ndim == 2
        assert result["boxes"].shape[1] == 4
        assert result["scores"].ndim == 1
        assert result["labels"].ndim == 1
        assert (
            result["boxes"].shape[0]
            == result["scores"].shape[0]
            == result["labels"].shape[0]
        )

    def test_dynamic_batching_inference(
        self,
        triton_client: zz.mlops.TritonClientURL,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        batch_size = 8

        image, box = test_data
        image = image[np.newaxis, ...]
        box = box[np.newaxis, ...]
        images = np.concatenate([image] * batch_size, axis=0)
        boxes = np.concatenate([box] * batch_size, axis=0)

        result = triton_client("dynamic_batching", images, boxes)

        assert isinstance(result, dict)
        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result
        assert "batch_index" in result

        assert result["batch_index"].dtype == np.int64

        assert result["boxes"].ndim == 2
        assert result["boxes"].shape[1] == 4
        assert result["scores"].ndim == 1
        assert result["labels"].ndim == 1
        assert result["batch_index"].ndim == 1

        assert (
            result["boxes"].shape[0]
            == result["scores"].shape[0]
            == result["labels"].shape[0]
            == result["batch_index"].shape[0]
        )
        assert batch_size == len(np.unique(result["batch_index"]))

    def test_without_batching_inference_empty_boxes(
        self,
        triton_client: zz.mlops.TritonClientURL,
        test_data_empty_boxes: tuple[np.ndarray, np.ndarray],
    ) -> None:
        images, boxes = test_data_empty_boxes
        result = triton_client("without_batching", images, boxes)

        assert isinstance(result, dict)
        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result

        assert result["boxes"].dtype == np.float32
        assert result["scores"].dtype == np.float32
        assert result["labels"].dtype == np.int64

        assert result["boxes"].ndim == 2
        assert result["boxes"].shape[1] == 4
        assert result["scores"].ndim == 1
        assert result["labels"].ndim == 1
        assert (
            0
            == result["boxes"].shape[0]
            == result["scores"].shape[0]
            == result["labels"].shape[0]
        )

    def test_dynamic_batching_inference_empty_boxes(
        self,
        triton_client: zz.mlops.TritonClientURL,
        test_data_empty_boxes: tuple[np.ndarray, np.ndarray],
    ) -> None:
        batch_size = 8

        image, box = test_data_empty_boxes
        image = image[np.newaxis, ...]
        box = box[np.newaxis, ...]
        images = np.concatenate([image] * batch_size, axis=0)
        boxes = np.concatenate([box] * batch_size, axis=0)

        result = triton_client("dynamic_batching", images, boxes)

        assert isinstance(result, dict)
        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result
        assert "batch_index" in result

        assert result["batch_index"].dtype == np.int64

        assert result["boxes"].ndim == 2
        assert result["boxes"].shape[1] == 4
        assert result["scores"].ndim == 1
        assert result["labels"].ndim == 1
        assert result["batch_index"].ndim == 1

        assert (
            result["boxes"].shape[0]
            == result["scores"].shape[0]
            == result["labels"].shape[0]
            == result["batch_index"].shape[0]
        )
        assert batch_size == len(np.unique(result["batch_index"]))
