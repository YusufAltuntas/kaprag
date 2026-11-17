import pytest

pytestmark = pytest.mark.gpu


def test_torch_sees_cuda():
    # Imported here so collection works without the `ml` group installed.
    import torch

    assert torch.cuda.is_available()
