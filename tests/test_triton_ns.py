import pytest
import torch
pytest.importorskip("triton")
from gram_newton_schulz_ampere.kernels.triton_ns import triton_baddbmm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_baddbmm_masks_partial_tiles() -> None:
    generator = torch.Generator(device="cuda").manual_seed(67)
    a = torch.randn(2, 35, 19, device="cuda", generator=generator)
    b = torch.randn(2, 19, 37, device="cuda", generator=generator)
    c = torch.randn(2, 35, 37, device="cuda", generator=generator)

    expected = torch.baddbmm(c, a, b, beta=0.3, alpha=0.7)
    actual = triton_baddbmm(c, a, b, beta=0.3, alpha=0.7)


    # since tl.dot uses TF32 on ampere while torch.baddbmm is fp32, I set rtol and atol to 2e-2
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
