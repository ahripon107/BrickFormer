"""Reference parity for the recurrent kernels: chunked torch vs sequential, and vs fla / mamba_ssm."""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from brickformer.mixers.gdn import GatedDeltaNet, gated_delta_rule_chunked, gated_delta_rule_recurrent
from brickformer.mixers.mamba2 import Mamba2, ssd_chunked, ssd_recurrent

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
has_fla = importlib.util.find_spec("fla") is not None
has_mamba = importlib.util.find_spec("mamba_ssm") is not None


def _gdn_inputs(b=2, t=77, h=3, k=32, v=48, device="cpu", dtype=torch.float32):
    q = F.normalize(torch.randn(b, t, h, k, device=device, dtype=dtype), dim=-1)
    kk = F.normalize(torch.randn(b, t, h, k, device=device, dtype=dtype), dim=-1)
    vv = torch.randn(b, t, h, v, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(b, t, h, device=device, dtype=torch.float32))
    beta = torch.rand(b, t, h, device=device, dtype=dtype)
    s0 = torch.randn(b, h, k, v, device=device, dtype=torch.float32)
    return q, kk, vv, g, beta, s0


@pytest.mark.parametrize("chunk", [8, 16, 64])
def test_gdn_chunked_matches_recurrent(chunk):
    q, k, v, g, beta, s0 = (z.double() for z in _gdn_inputs())
    o1, s1 = gated_delta_rule_chunked(q, k, v, g, beta, initial_state=s0, chunk_size=chunk)
    o2, s2 = gated_delta_rule_recurrent(q, k, v, g, beta, initial_state=s0)
    torch.testing.assert_close(o1, o2)
    torch.testing.assert_close(s1, s2)


@pytest.mark.parametrize("chunk", [8, 32])
def test_ssd_chunked_matches_recurrent(chunk):
    b, t, h, p, n = 2, 45, 4, 8, 6
    x = torch.randn(b, t, h, p, dtype=torch.float64)
    a = -torch.rand(b, t, h, dtype=torch.float64)
    bb, cc = torch.randn(b, t, h, n, dtype=torch.float64), torch.randn(b, t, h, n, dtype=torch.float64)
    s0 = torch.randn(b, h, p, n, dtype=torch.float64)
    y1, f1 = ssd_chunked(x, a, bb, cc, chunk, s0)
    y2, f2 = ssd_recurrent(x, a, bb, cc, s0)
    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(f1, f2)


@cuda
@pytest.mark.skipif(not has_fla, reason="fla not installed")
def test_gdn_matches_fla_kernel():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q, k, v, g, beta, s0 = _gdn_inputs(t=200, k=64, v=64, device="cuda")
    o_ref, s_ref = gated_delta_rule_chunked(q, k, v, g, beta, initial_state=s0)
    o, s = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=s0, output_final_state=True)
    torch.testing.assert_close(o.float(), o_ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(s.float(), s_ref, atol=2e-3, rtol=2e-3)


@cuda
@pytest.mark.skipif(not has_fla, reason="fla not installed")
def test_gdn_layer_fla_backend_matches_torch_backend():
    layer = GatedDeltaNet(128, heads=2, head_dim=64, backend="torch").cuda()
    x = torch.randn(2, 150, 128, device="cuda")
    y_ref, _ = layer(x)
    layer.backend = "fla"
    y, _ = layer(x)
    torch.testing.assert_close(y, y_ref, atol=2e-3, rtol=2e-3)
    # with state (prefill) too
    st_ref = layer.init_state(2, 150, device="cuda")
    st = layer.init_state(2, 150, device="cuda")
    layer.backend = "torch"
    _, st_ref = layer(x, st_ref)
    layer.backend = "fla"
    _, st = layer(x, st)
    torch.testing.assert_close(st.recurrent, st_ref.recurrent, atol=2e-3, rtol=2e-3)


@cuda
@pytest.mark.skipif(not has_mamba, reason="mamba_ssm not installed")
def test_mamba2_kernel_backend_matches_torch_backend():
    layer = Mamba2(256, d_state=64, headdim=64, backend="torch").cuda()
    x = torch.randn(2, 130, 256, device="cuda")
    y_ref, _ = layer(x)
    layer.backend = "mamba_ssm"
    y, _ = layer(x)
    torch.testing.assert_close(y, y_ref, atol=2e-3, rtol=2e-3)


def test_missing_backend_raises_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith(("fla", "mamba_ssm")):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="flash-linear-attention"):
        GatedDeltaNet(64, heads=2, backend="fla")
    with pytest.raises(ImportError, match="mamba-ssm"):
        Mamba2(64, headdim=16, backend="mamba_ssm")
    # 'auto' silently falls back to torch
    GatedDeltaNet(64, heads=2)(torch.randn(1, 5, 64))
