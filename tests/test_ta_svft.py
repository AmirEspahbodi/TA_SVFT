import copy
import pytest
import torch
from torch import nn
from svft.ta_svft import TASVFT, TASVFTConfig


def make(shape=(5, 3), **options):
    torch.manual_seed(21)
    model = nn.Sequential(nn.Linear(shape[1], shape[0], dtype=torch.float64), nn.Tanh(),
                          nn.Linear(shape[0], 2, dtype=torch.float64))
    original = copy.deepcopy(model)
    cfg = TASVFTConfig(off_budget=4, calibration_batches=2, basis_dtype="float64", **options)
    ctl = TASVFT(model, ["0", "2"], cfg, head_paths=())
    batches = [(torch.randn(4, shape[1], dtype=torch.float64), torch.randn(4, 2, dtype=torch.float64)) for _ in range(2)]
    factory = lambda: iter(batches)
    loss = lambda m, b: (m(b[0]) - b[1]).square().mean()
    return ctl, original, batches, factory, loss


@pytest.mark.parametrize("shape", [(3, 3), (5, 3), (3, 5)])
def test_math_gradients_initialization_and_merge(shape):
    ctl, original, batches, factory, loss = make(shape)
    x = batches[0][0]
    torch.testing.assert_close(ctl.model(x), original(x), rtol=0, atol=0)
    ctl.select_support(factory, loss)
    ctl.validate_support()
    torch.testing.assert_close(ctl.model(x), original(x), rtol=0, atol=0)
    with torch.no_grad():
        ctl.bank.off_values.normal_()
        for layer in ctl.layers.values():
            layer.diagonal.normal_()
    for layer in ctl.layers.values():
        m = torch.zeros(layer.u.shape[1], layer.v.shape[1], dtype=torch.float64)
        m[range(layer.rank), range(layer.rank)] = layer.diagonal.detach()
        m[layer.rows, layer.cols] = ctl.bank.off_values[layer.slots].detach()
        torch.testing.assert_close(layer.spectral_update(), layer.u @ m @ layer.v.T)
        m = m.detach().requires_grad_()
        g = torch.randn_like(layer.weight)
        grad, = torch.autograd.grad(((layer.u @ m @ layer.v.T) * g).sum(), m)
        torch.testing.assert_close(grad, layer.projected_gradient(g))
    loss(ctl.model, batches[0]).backward()
    assert all(p.grad is not None for p in ctl.model.parameters())
    assert all(b.grad is None and not b.requires_grad for b in ctl.model.buffers())
    merged = ctl.merged_copy()
    torch.testing.assert_close(ctl.model(x), merged(x), atol=1e-12, rtol=1e-12)
    assert not hasattr(merged, "_ta_svft")
    assert ctl.report()["adapter_parameters"] == sum(l.rank for l in ctl.layers.values()) + 4


def test_global_selection_matches_dense_reference():
    ctl, _, _, factory, loss = make(complement_rank=0)
    candidates = []
    for name, layer in ctl.layers.items():
        grads = list(ctl._gradients(layer, factory, loss))
        score = torch.stack([layer.projected_gradient(g).square() for g in grads]).mean(0)
        for i in range(layer.rank):
            for j in range(layer.rank):
                if i != j:
                    candidates.append((float(score[i, j]), name, i, j))
    expected = {(n, i, j) for _, n, i, j in sorted(candidates, key=lambda x: (-x[0], x[1:]))[:4]}
    ctl.select_support(factory, loss)
    actual = {(n, int(i), int(j)) for n, l in ctl.layers.items() for i, j in zip(l.rows, l.cols)}
    assert actual == expected


@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_complements_orthogonal(shape):
    ctl, _, _, factory, loss = make(shape, complement_rank=2)
    ctl.select_support(factory, loss)
    layer = ctl.layers["0"]
    for basis in (layer.u, layer.v):
        torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1], dtype=basis.dtype), atol=1e-12, rtol=1e-12)
    assert max(layer.u.shape[1], layer.v.shape[1]) > layer.rank


def test_checkpoint_full_and_adapter(tmp_path):
    ctl, original, batches, factory, loss = make()
    ctl.select_support(factory, loss)
    optimizer = torch.optim.AdamW(ctl.model.parameters(), lr=.01)
    loss(ctl.model, batches[0]).backward()
    optimizer.step()
    expected = ctl.model(batches[0][0]).detach()
    ctl.save_adapter(tmp_path / "adapter.pt")
    restored = TASVFT.load_adapter(copy.deepcopy(original), tmp_path / "adapter.pt")
    torch.testing.assert_close(expected, restored.model(batches[0][0]), rtol=0, atol=0)
    fresh = TASVFT(copy.deepcopy(original), ctl.targets, ctl.config, head_paths=())
    fresh.model.load_state_dict(ctl.model.state_dict())
    fresh.validate_support()
    torch.testing.assert_close(expected, fresh.model(batches[0][0]), rtol=0, atol=0)


def test_refinement_optimizer_and_global_budget():
    ctl, _, batches, factory, loss = make(replace_fraction=.5)
    ctl.select_support(factory, loss)
    optimizer = torch.optim.AdamW(ctl.model.parameters(), lr=.01, amsgrad=True)
    loss(ctl.model, batches[0]).backward()
    optimizer.step()
    old = {(n, int(i), int(j)): int(s) for n, l in ctl.layers.items() for i, j, s in zip(l.rows, l.cols, l.slots)}
    parameter = ctl.bank.off_values
    old_value = parameter.detach().clone()
    old_moment = optimizer.state[parameter]["exp_avg"].clone()
    ctl.select_support(factory, loss, optimizer, refine=True)
    ctl.validate_support()
    new = {(n, int(i), int(j)): int(s) for n, l in ctl.layers.items() for i, j, s in zip(l.rows, l.cols, l.slots)}
    assert len(new.keys() - old.keys()) == 2
    assert parameter is ctl.bank.off_values
    for key, slot in new.items():
        if key in old:
            assert slot == old[key]
            assert parameter[slot] == old_value[slot]
            assert optimizer.state[parameter]["exp_avg"][slot] == old_moment[slot]
        else:
            assert parameter[slot] == 0
            for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                assert optimizer.state[parameter][field][slot] == 0
    optimizer.zero_grad(set_to_none=True)
    loss(ctl.model, batches[1]).backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()


def test_determinism_and_budget_errors():
    first, _, _, factory, loss = make()
    second, _, _, factory2, loss2 = make()
    first.select_support(factory, loss)
    second.select_support(factory2, loss2)
    for name in first.targets:
        assert torch.equal(first.layers[name].rows, second.layers[name].rows)
        assert torch.equal(first.layers[name].cols, second.layers[name].cols)
    cfg = TASVFTConfig(off_budget=7, complement_rank=0, calibration_batches=1)
    ctl = TASVFT(nn.Sequential(nn.Linear(2, 2)), ["0"], cfg, head_paths=())
    with pytest.raises(ValueError, match="exceeds"):
        ctl.select_support(lambda: iter([torch.randn(2, 2)]), lambda m, b: m(b).sum())


def test_diagonal_disabled_and_zero_budget():
    ctl, _, batches, factory, loss = make(diagonal=False, complement_rank=0)
    ctl.select_support(factory, loss)
    assert ctl.report()["adapter_parameters"] == 4
    loss(ctl.model, batches[0]).backward()
    cfg = TASVFTConfig(off_budget=0, complement_rank=0, calibration_batches=1)
    m = nn.Sequential(nn.Linear(3, 3))
    ctl = TASVFT(m, ["0"], cfg, head_paths=())
    ctl.select_support(lambda: iter([torch.randn(2, 3)]), lambda m, b: m(b).sum())
    ctl.validate_support()
    m(torch.randn(2, 3)).sum().backward()


def test_zero_residual_no_spurious_complement_and_full_support_refinement():
    cfg = TASVFTConfig(off_budget=2, complement_rank=0, calibration_batches=1, replace_fraction=1)
    ctl = TASVFT(nn.Sequential(nn.Linear(2, 2)), ["0"], cfg, head_paths=())
    factory = lambda: iter([torch.eye(2)])
    loss = lambda m, b: m(b).sum()
    ctl.select_support(factory, loss)
    opt = torch.optim.SGD(ctl.model.parameters(), lr=.01, momentum=.9)
    ctl.select_support(factory, loss, opt, refine=True)
    ctl.validate_support()
    tall, _, _, _, _ = make((5, 3))
    layer = tall.layers["0"]
    layer.augment(torch.zeros_like(layer.weight), 2)
    assert layer.u.shape[1] == layer.rank


def test_calibration_restores_rng_and_trainability():
    import random
    import numpy as np
    ctl, _, _, factory, loss = make()
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    flags = [p.requires_grad for p in ctl.model.parameters()]
    ctl.select_support(factory, loss)
    assert flags == [p.requires_grad for p in ctl.model.parameters()]
    assert before[0] == random.getstate()
    np.testing.assert_equal(before[1], np.random.get_state())
    assert torch.equal(before[2], torch.get_rng_state())
