"""Task-Adaptive SVFT: frozen spectral dictionaries and a global sparse budget.

Linear weights use [out, in]; V stores columns, unlike torch.linalg.svd's Vh.
Calibration is deliberately serial over layers: no model-wide dense statistics.
"""
from __future__ import annotations

import copy
import json
import math
import random
import re
import numpy as np
import time
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TASVFTConfig:
    off_budget: int = 1024
    diagonal: bool = True
    complement_rank: int = 4
    calibration_batches: int = 8
    selection: str = "gradient"  # random is a matched-budget ablation
    seed: int = 0
    update_interval: int = 0  # optimizer steps; 0 = one-shot
    freeze_step: int = 1000
    replace_fraction: float = 0.1
    basis_dtype: str = "float32"

    def __post_init__(self):
        for name in ("off_budget", "complement_rank", "update_interval", "freeze_step"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.calibration_batches < 1 or not 0 <= self.replace_fraction <= 1:
            raise ValueError("Need calibration_batches >= 1 and replace_fraction in [0, 1]")
        if self.selection not in {"gradient", "random"}:
            raise ValueError("selection must be gradient or random")
        if self.basis_dtype not in {"float32", "float64"}:
            raise ValueError("basis_dtype must be float32 or float64")


class SpectralLinear(nn.Module):
    def __init__(self, linear: nn.Linear, config: TASVFTConfig, decompose: bool = True):
        super().__init__()
        if type(linear) is not nn.Linear:
            raise TypeError("TA-SVFT requires ordinary, unquantized nn.Linear targets")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.rank = min(linear.weight.shape)
        self.register_buffer("weight", linear.weight.detach())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach())
        dtype = getattr(torch, config.basis_dtype)
        if decompose:
            u, sigma, vh = torch.linalg.svd(self.weight.to(dtype), full_matrices=False)
        else:
            u = torch.empty(self.out_features, self.rank, device=self.weight.device, dtype=dtype)
            sigma = torch.empty(self.rank, device=self.weight.device, dtype=dtype)
            vh = torch.empty(self.rank, self.in_features, device=self.weight.device, dtype=dtype)
        self.register_buffer("u", u.contiguous())
        self.register_buffer("v", vh.T.contiguous())
        self.register_buffer("sigma", sigma)
        self.diagonal = nn.Parameter(torch.zeros(self.rank if config.diagonal else 0,
                                                 device=self.weight.device, dtype=dtype))
        for key in ("rows", "cols", "slots"):
            self.register_buffer(key, torch.empty(0, dtype=torch.long, device=self.weight.device))
        self._bank_ref = None
        self._probe = None

    def bind(self, bank):
        self._bank_ref = weakref.ref(bank)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Variable support sizes and complement ranks are checkpoint state, not calibration.
        for key in ("u", "v", "rows", "cols", "slots"):
            if prefix + key in state_dict:
                old = getattr(self, key)
                setattr(self, key, torch.empty_like(state_dict[prefix + key], device=old.device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def sparse_coefficients(self) -> Tensor:
        # Sparse [left, right] coordinates; no dense trainable M is ever created.
        bank = self._bank_ref()
        if bank is None:
            raise RuntimeError("Detached TA-SVFT bank; use attach/load_adapter/merged_copy")
        rows, cols = self.rows, self.cols
        values = bank.off_values[self.slots]
        if self.diagonal.numel():
            diag = torch.arange(self.rank, device=rows.device)
            rows, cols = torch.cat((diag, rows)), torch.cat((diag, cols))
            values = torch.cat((self.diagonal, values))
        sparse = torch.sparse_coo_tensor(torch.stack((rows, cols)), values,
                                        (self.u.shape[1], self.v.shape[1]), check_invariants=False).coalesce()
        return sparse

    def spectral_update(self) -> Tensor:
        return self.u @ torch.sparse.mm(self.sparse_coefficients(), self.v.T)

    def forward(self, x: Tensor) -> Tensor:
        # Probe is a temporary leaf only during support estimation, never an optimizer param.
        weight = self.weight if self._probe is None else self._probe
        base = F.linear(x, weight, self.bias)
        # Apply sparse spectral coefficients to activations. Dense delta-W is only
        # materialized for explicit mathematical inspection or inference merging.
        sparse = self.sparse_coefficients()
        if self._probe is not None:
            sparse = sparse.detach()
        original_shape = x.shape[:-1]
        projected = x.reshape(-1, self.in_features).to(self.v.dtype) @ self.v
        mixed = torch.sparse.mm(sparse, projected.T).T
        update = (mixed @ self.u.T).reshape(*original_shape, self.out_features)
        return base + update.to(base.dtype)

    def merged(self) -> nn.Linear:
        linear = nn.Linear(self.in_features, self.out_features, self.bias is not None,
                           device=self.weight.device, dtype=self.weight.dtype)
        with torch.no_grad():
            linear.weight.copy_(self.weight + self.spectral_update().to(self.weight.dtype))
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def projected_gradient(self, gradient: Tensor) -> Tensor:
        return self.u.T @ gradient.to(self.u.dtype) @ self.v

    @torch.no_grad()
    def augment(self, mean_gradient: Tensor, p: int):
        if not p or self.out_features == self.in_features:
            return
        tall = self.out_features > self.in_features
        basis = self.u if tall else self.v
        g = mean_gradient.to(basis.dtype)
        g = g if tall else g.T
        residual = g - basis @ (basis.T @ g)
        vectors, values, _ = torch.linalg.svd(residual, full_matrices=False)
        # Reject numerical null residuals relative to the *original* gradient scale.
        tol = torch.finfo(g.dtype).eps * max(g.shape) * torch.linalg.norm(g)
        count = min(p, basis.shape[0] - basis.shape[1], int((values > tol).sum()))
        if count:
            vectors = vectors[:, :count]
            vectors = vectors - basis @ (basis.T @ vectors)
            vectors = torch.linalg.qr(vectors, mode="reduced").Q
            if tall:
                self.u = torch.cat((basis, vectors), dim=1)
            else:
                self.v = torch.cat((basis, vectors), dim=1)


class TASVFTBank(nn.Module):
    def forward(self, x):
        # Harmless when the host itself is nn.Sequential.
        return x

    def __init__(self, config: TASVFTConfig, device):
        super().__init__()
        self.off_values = nn.Parameter(torch.zeros(config.off_budget, device=device,
                                                   dtype=getattr(torch, config.basis_dtype)))
        self.register_buffer("selected_saliency", torch.zeros(config.off_budget, device=device, dtype=getattr(torch, config.basis_dtype)))
        self.register_buffer("initialized", torch.tensor(False, device=device))
        self.register_buffer("refinements", torch.tensor(0, dtype=torch.long, device=device))


def vit_targets(model: nn.Module, families: Iterable[str] = ("q", "k", "v", "o", "up", "down")):
    """Resolve actual HF ViT/DINOv2 transformer projections; never include the head."""
    suffixes = {
        "q": ("attention.attention.query",), "k": ("attention.attention.key",),
        "v": ("attention.attention.value",), "o": ("attention.output.dense",),
        "up": ("intermediate.dense", "mlp.fc1"), "down": ("output.dense", "mlp.fc2"),
    }
    families = tuple(families)
    if not families or set(families) - set(suffixes):
        raise ValueError("ViT families must be a nonempty subset of q k v o up down")
    result = []
    for name, module in model.named_modules():
        if type(module) is not nn.Linear or ".encoder.layer." not in "." + name:
            continue
        for family in families:
            if family == "down" and name.endswith("attention.output.dense"):
                continue
            if any(name.endswith(suffix) for suffix in suffixes[family]):
                result.append(name)
                break
    if not result:
        raise ValueError("No supported ViT projections found; supply exact nn.Linear target paths")
    return result


class TASVFT:
    """Controller. Owns no trainable state; model._ta_svft owns the global bank.

    Calibration loss_fn(model, batch) returns a scalar *mean minibatch loss*.
    batch_factory must return a fresh deterministic training-data iterator.
    """
    def __init__(self, model: nn.Module, targets: list[str], config: TASVFTConfig,
                 head_paths: tuple[str, ...] = ("classifier",), *, decompose: bool = True):
        if hasattr(model, "_ta_svft"):
            raise ValueError("Model already has TA-SVFT attached")
        self.model, self.config, self.targets = model, config, sorted(set(targets))
        self.head_paths = tuple(head_paths)
        self.timings = {"svd_seconds": 0.0, "support_seconds": 0.0}
        if not self.targets:
            raise ValueError("At least one target is required")
        modules = [model.get_submodule(n) for n in self.targets]
        if any(type(m) is not nn.Linear for m in modules):
            raise TypeError("All targets must be ordinary nn.Linear modules")
        if len({m.weight.device for m in modules}) != 1:
            raise ValueError("TA-SVFT currently requires targets on one device")
        for p in model.parameters():
            p.requires_grad_(False)
        for name in self.head_paths:
            for p in model.get_submodule(name).parameters():
                p.requires_grad_(True)
        model.add_module("_ta_svft", TASVFTBank(config, modules[0].weight.device))
        start = time.perf_counter()
        for name, old in zip(self.targets, modules):
            new = SpectralLinear(old, config, decompose)
            new.bind(model._ta_svft)
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, new)
        self.timings["svd_seconds"] = time.perf_counter() - start

    @property
    def layers(self):
        return {n: self.model.get_submodule(n) for n in self.targets}

    @property
    def bank(self):
        return self.model._ta_svft

    def _gradients(self, layer, batch_factory, loss_fn):
        device = layer.weight.device
        # Replaying the RNG makes every layer see the same minibatches/transforms.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            yield from self._replayed_gradients(layer, batch_factory, loss_fn, devices)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _replayed_gradients(self, layer, batch_factory, loss_fn, devices):
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed)
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            for index, batch in enumerate(batch_factory()):
                if index >= self.config.calibration_batches:
                    break
                layer._probe = layer.weight.detach().requires_grad_(True)
                try:
                    with torch.enable_grad():
                        loss = loss_fn(self.model, batch)
                        gradient, = torch.autograd.grad(loss, layer._probe)
                    if not torch.isfinite(gradient).all():
                        raise ValueError("Nonfinite calibration gradient")
                    yield gradient.detach()
                finally:
                    layer._probe = None

    @torch.no_grad()
    def select_support(self, batch_factory: Callable, loss_fn: Callable,
                       optimizer=None, *, refine: bool = False):
        if refine and not bool(self.bank.initialized):
            raise ValueError("Initialize support before refinement")
        if not refine and bool(self.bank.initialized):
            raise ValueError("Support already initialized; use refine=True")
        if refine and optimizer is None:
            raise ValueError("Refinement requires the live optimizer")
        if refine and not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD)):
            raise TypeError("Support refinement supports unsharded Adam, AdamW and SGD only")
        if refine and not any(p is self.bank.off_values for group in optimizer.param_groups for p in group["params"]):
            raise ValueError("Optimizer does not own the global coefficient bank")
        start = time.perf_counter()
        modes = {m: m.training for m in self.model.modules()}
        self.model.eval()
        trainability = {p: p.requires_grad for p in self.model.parameters()}
        for p in trainability:
            p.requires_grad_(False)
        layers = self.layers
        active = {(name, int(i), int(j)): int(slot)
                  for name, layer in layers.items()
                  for i, j, slot in zip(layer.rows, layer.cols, layer.slots)}
        budget = self.config.off_budget
        prune_count = math.floor(budget * self.config.replace_fraction) if refine else budget
        survivors = dict(active)
        if refine:
            weakest = sorted(active, key=lambda key: (abs(float(self.bank.off_values[active[key]])), key))
            for key in weakest[:prune_count]:
                del survivors[key]
        candidates = []
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed + int(self.bank.refinements))
        try:
            for name, layer in layers.items():
                if not refine and self.config.complement_rank and layer.in_features != layer.out_features:
                    mean, count = torch.zeros_like(layer.weight, dtype=layer.u.dtype), 0
                    for gradient in self._gradients(layer, batch_factory, loss_fn):
                        mean.add_(gradient.to(mean.dtype))
                        count += 1
                    if not count:
                        raise ValueError("Calibration iterator is empty")
                    layer.augment(mean / count, self.config.complement_rank)
                    del mean
                shape = (layer.u.shape[1], layer.v.shape[1])
                if self.config.selection == "random":
                    score = torch.rand(shape, generator=generator, dtype=layer.u.dtype)
                else:
                    score = torch.zeros(shape, device=layer.u.device, dtype=layer.u.dtype)
                    count = 0
                    for gradient in self._gradients(layer, batch_factory, loss_fn):
                        projected = layer.projected_gradient(gradient)
                        score.add_(projected.square())
                        count += 1
                    if not count:
                        raise ValueError("Calibration iterator is empty")
                    score = (score / count).cpu()
                if refine:
                    self.bank.selected_saliency[layer.slots] = score[layer.rows.cpu(), layer.cols.cpu()].to(self.bank.selected_saliency)
                diagonal = torch.arange(layer.rank)
                score[diagonal, diagonal] = -torch.inf
                # Exclude all old edges during regrowth, including just-pruned ones.
                if refine:
                    score[layer.rows.cpu(), layer.cols.cpu()] = -torch.inf
                flat = score.flatten()
                # Stable ties: layer path then row-major coordinate. CPU selection is reproducible.
                indices = torch.argsort(flat, descending=True, stable=True)[:prune_count]
                for idx in indices.tolist():
                    value = float(flat[idx])
                    if math.isfinite(value):
                        candidates.append((value, name, idx // shape[1], idx % shape[1]))
                del score, flat
                candidates = sorted(candidates, key=lambda x: (-x[0], x[1], x[2], x[3]))[:prune_count]
            if len(candidates) < prune_count:
                if not refine:
                    raise ValueError(f"off_budget={budget} exceeds {len(candidates)} available off-diagonal atoms")
                # Dense support may have too few inactive edges; retain additional old edges.
                missing = prune_count - len(candidates)
                dropped = [key for key in active if key not in survivors]
                for key in sorted(dropped, key=lambda k: (-abs(float(self.bank.off_values[active[k]])), k))[:missing]:
                    survivors[key] = active[key]
            used = set(survivors.values())
            free = sorted(set(range(budget)) - used)
            assignments = dict(survivors)
            for slot, (saliency, name, row, col) in zip(free, candidates):
                assignments[name, row, col] = slot
                self.bank.selected_saliency[slot] = saliency
            if len(assignments) != budget:
                raise RuntimeError("Global support budget invariant violated")
            if free:
                self.bank.off_values[free] = 0
                self.bank.off_values.grad = None
                if optimizer is not None:
                    state = optimizer.state.get(self.bank.off_values, {})
                    for value in state.values():
                        if torch.is_tensor(value) and value.shape == self.bank.off_values.shape:
                            value[free] = 0
            for name, layer in layers.items():
                edges = sorted((i, j, slot) for (n, i, j), slot in assignments.items() if n == name)
                data = torch.tensor(edges, device=layer.weight.device, dtype=torch.long).reshape(-1, 3)
                layer.rows, layer.cols, layer.slots = (data[:, i].contiguous() for i in range(3))
            self.bank.initialized.fill_(True)
            if refine:
                self.bank.refinements.add_(1)
        finally:
            for p, requires_grad in trainability.items():
                p.requires_grad_(requires_grad)
            for module, mode in modes.items():
                module.training = mode
            self.timings["support_seconds"] += time.perf_counter() - start

    def report(self, include_edges: bool = False):
        layers = {}
        for name, layer in self.layers.items():
            standard = (layer.rows < layer.rank) & (layer.cols < layer.rank)
            gaps = (layer.sigma[layer.rows[standard]] - layer.sigma[layer.cols[standard]]).abs()
            item = {"diagonal": layer.diagonal.numel(), "off_diagonal": layer.slots.numel(),
                    "complement_edges": int((~standard).sum()),
                    "basis_shape": [layer.u.shape[1], layer.v.shape[1]],
                    "mean_index_distance": float((layer.rows - layer.cols).abs().float().mean()) if layer.rows.numel() else None,
                    "mean_spectral_gap": float(gaps.mean()) if gaps.numel() else None}
            if include_edges:
                item["edges"] = torch.stack((layer.rows, layer.cols, layer.slots), dim=1).cpu().tolist()
                item["saliency"] = self.bank.selected_saliency[layer.slots].cpu().tolist()
                item["coefficients"] = self.bank.off_values[layer.slots].detach().cpu().tolist()
            layers[name] = item
        by_family, by_depth = {}, {}
        for name, item in layers.items():
            family = "other"
            for suffix, label in (("query", "q"), ("key", "k"), ("value", "v"),
                                  ("attention.output.dense", "o"), ("intermediate.dense", "up"),
                                  ("mlp.fc1", "up"), ("mlp.fc2", "down"), ("output.dense", "down")):
                if name.endswith(suffix):
                    family = label
                    break
            match = re.search(r"encoder\.layer\.(\d+)\.", name)
            depth = match.group(1) if match else "other"
            by_family[family] = by_family.get(family, 0) + item["off_diagonal"]
            by_depth[depth] = by_depth.get(depth, 0) + item["off_diagonal"]
        diagonal = sum(x["diagonal"] for x in layers.values())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        adapter = diagonal + self.bank.off_values.numel()
        return {"method": "TA-SVFT", "adapter_parameters": adapter,
                "diagonal_parameters": diagonal, "off_parameters": self.bank.off_values.numel(),
                "active_off_support": sum(x["off_diagonal"] for x in layers.values()),
                "head_parameters": trainable - adapter, "total_trainable_parameters": trainable,
                "frozen_buffer_bytes": sum(b.numel() * b.element_size() for b in self.model.buffers() if b.is_floating_point()),
                "allocation_by_family": by_family, "allocation_by_depth": by_depth,
                "refinements": int(self.bank.refinements), "timings": dict(self.timings), "layers": layers}

    def export_report(self, path, include_edges=True):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.report(include_edges), indent=2))

    def save_adapter(self, path):
        keys = [n + "." for n in self.targets]
        heads = [n + "." for n in self.head_paths]
        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()
                 if k.startswith("_ta_svft.") or any(k.startswith(h) for h in heads)
                 or (any(k.startswith(n) for n in keys) and not k.endswith((".weight", ".bias")))}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "config": asdict(self.config), "targets": self.targets,
                    "head_paths": self.head_paths, "state": state}, path)

    @classmethod
    def load_adapter(cls, model, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1:
            raise ValueError("Unsupported TA-SVFT checkpoint format")
        controller = cls(model, checkpoint["targets"], TASVFTConfig(**checkpoint["config"]),
                         tuple(checkpoint["head_paths"]), decompose=False)
        missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
        required = {k for k in model.state_dict() if k.startswith("_ta_svft.") or
                    any(k.startswith(n + ".") and not k.endswith((".weight", ".bias")) for n in controller.targets) or
                    any(k.startswith(n + ".") for n in controller.head_paths)}
        if unexpected or required.intersection(missing):
            raise ValueError("Incomplete or incompatible TA-SVFT checkpoint")
        controller.validate_support()
        return controller

    def validate_support(self):
        slots = []
        for layer in self.layers.values():
            if not (layer.rows.shape == layer.cols.shape == layer.slots.shape):
                raise ValueError("Invalid support metadata shapes")
            if layer.rows.numel():
                if (layer.rows < 0).any() or (layer.rows >= layer.u.shape[1]).any() or (layer.cols < 0).any() or (layer.cols >= layer.v.shape[1]).any():
                    raise ValueError("Out-of-range support coordinates")
                if (layer.rows == layer.cols).any():
                    raise ValueError("Off-diagonal support includes a diagonal")
                if torch.unique(layer.rows * layer.v.shape[1] + layer.cols).numel() != layer.rows.numel():
                    raise ValueError("Duplicate support coordinates")
            slots.extend(layer.slots.tolist())
        if sorted(slots) != list(range(self.config.off_budget)):
            raise ValueError("Checkpoint does not satisfy the global budget")

    def merge(self):
        self.validate_support()
        for name, layer in self.layers.items():
            parent, _, leaf = name.rpartition(".")
            setattr(self.model.get_submodule(parent), leaf, layer.merged())
        delattr(self.model, "_ta_svft")
        return self.model

    def merged_copy(self):
        copied = copy.deepcopy(self.model)
        for name in self.targets:
            copied.get_submodule(name).bind(copied._ta_svft)
        for name in self.targets:
            parent, _, leaf = name.rpartition(".")
            setattr(copied.get_submodule(parent), leaf, copied.get_submodule(name).merged())
        delattr(copied, "_ta_svft")
        return copied
