import copy
import pytest
import torch
from transformers import ViTConfig, ViTForImageClassification, TrainingArguments
from svft.ta_svft import TASVFT, TASVFTConfig, vit_targets
from svft.trainer import TASVFTTrainer


def model():
    torch.manual_seed(11)
    return ViTForImageClassification(ViTConfig(image_size=16, patch_size=8,
        hidden_size=12, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=3, num_labels=3, attn_implementation="eager"))


def data():
    return [{"pixel_values": torch.randn(3, 16, 16), "labels": torch.tensor(i % 3)} for i in range(8)]


def test_vit_target_families():
    m = model()
    assert len(vit_targets(m)) == 12
    assert len(vit_targets(m, ["q", "v"])) == 4
    assert len(vit_targets(m, ["up", "down"])) == 4
    assert len(vit_targets(m, ["o"])) == 2


def test_vit_training_refinement_resume_and_merge(tmp_path):
    m = model()
    original = copy.deepcopy(m)
    samples = data()
    config = TASVFTConfig(off_budget=20, complement_rank=2, calibration_batches=1,
                          update_interval=1, freeze_step=3, replace_fraction=.25)
    ctl = TASVFT(m, vit_targets(m), config)
    args = TrainingArguments(output_dir=str(tmp_path), max_steps=2,
        per_device_train_batch_size=2, save_steps=1, learning_rate=.01,
        report_to=[], use_cpu=True, disable_tqdm=True, save_safetensors=False)
    trainer = TASVFTTrainer(model=m, args=args, train_dataset=samples, ta_controller=ctl)
    trainer.train()
    assert int(ctl.bank.refinements) == 2
    ctl.validate_support()
    report = ctl.report()
    assert report["adapter_parameters"] == 12 * 12 + 20
    assert report["head_parameters"] == 12 * 3 + 3
    assert report["total_trainable_parameters"] == 203
    m.eval()
    pixels = torch.stack([x["pixel_values"] for x in samples[:2]])
    with torch.no_grad():
        expected = m(pixels).logits
        torch.testing.assert_close(expected, ctl.merged_copy()(pixels).logits, rtol=1e-5, atol=1e-6)
    ctl.save_adapter(tmp_path / "adapter.pt")
    restored = TASVFT.load_adapter(original, tmp_path / "adapter.pt")
    restored.model.eval()
    torch.testing.assert_close(expected, restored.model(pixels).logits, rtol=0, atol=0)
    # Real Trainer resume reloads variable-length support and optimizer state.
    fresh = model()
    resumed = TASVFT(fresh, vit_targets(fresh), config)
    args.max_steps = 3
    resume_trainer = TASVFTTrainer(model=fresh, args=args, train_dataset=samples, ta_controller=resumed)
    resume_trainer.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
    assert resume_trainer.state.global_step == 3
    assert int(resumed.bank.refinements) == 2
    resumed.validate_support()
    assert torch.isfinite(resumed.bank.off_values).all()


@pytest.mark.parametrize("pattern", ["banded", "random", "top_k"])
def test_legacy_baseline(pattern):
    from svft.svft_layers import LinearWithSVFT
    torch.manual_seed(4)
    layer = LinearWithSVFT(torch.nn.Linear(4, 4), off_diag=1, pattern=pattern)
    x = torch.randn(2, 4)
    y = layer(x)
    s = layer.svft_layer
    reference = torch.diag(s.s_pre).clone()
    reference.index_put_((s.s_row, s.s_col), s.s * torch.sigmoid(s.gate), accumulate=True)
    torch.testing.assert_close(y, torch.nn.functional.linear(x, s.u @ reference @ s.v, layer.bias))
    y.sum().backward()
    assert s.s.grad is not None
    assert s.u.grad is None
    torch.testing.assert_close(y, torch.nn.functional.linear(x, layer.merge_and_unload(), layer.bias))


@pytest.mark.parametrize("method", ["ta_svft", "svft"])
def test_complete_vision_entry_point(tmp_path, monkeypatch, method):
    """Exercise the real main(), replacing only external model/dataset downloads."""
    import json
    import sys
    import numpy as np
    from PIL import Image
    from datasets import Dataset
    from transformers import ViTImageProcessor
    import vision_experiments.finetuning_setup as entry
    images = [Image.fromarray(np.full((16, 16, 3), i * 20, dtype=np.uint8)) for i in range(6)]
    dataset = Dataset.from_dict({"image": images, "label": [0, 1, 2, 0, 1, 2]})
    monkeypatch.setattr(entry, "get_dataset", lambda _: (dataset, copy.deepcopy(dataset), copy.deepcopy(dataset)))
    monkeypatch.setattr(entry.AutoModelForImageClassification, "from_pretrained", lambda *a, **k: model())
    monkeypatch.setattr(entry.AutoImageProcessor, "from_pretrained", lambda *a, **k: ViTImageProcessor(size={"height": 16, "width": 16}))
    monkeypatch.setattr(sys, "argv", ["finetuning_setup.py", "--finetuning_method", method,
        "--output_dir", str(tmp_path / "run"), "--results_json", str(tmp_path / "results.json"),
        "--max_steps", "2", "--per_device_train_batch_size", "2", "--per_device_eval_batch_size", "2",
        "--report_to", "none", "--use_cpu", "true", "--disable_tqdm", "true",
        "--remove_unused_columns", "false", "--save_steps", "1",
        "--ta_off_budget", "20", "--ta_complement_rank", "2", "--ta_calibration_batches", "1",
        "--ta_update_interval", "1", "--ta_freeze_step", "3", "--ta_detailed_support", "true"])
    entry.main()
    result = json.loads((tmp_path / "results.json").read_text())[0]
    assert "eval_accuracy" in result
    if method == "ta_svft":
        assert result["adapter_parameters"] == 164
        assert (tmp_path / "run/ta_adapter.pt").exists()
        assert (tmp_path / "run/merged/model.safetensors").exists()
        assert (tmp_path / "run/ta_support.json").exists()


def test_dinov2_targets_and_forward():
    from transformers import Dinov2Config, Dinov2ForImageClassification
    m = Dinov2ForImageClassification(Dinov2Config(image_size=16, patch_size=8,
        hidden_size=12, num_hidden_layers=1, num_attention_heads=3, mlp_ratio=2, num_labels=3))
    targets = vit_targets(m)
    assert len(targets) == 6
    ctl = TASVFT(m, targets, TASVFTConfig(off_budget=5, complement_rank=1, calibration_batches=1))
    batch = {"pixel_values": torch.randn(2, 3, 16, 16), "labels": torch.tensor([0, 1])}
    ctl.select_support(lambda: iter([batch]), lambda m, b: m(**b).loss)
    m(**batch).loss.backward()
    ctl.validate_support()
