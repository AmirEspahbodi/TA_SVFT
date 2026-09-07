"""Single-process Hugging Face Trainer integration for TA-SVFT."""
from pathlib import Path
from dataclasses import asdict
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback


class SupportCallback(TrainerCallback):
    def __init__(self, controller, batch_factory, loss_fn, detailed=False):
        self.controller, self.batch_factory, self.loss_fn = controller, batch_factory, loss_fn
        self.detailed = detailed

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        config = self.controller.config
        if (config.update_interval and state.global_step % config.update_interval == 0
                and state.global_step < config.freeze_step):
            # Accelerate wraps the optimizer in single-process training too.
            raw_optimizer = getattr(optimizer, "optimizer", optimizer)
            self.controller.select_support(self.batch_factory, self.loss_fn,
                                           raw_optimizer, refine=True)
            if self.detailed:
                self.controller.export_report(Path(args.output_dir) / f"support-{state.global_step}.json")
        return control


class TASVFTTrainer(Trainer):
    """Stock optimizer/scheduler/checkpoint machinery, with support state in state_dict.

    Unsupported distributed/sharded modes fail before training: dynamically sized
    per-layer metadata cannot be safely broadcast by the stock DDP wrapper.
    """
    def __init__(self, *args, ta_controller, detailed_support=False, **kwargs):
        self.ta_controller = ta_controller
        super().__init__(*args, **kwargs)
        if (self.args.world_size != 1 or self.args.n_gpu > 1 or self.is_deepspeed_enabled
                or self.is_fsdp_enabled or self.args.gradient_checkpointing
                or self.args.torch_compile):
            raise ValueError("TA-SVFT Trainer supports one process/device, without sharding, compile or gradient checkpointing")
        def batch_factory():
            generator = torch.Generator().manual_seed(self.ta_controller.config.seed)
            return DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size,
                              shuffle=True, generator=generator, collate_fn=self.data_collator,
                              num_workers=0)
        def loss_fn(model, batch):
            inputs = self._prepare_inputs(batch)
            return model(**inputs).loss
        self.ta_batch_factory, self.ta_loss_fn = batch_factory, loss_fn
        self.add_callback(SupportCallback(ta_controller, batch_factory, loss_fn, detailed_support))

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        directory = Path(self._get_output_dir(trial)) / f"checkpoint-{self.state.global_step}"
        config = {"config": asdict(self.ta_controller.config),
                  "targets": self.ta_controller.targets}
        (directory / "ta_config.json").write_text(json.dumps(config, indent=2))
        # Stock 4.44 RNG checkpoints pickle numpy arrays, rejected by torch >= 2.6.
        # Store only primitive containers and torch tensors; never disable weights_only.
        np_state = np.random.get_state()
        rng = {"python": random.getstate(), "numpy_kind": np_state[0],
               "numpy_keys": torch.tensor(np_state[1].astype(np.int64)),
               "numpy_rest": np_state[2:], "cpu": torch.random.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.random.get_rng_state_all()
        torch.save(rng, directory / "ta_rng.pt")

    def _load_rng_state(self, checkpoint):
        path = Path(checkpoint) / "ta_rng.pt"
        if not path.exists():
            raise ValueError("TA-SVFT resume requires ta_rng.pt from TASVFTTrainer")
        rng = torch.load(path, map_location="cpu", weights_only=True)
        random.setstate(rng["python"])
        np.random.set_state((rng["numpy_kind"], rng["numpy_keys"].numpy().astype(np.uint32), *rng["numpy_rest"]))
        torch.random.set_rng_state(rng["cpu"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.random.set_rng_state_all(rng["cuda"])

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        path = Path(resume_from_checkpoint) / "ta_config.json"
        if not path.exists():
            raise ValueError("Missing TA-SVFT configuration in Trainer checkpoint")
        saved = json.loads(path.read_text())
        if saved != {"config": asdict(self.ta_controller.config), "targets": self.ta_controller.targets}:
            raise ValueError("Resume must use the saved TA-SVFT configuration and target paths")
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        self.ta_controller.validate_support()

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        # Resume restores basis/support through the normal state_dict path; no recalibration.
        if not resume_from_checkpoint and not bool(self.ta_controller.bank.initialized):
            self.ta_controller.select_support(self.ta_batch_factory, self.ta_loss_fn)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)
