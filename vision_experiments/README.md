# Singular Value Fine-Tuning


## Vision Baselines

Example command to run ViT baseline:

```bash
python finetuning_setup.py --evaluation_strategy epoch --save_strategy epoch --gradient_accumulation_steps 1 --logging_steps 10 --load_best_model_at_end True --save_total_limit 2 --metric_for_best_model eval_accuracy --label_names labels --remove_unused_columns False --per_device_train_batch_size 64 --per_device_eval_batch_size 256 --seed 42 --num_train_epochs 10   --output_dir ./results/vit-base/cifar100/svft_/seed_42 --model_name vit-base --finetuning_method svft --dataset_name cifar100 --clf_learning_rate 4e-3 --other_learning_rate 5e-2 --warmup_ratio 0.1 --weight_decay 0.01
```


## Task-Adaptive SVFT

Use `bash vision_experiments/run_ta_svft.sh` from the repository root. See
[the complete TA-SVFT guide](../docs/TA_SVFT.md) for global budgets, module families,
calibration, support refinement, checkpoints and inference merging.
