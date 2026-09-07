#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python "$project_dir/vision_experiments/finetuning_setup.py" \
  --finetuning_method ta_svft --model_name vit-base --dataset_name cifar100 \
  --ta_families q k v o up down --ta_off_budget 10000 \
  --ta_complement_rank 4 --ta_calibration_batches 8 \
  --ta_update_interval 0 --ta_detailed_support true \
  --clf_learning_rate 0.001 --other_learning_rate 0.01 \
  --num_train_epochs 10 --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 \
  --evaluation_strategy epoch --save_strategy epoch --load_best_model_at_end true \
  --metric_for_best_model eval_accuracy --save_total_limit 2 \
  --remove_unused_columns false --warmup_ratio 0.1 --weight_decay 0.01 \
  --seed 42 --report_to none --output_dir results/ta-svft --results_json results-ta-svft.json "$@"
