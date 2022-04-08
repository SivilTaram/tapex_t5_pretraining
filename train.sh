export EXP_NAME=tapex_t5_efficient_small_4M_3w_lr_3e5

python run_model.py \
  --do_train \
  --do_eval \
  --train_file /home/t-qianli/team_drive/t-qianli/tapex_t5_400w/train.json \
  --validation_file /home/t-qianli/team_drive/t-qianli/tapex_t5_400w/dev.json \
  --output_dir /home/t-qianli/team_drive/t-qianli/$EXP_NAME \
  --model_name_or_path google/t5-efficient-small-dm128 \
  --overwrite_output_dir \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 128 \
  --learning_rate 3e-5 \
  --logging_steps 10 \
  --eval_steps 2000 \
  --save_steps 2000 \
  --warmup_steps 1500 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 30000
