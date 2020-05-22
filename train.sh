CUDA_VISIBLE_DEVICES=1,3 python main.py \
  --n_GPUs 2 --batch_size 4 --save_results  \
  --img_width 224 --img_height 224 \
  --save_models --normalization 0.5+0.5+0.5+1+1+1 \
  --epochs 10 \
  --data_train DINFO \
  --data_test DINFO \
  --loss 0.5*RGAN+0.5*MSE \
  --reset