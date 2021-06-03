CUDA_VISIBLE_DEVICES=3 python main.py \
  --n_GPUs 1 --batch_size 4 --save_results  \
  --img_width 224 --img_height 224 \
  --save_models --normalization 0.5+0.5+0.5+1+1+1 \
  --epochs 300 \
  --data_train HDRPS+HDRPLUS \
  --print_every 100 \
  --data_test HDRPLUS \
  --model enhance_na \
  --loss 0.0002*color_diff_loss+0.01*HDR_loss+1*MSE \
  --reset \
  --save test_color
#  --pre_train ../self_preserve.pt \
