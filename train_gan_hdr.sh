CUDA_VISIBLE_DEVICES=0 python main.py \
  --n_GPUs 1 --batch_size 4 --save_results  \
  --img_width 224 --img_height 224 \
  --save_models --normalization 0.5+0.5+0.5+1+1+1 \
  --epochs 300 \
  --data_train HDRPS+HDRPLUS \
  --print_every 100 \
  --data_test HDRPS+HDRPLUS+Huawei \
  --model enhance \
  --loss 20*self_preserve_res_loss+0.0002*color_diff_loss+0.01*HDR_loss+0.2*GAN+1*MSE \
  --reset \
  --save test_gan
#  --pre_train ../self_preserve.pt \