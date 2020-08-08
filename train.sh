CUDA_VISIBLE_DEVICES=1,2 python main.py \
  --n_GPUs 2 --batch_size 4 --save_results  \
  --img_width 224 --img_height 224 \
  --save_models --normalization 0.5+0.5+0.5+1+1+1 \
  --epochs 600 \
  --data_train HDRPS+HDRPLUS \
  --print_every 100 \
  --data_test HDRPLUS+HDRPS+Huawei \
  --loss 20*self_preserve_res_loss+0.0002*color_diff_loss+0.01*HDR_loss+0.2*GAN+1*MSE \
  --pre_train ../self_preserve.pt \