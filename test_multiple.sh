echo "Enter start epoch:"
read start
echo "Enter end epoch:"
read end
echo $start $end > test_only_epoch.txt
for i in $(seq $start $end)
do
    CUDA_VISIBLE_DEVICES=1 python main.py \
      --test_only \
      --n_GPUs 1 --batch_size 1 --save_results  \
      --img_width 224 --img_height 224 \
      --save_models --normalization 0.5+0.5+0.5+1+1+1 \
      --epochs 300 \
      --data_train HDRPS+HDRPS+HDRPLUS \
      --print_every 25 \
      --data_test Huawei+HDRPLUS+Enlighten \
      --loss 1*MSE \
      --pre_train ../experiment/test/model/model_$i.pt \
done








