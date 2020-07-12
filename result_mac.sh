#!/bin/zsh
scp -r -P 24617 freddieonfire.ga:/home/shupeizhang/compressed ./
vared -p 'Start Epoch Num: ' -c start
vared -p 'End Epoch Num: ' -c end
for i in $(seq -f "%03g" $start $end)
do
    unzip -q ./compressed/$i.zip
    mv experiment $i
    rm ./compressed/$i.zip
done
rm -rf compressed

