rm -rf ./experiment/test/compressed
mkdir ./experiment/test/compressed
echo "Enter start epoch:"
read start
echo "Enter end epoch:"
read end
for i in $(seq -f "%03g" $start $end)
do
    zip -q ./experiment/test/compressed/$i.zip experiment/test/results-HDRPS/*-$i-*
done 
