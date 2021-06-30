SEED=0
SPLIT=2
HEPOCH=3.5
GPU="0"

for TASK in "cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb"
do
    bash scripts/distributed_training.sh $TASK $SEED $SPLIT $HEPOCH $GPU
done