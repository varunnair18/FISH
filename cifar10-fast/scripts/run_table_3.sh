num_samples=256
split=2
epochs=50

DIR=cifar10-fast

# FISH
for keep_ratio in 0.005 0.02 0.1
do
    for merge_steps in 100 500 2500
    do
        for lr in 0.4 0.2 0.08 0.04 0.02
        do
            bash ${DIR}/scripts/distributed_training_fish.sh $num_samples $keep_ratio $epochs $merge_steps $lr $split
        done
    done
done

# Random mask
for keep_ratio in 0.005 0.02 0.1
do
    for merge_steps in 100 500 2500
    do
        for lr in 0.4 0.2 0.08 0.04 0.02
        do
            bash ${DIR}/scripts/distributed_training_random.sh $keep_ratio $epochs $merge_steps $lr $split
        done
    done
done

# Densely-updated
for merge_steps in 100 500 2500
do
    for lr in 0.4 0.2 0.08 0.04 0.02
    do
        bash ${DIR}/scripts/distributed_training_dense.sh $epochs $merge_steps $lr $split
    done
done


