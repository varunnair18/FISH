# for num_samples in 256 1024
# do
#     for keep_ratio in 0.005 0.02 0.10
#     do
#         for epochs in 24 40 60 100
#         do
#             CUDA_VISIBLE_DEVICES="2" python su_dawn.py \
#                 --num_samples $num_samples \
#                 --keep_ratio $keep_ratio \
#                 --epochs $epochs \
#                 --fix_mask \
#                 --save_file su_s${num_samples}_k${keep_ratio}_e${epochs}_fix.tsv
#         done
#     done
# done

num_samples=1024

for keep_ratio in 0.005
do
    for epochs in 24 40 60 100
    do
        for lr in 0.8
        do
            CUDA_VISIBLE_DEVICES="0" python su_dawn.py \
                --num_samples $num_samples \
                --keep_ratio $keep_ratio \
                --epochs $epochs \
                --lr $lr \
                --save_file su_s${num_samples}_k${keep_ratio}_e${epochs}_l${lr}.tsv
        done
    done
done

# num_samples=1024
# keep_ratio=0.005
# epochs=100

# for warmup_epoch in 1 5 10 25
# do
#     CUDA_VISIBLE_DEVICES="2" python su_dawn.py \
#         --num_samples $num_samples \
#         --keep_ratio $keep_ratio \
#         --epochs $epochs \
#         --warmup_epoch $warmup_epoch \
#         --save_file su_s${num_samples}_k${keep_ratio}_e${epochs}_w${warmup_epoch}.tsv
# done

# num_samples=256
# # keep_ratio=0.10

# for keep_ratio in 0.005 0.02
# do
#     for epochs in 12 20 30 50
#     do
#         for merge_steps in 10 30 100
#         do
#             CUDA_VISIBLE_DEVICES="0" python asgd_dawn.py \
#                 --num_samples $num_samples \
#                 --keep_ratio $keep_ratio \
#                 --epochs $epochs \
#                 --merge_steps $merge_steps \
#                 --save_file asgd_m${merge_steps}_k${keep_ratio}_e${epochs}.tsv

#         done
#     done
# done
