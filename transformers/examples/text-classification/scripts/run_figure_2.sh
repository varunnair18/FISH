bash run_glue_sparse_update.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 0.0002 1024
bash run_glue_sparse_update.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 0.001 1024
bash run_glue_sparse_update.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 0.005 1024
bash run_glue_sparse_update.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 0.021 1024

bash ablation_sample_size.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0