bash run_glue.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 
bash run_bitfit.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0
bash run_random_baseline.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 
bash run_glue_sparse_update.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 0.005 1024