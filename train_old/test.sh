sleep 1h

python trainSpeakerNet_alda.py --model X_vector --encoder_type ASP --trainfunc amsoftmax --initial_model /workspace/LOGS_OUTPUT/server9_nvme1/ASV_LOGS_202102/train_logs_201120/Xvecor_alda_0.0/model/model000000123.model --mixedprec
