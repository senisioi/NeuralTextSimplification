#!/bin/bash
source ./base_conf.sh

RES_DIR=`readlink -f ../../results_${CUR_EXP}`
mkdir -p $RES_DIR
MODEL_PATH=`readlink -f /media/er/Data/simplification/paper/acl207/neural_TS/release_models/NTS_epoch11_10.19_release.t7`
MODEL=${MODEL_PATH##*/}
OUTPUT=${RES_DIR}/result_${MODEL}
LOG_OUT=${RES_DIR}/result_${MODEL}.log
#GPUS=1,2
GPUS=0
BEAM_SIZE=12
SRC=${DIRECTORY}/test.en
TGT=${DIRECTORY}/test.sen
cd $OPENNMT_PATH 
th translate.lua -replace_unk -beam_size ${BEAM_SIZE} -gpuid ${GPUS} -n_best 4 -model ${MODEL_PATH} -src ${SRC} -tgt ${TGT} -output ${OUTPUT} -log_file ${LOG_OUT}
cd $CWD
echo "Check results in "${OUTPUT}
