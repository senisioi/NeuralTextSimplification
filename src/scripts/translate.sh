#!/bin/bash
source ./base_conf.sh

RES_DIR=`readlink -f ../../results_${CUR_EXP}`
mkdir -p $RES_DIR
MODEL_PATH=`readlink -f ../../models/NTS_epoch11_10.19.t7`
MODEL=${MODEL_PATH##*/}

BEAM_SIZE=5
#GPUS=1,2
GPUS=0
OUTPUT=${RES_DIR}/result_${MODEL}_${BEAM_SIZE}
LOG_OUT=${RES_DIR}/result_${MODEL}_${BEAM_SIZE}.log

SRC=${DIRECTORY}/test.en
TGT=${DIRECTORY}/test.sen
cd $OPENNMT_PATH 
th translate.lua -replace_unk -beam_size ${BEAM_SIZE} -gpuid ${GPUS} -n_best 4 -model ${MODEL_PATH} -src ${SRC} -tgt ${TGT} -output ${OUTPUT} -log_file ${LOG_OUT}
cd $CWD
echo "Check results in "${OUTPUT}
