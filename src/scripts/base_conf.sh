#!/bin/bash
CWD=`pwd`
CUR_EXP=NTS
DIRECTORY=`readlink -f ../../data`
OPENNMT_PATH=`readlink -f ../../OpenNMT`

if [[ -z "${OPENNMT_PATH}" ]]; then 
  echo "OPENNMT_PATH is unset, please update base_conf.sh"; 
fi 

DATA_DIRECTORY=${DIRECTORY}
DATA_OUT=${CUR_EXP}
MODEL_DIRECTORY=${DIRECTORY}/models/${CUR_EXP}
mkdir -p ${MODEL_DIRECTORY}