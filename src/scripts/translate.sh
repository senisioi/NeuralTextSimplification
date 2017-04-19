source ./base_conf.sh

RES_DIR=`readlink -f ../../results_{CUR_EXP}`
mkdir -p RES_DIR
MODEL=`readlink -f ../../models/NTS_epoch11_10.19.t7`
OUTPUT=${RES_DIR}/result_${MODEL}
LOG_OUT=${RES_DIR}/result_${MODEL}.log
GPUS=1,2
BEAM_SIZE=12
SRC=${DIRECTORY}/test.en
TGT=${DIRECTORY}/test.sen
cd $OpenNMT_PATH 
th translate.lua -replace_unk -beam_size ${BEAM_SIZE} -gpuid ${GPUS} -n_best 4 -model ${MODEL} -src ${SRC} -tgt ${TGT} -output ${OUTPUT} -log_file ${LOG_OUT}
cd $CWD
echo "Check results in "${OUTPUT}
