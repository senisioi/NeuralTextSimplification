#!/bin/bash
source ./base_conf.sh

if [[ -z "${GLOBAL_EMBED}" ]]; then 
  GLOBAL_EMBED=${DIRECTORY}/embed/GoogleNews-vectors-negative300.bin
  echo "GLOBAL_EMBED path is not set. Maybe run download_global_embeddings.sh first?";
  echo "The default is: "${GLOBAL_EMBED}
fi 

mkdir -p ${DIRECTORY}/embed
cd ${DIRECTORY} && cat *.en > ${DIRECTORY}/embed/corpus.en && cat *.sen > ${DIRECTORY}/embed/corpus.sen && cd ${CWD}/..
python train_word2vec.py ${DIRECTORY}/embed/corpus.en 200 && python train_word2vec.py ${DIRECTORY}/embed/corpus.en 200 && cd $OPENNMT_PATH && th tools/concat_embedding.lua -dict_file ${DIRECTORY}/${CUR_EXP}.src.dict -global_embed ${GLOBAL_EMBED} -local_embed ${DIRECTORY}/embed/corpus.sen.bin -save_data ${DIRECTORY}/embed/sen-embeddings-${CUR_EXP} && th tools/concat_embedding.lua -dict_file ${DIRECTORY}/${CUR_EXP}.tgt.dict -global_embed ${GLOBAL_EMBED} -local_embed ${DIRECTORY}/embed/corpus.en.bin -save_data ${DIRECTORY}/embed/en-embeddings-${CUR_EXP}

cd $CWD