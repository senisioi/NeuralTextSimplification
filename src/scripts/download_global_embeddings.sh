#!/bin/bash
source ./base_conf.sh
EMBED_DIR=${DIRECTORY}/embed/
mkdir -p $EMBED_DIR
cd $EMBED_DIR
wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz && gunzip GoogleNews-vectors-negative300.bin.gz
