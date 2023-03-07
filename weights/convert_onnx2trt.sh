#!/bin/bash

# if $1 is empty
if [ -z "$1" ]; then
    echo "Usage: $0 <onnx model path> <workspace>"
    echo "ONNX-Model-Path : "
    echo "WORKSPACE : GPU memory workspace. Default 16."
    exit 1
fi

ONNX_MODEL_PATH=$1
TRT_WORKSPACE=$2
if [ -z "$2" ]; then
    TRT_WORKSPACE=16
fi
BASENAME=`echo $(basename ${ONNX_MODEL_PATH})`


SCRIPT_DIR=$(cd $(dirname $0); pwd)

echo "Model Path: ${ONNX_MODEL_PATH}"
echo "Workspace size: ${TRT_WORKSPACE}"
echo ""

if [ ! -e $ONNX_MODEL_PATH ]; then
    echo "[ERROR] Not Found ${ONNX_MODEL_PATH}"
    echo "[ERROR] Please check onnx model path."
    exit 1
fi

/usr/src/tensorrt/bin/trtexec \
    --onnx=$ONNX_MODEL_PATH \
    --saveEngine=$SCRIPT_DIR/$BASENAME.trt \
    --verbose --workspace=$((1<<$TRT_WORKSPACE))

