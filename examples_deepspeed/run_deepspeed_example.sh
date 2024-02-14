#!/bin/bash
set -ex

VOCAB="/workspace/dataset/gpt2-vocab.json"
MERGE="/workspace/dataset/gpt2-merges.txt"
PreProcessedCorpus="/workspace/dataset/oscar-gpt2_text_document"


TP=1
PP=1
NLAYERS=24
HIDDEN=512

SEQ_LENGTH=20
GLOBAL_BATCH=64
MICRO_BATCH=8

DS_CONFIG="examples_deepspeed/ds_config.json"
ZERO_STAGE=0
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

OUTPUT_DIR=ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
#OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR

export NCCL_DEBUG=warn 

ds_args=""
ds_args=" --deepspeed ${ds_args}"
#ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


train_iters=5
eval_iters=1
profile_step_start=3
profile_step_end=3
profile_trace_path="/workspace/megatron_deepspeed/astrasim"
rm -fr ${profile_trace_path}
mkdir -p ${profile_trace_path}/et/
mkdir -p ${profile_trace_path}/kineto/

nsys="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report -f true --capture-range=cudaProfilerApi --cudabacktrace=true --osrt-threshold=10000 -x true"
#$nsys \
deepspeed pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads 16 \
    --seq-length $SEQ_LENGTH \
    --loss-scale 12 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters $train_iters \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters $eval_iters \
    --eval-interval 1000 \
    --data-path $PreProcessedCorpus \
    --vocab-file $VOCAB \
    --merge-file $MERGE \
    --save-interval 1000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --fp16 \
    --checkpoint-activations \
    --tensorboard-dir $OUTPUT_DIR \
    --profile \
    --profile-step-start $profile_step_start \
    --profile-step-end $profile_step_end \
    --profile-trace-path $profile_trace_path \
    $ds_args \
    --exit-interval 5000 | tee ${OUTPUT_DIR}/output.log

