#docker build -t megatron -f ./Dockerfile .

docker_img="megatron"

megatron_deepspeed_path="/home/ubuntu/Megatron-DeepSpeed-TT"
llm_path="/home/ubuntu/Megatron-llm"

work_dir="/workspace/megatron_deepspeed/"

docker run --gpus all --rm -it -P \
    --cap-add SYS_NICE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $megatron_deepspeed_path:/workspace/megatron_deepspeed \
    -v $llm_path:/workspace/llm \
    -w /${work_dir} \
    ${docker_img}:latest \
    bash
    
