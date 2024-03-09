# docker build -t megatron -f ./Dockerfile .
# docker tag megatron:latest keliu354/megatron:v5
# docker push keliu354/megatron:v5

docker_img="megatron:latest"

megatron_deepspeed_path="/home/ubuntu/Megatron-DeepSpeed-TT"
datasets_path="/home/ubuntu/datasets-llm"

work_dir="/workspace/megatron_deepspeed/"

docker run --gpus all --rm -it -P \
    --cap-add SYS_NICE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $megatron_deepspeed_path:/workspace/megatron_deepspeed \
    -v $datasets_path:/workspace/datasets \ 
    -w /${work_dir} \
    ${docker_img} \
    bash
