wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

mkdir -p /workspace/dataset/
mv gpt2-vocab.json /workspace/dataset/
mv gpt2-merges.txt /workspace/dataset/
