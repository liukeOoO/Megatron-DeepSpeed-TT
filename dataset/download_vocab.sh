wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

mv gpt2-vocab.json /workspace/llm/dataset/
mv gpt2-merges.txt /workspace/llm/dataset/
