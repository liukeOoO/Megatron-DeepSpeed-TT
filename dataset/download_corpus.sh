wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
xz -d oscar-1GB.jsonl.xz

#mv oscar-1GB.jsonl.xz /workspace/llm/dataset/
mv oscar-1GB.jsonl /workspace/llm/dataset/
