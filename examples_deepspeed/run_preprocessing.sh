corpus_f="/workspace/dataset/oscar-1GB.jsonl"
vocab_f="/workspace/dataset/gpt2-vocab.json"
merge_f="/workspace/dataset/gpt2-merges.txt"
output_prefix="oscar-gpt2"
n_workers=8

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python tools/preprocess_data.py \
       --input ${corpus_f} \
       --output-prefix ${output_prefix} \
       --vocab-file ${vocab_f} \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file ${merge_f} \
       --append-eod \
       --workers $n_workers

mv oscar-gpt2_text_document.bin /workspace/dataset/
mv oscar-gpt2_text_document.idx /workspace/dataset/
