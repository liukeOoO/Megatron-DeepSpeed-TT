import torch
import json


def load_and_print_hf_weight(hf_ckpt_dir, hf_ckpt_num_of_shards):
    # Optimization point: We can selectively load specific 'shared' data to reduce CPU memory usage.
    loaded = {}
    for wid in range(1, hf_ckpt_num_of_shards + 1):
        d = torch.load(
            f"{hf_ckpt_dir}/pytorch_model-{wid:05d}-of-{hf_ckpt_num_of_shards:05d}.bin",
            map_location=torch.device('cpu'))
        for k in d:
            assert k not in loaded
            #loaded[k] = d[k].clone()
            loaded[k] = list(d[k].size())
            print(k, loaded[k])
    del d
    return loaded


if __name__ == "__main__":
    model_name, n_shards = "llama-7b", 2
    #model_name, n_shards = "falcon-40b", 9
    hf_ckpt_dir = f"/workspace/models/{model_name}"
    hf_ckpt_num_of_shards = n_shards
    print(hf_ckpt_dir, hf_ckpt_num_of_shards)

    model = load_and_print_hf_weight(hf_ckpt_dir, hf_ckpt_num_of_shards)
    with open(f"{model_name}.json", 'w') as f:
        json.dump(model, f)