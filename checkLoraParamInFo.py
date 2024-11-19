import torch
from safetensors.torch import load_file

def main():
    original_lora_param_info = {}
    original_lora_param_info2 = {}

    safetensors_data_path = "/mnt/share_disk/dorin/AquaLoRA/train/output_from_beginning/pytorch_lora_weights.safetensors"
    safetensors_data_path2 = "/mnt/share_disk/dorin/AquaLoRA/train/output_from_beginning/steploras/lora_weights_step_18602.safetensors"

    # original_lora_data_lengths = []
    model = load_file(safetensors_data_path)
    model2 = load_file(safetensors_data_path2)
    for key, value in model.items():
        param_info = {
            'shape': value.shape,
            'length': value.numel()
        }
        original_lora_param_info[key] = param_info
    for key, value in model2.items():
        param_info = {
            'shape': value.shape,
            'length': value.numel()
        }
        original_lora_param_info2[key] = param_info

    for key, value in zip(original_lora_param_info.items(), original_lora_param_info2.items()):
        print("weight 1:{key}".format(key=key))
        print("weight 2:{key}".format(key=value))



    # filename = "./checkpoints/pytorch_lora_weights.safetensors"
    # model = load_file(safetensors_data_path)
    # state_dict = model
    # for key, value in model.items():
        # print(key, value)

if __name__ == "__main__":
    main()