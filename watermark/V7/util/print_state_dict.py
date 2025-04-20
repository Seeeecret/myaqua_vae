
import torch

state_dict_path = '/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/free_0101/4bit_encoder_state_dict.pt'
state_dict_path2 = '/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/msgdecoder.pt'
state_dict_path3 = '/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/free_0101/4bit_state_dict_29.pth'
state_dict = torch.load(state_dict_path)

print(state_dict)