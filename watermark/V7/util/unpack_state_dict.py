import torch
import sys
sys.path.append('../../')
sys.path.append('../../../')
from utils.models import SecretEncoder, SecretDecoder
pretrain_dict = torch.load('/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/free_0101/4bit_state_dict_29.pth')
sec_encoder = SecretEncoder(secret_len=4, resolution=64)
sec_encoder.load_state_dict(pretrain_dict['sec_encoder'])
msgdecoder = SecretDecoder(output_size=4)
msgdecoder.load_state_dict(pretrain_dict['sec_decoder'])

torch.save(msgdecoder.state_dict(), '/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/free_0101/4bit_encoder_state_dict.pt')