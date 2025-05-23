import sys
sys.path.append("../")
import torch
import safetensors
from safetensors.torch import load_file
import os
from utils.models import MapperNet

def create_watermark_lora(train_folder, scale, msg_bits=48, hidinfo=None, save=True,output_size=None):
    lora_state_dict = load_file(f"{train_folder}/pytorch_lora_weights.safetensors",device='cpu')

    if hidinfo is None:
        hidinfo = torch.randint(0, 2, (1, msg_bits))
    else:
        assert len(hidinfo) == msg_bits
        hidinfo = torch.tensor([int(i) for i in hidinfo]).unsqueeze(0)
    hidinfo_ = hidinfo.float()

    if output_size is None:
        output_size = msg_bits

    mapper = MapperNet(input_size=msg_bits, output_size=output_size)
    mapper.load_state_dict(torch.load(f"{train_folder}/mapper.pt"))
    if args.use_sampleMsgVector:
        if args.sampleMsgVector_path is None:
            mapped_loradiag = torch.load(f"{train_folder}/tensor_b_with_grad.pth")
        else:
            mapped_loradiag = torch.load(args.sampleMsgVector_path)

    else:
                mapped_loradiag = mapper(hidinfo_)


    c_lora_state_dict = dict()
    for key in lora_state_dict:
        if 'unet' in key:
            if 'attn' in key or 'ff' in key:
                if 'up.weight' in key:
                    c_lora_state_dict[key] = lora_state_dict[key]
                elif 'down.weight' in key:
                    mid = torch.diag_embed(mapped_loradiag)[0]
                    c_lora_state_dict[key] = mid @ lora_state_dict[key] * scale
            if 'proj_in' in key or 'proj_out' in key:
                if 'up.weight' in key:
                    c_lora_state_dict[key] = lora_state_dict[key]
                elif 'down.weight' in key:
                    mid = mapped_loradiag[0]
                    c_lora_state_dict[key] = lora_state_dict[key] * mid[:, None, None, None] * scale
        elif 'text_encoder' in key:
            pass
        else:
            raise ValueError(f"key {key} not found")

    hidinfo = ''.join(map(str, hidinfo.tolist()[0]))

    # save c_lora_state_dict

    if args.use_sampleMsgVector:
        if args.sampleMsgVector_path is None:
            if not os.path.exists(f"{train_folder}/sample_{hidinfo}"):
                os.makedirs(f"{train_folder}/sample_{hidinfo}")
            safetensors.torch.save_file(c_lora_state_dict, f"{train_folder}/sample_{hidinfo}/pytorch_lora_weights.safetensors")
    else:
        if not os.path.exists(f"{train_folder}/{hidinfo}"):
            os.makedirs(f"{train_folder}/{hidinfo}")
        print(f"{train_folder}/{hidinfo}/pytorch_lora_weights.safetensors")
        safetensors.torch.save_file(c_lora_state_dict, f"{train_folder}/{hidinfo}/pytorch_lora_weights.safetensors")

    return hidinfo, c_lora_state_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str, required=True)
    parser.add_argument("--msg_bits", type=int, default=48)
    parser.add_argument("--scale", type=float, default=1.03)
    parser.add_argument("--hidinfo", type=str, default=None, help="your secret message, if None, it will be randomly generated")
    parser.add_argument("--use_sampleMsgVector",action="store_true")
    parser.add_argument("--sampleMsgVector_path", type=str, default=None)
    parser.add_argument("--output_size", type=int,default=None)
    args = parser.parse_args()
    output_size = args.output_size
    if output_size is None:
        output_size = args.msg_bits


    hidinfo, _ = create_watermark_lora(args.train_folder, args.scale, args.msg_bits, args.hidinfo,output_size=args.output_size)
    print(hidinfo)
