import os

metadata_path = os.path.join(r"/mnt/share_disk/dorin/AquaLoRA/train/Gustavosta_sample", r"metadata.jsonl")
print(f"Looking for metadata.jsonl at: {metadata_path}")
print(f"Does file exist? {os.path.exists(metadata_path)}")
print(f"Is path a file? {os.path.isfile(metadata_path)}")
print(f"Can file be read? {os.access(metadata_path, os.R_OK)}")

if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
