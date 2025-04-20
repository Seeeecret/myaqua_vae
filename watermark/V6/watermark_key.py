import hashlib
import json


class WatermarkKeyManager:
    @staticmethod
    def generate(config, msg_bits, save_path):
        # 生成唯一指纹
        model_fingerprint = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:8]

        key = {
            "version": "1.0",
            "config": config,
            "message": msg_bits,
            "mapping": [
                {"bit_idx": i, "channel": ch, "freq": [u, v]}
                for i, (ch, u, v) in enumerate(
                    [(c, u, v)
                     for c in range(config['dct_channels'])
                     for (u, v) in config['freq_bands']]
                )  # 这里添加了闭合括号
            ],  # 列表推导式结束
            "fingerprint": model_fingerprint  # 移到字典的顶层
        }

        with open(save_path, 'w') as f:
            json.dump(key, f, indent=2)
        return key

    @staticmethod
    def load(key_path):
        with open(key_path) as f:
            key = json.load(f)
        key['config']['freq_bands'] = [tuple(b) for b in key['config']['freq_bands']]
        return key

    @staticmethod
    def validate(watermarked_image, key):
        # 实现校验逻辑
        pass
