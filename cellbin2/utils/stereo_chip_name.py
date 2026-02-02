import json
from .stereo_chip_name_c import decrypt_mask  

def load_chip_mask(path: str):

    if path.endswith('.enc'):
        with open(path, 'rb') as f:
            enc = f.read()
        json_text = decrypt_mask(enc)
        obj = json.loads(json_text)
        return obj
    else:
        with open(path, 'r') as f:
            return json.load(f)
