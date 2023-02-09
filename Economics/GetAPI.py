import json
import os.path

def get_keys(file_path):
    expanded_path = os.path.expanduser(file_path)
    with open(expanded_path) as f:
        return json.load(f)