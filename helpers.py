from typing import Dict, Any, Optional  
import yaml
import json
import base64
import re


def read_config(config_path =  'config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        return config
    
def save_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8")as fp:
        json.dump(data, fp)


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8")as fp:
        return json.load(fp)
    

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def generator_simulator(content: str):
    for token in content.split(" "):
        yield token + " "


def decode_and_reverse_password(encoded_password):
    decoded_bytes = base64.b64decode(encoded_password)
    decoded_password = decoded_bytes.decode('utf-8')
    reversed_password = decoded_password[::-1]
    return reversed_password


@staticmethod
def extract_json_content(response: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON content from LLM response."""
    # Try to find content within code blocks first
    code_block_pattern = r'\`\`\`(?:json|yml)?\n([\s\S]*?)\n?\`\`\`'
    code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
    try:
        if code_blocks:
            parsed_json = json.loads(code_blocks[0])
        else:
            # If no code blocks found, try to parse the entire response as YAML
            parsed_json = json.loads(response)
        # If the parsed result is a list, merge all dictionaries in the list
        if isinstance(parsed_json, list):
            return parsed_json
        return parsed_json if isinstance(parsed_json, dict) else None

    except yaml.YAMLError:
        return None


if __name__ == "__main__":
    content= """```json\n[1, 2, 3]```"""
    e = extract_json_content(content)
    print(e)
    print(type(e))  