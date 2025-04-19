"""Configuration loader with recursive variable substitution."""

import os
import yaml
import re


def load_config():
    """
    Load configuration from config.yaml and resolve nested ${var} substitutions recursively.

    Returns:
        dict: Configuration dictionary with resolved paths.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_raw = yaml.safe_load(f)

    # Regular expression for ${var} placeholders
    pattern = re.compile(r'\$\{([^}^{]+)\}')

    # Flatten the config dictionary for environment-style resolution
    def flatten(d, parent_key='', sep='.'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten(v, new_key, sep=sep))
            else:
                items[new_key] = v
                items[k] = v  # allow shorthand reference too
        return items

    env_map = flatten(config_raw)

    def resolve(value):
        """Recursively resolve a string with ${} placeholders."""
        while isinstance(value, str) and pattern.search(value):
            for var in pattern.findall(value):
                if var not in env_map:
                    raise ValueError(f"Undefined variable: {var}")
                value = value.replace(f"${{{var}}}", str(env_map[var]))
        return value

    # Recursively resolve all string values
    def recursive_resolve(obj):
        if isinstance(obj, dict):
            return {k: recursive_resolve(resolve(v)) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_resolve(resolve(i)) for i in obj]
        elif isinstance(obj, str):
            return resolve(obj)
        else:
            return obj

    return recursive_resolve(config_raw)
