#!/usr/bin/env python
#  coding=utf-8

import sys
import json
import argparse
from pathlib import Path
from running_modes.manager import Manager


DEFAULT_BASE_CONFIG_PATH = (Path(__file__).parent / 'configs/config.json').resolve()

parser = argparse.ArgumentParser(description='Run Reinvent.')
parser.add_argument(
    '--base_config', type=str, default=DEFAULT_BASE_CONFIG_PATH,
    help='Path to basic configuration for Reinvent environment.'
)
parser.add_argument(
    'run_config', type=str,
    help='Path to configuration json file for this run.'
)


def read_json_file(path):
    with open(path) as f:
        json_input = f.read().replace('\r', '').replace('\n', '')
    try:
        return json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")


if __name__ == "__main__":
    args = parser.parse_args()

    base_config = read_json_file(args.base_config)
    run_config = read_json_file(args.run_config)

    manager = Manager(base_config, run_config)
    manager.run()
