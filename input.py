#!/usr/bin/env python
#  coding=utf-8

import sys
import json
from running_modes.manager import Manager
# this import is needed for unpickling the models
from model_container import ModelContainer

def main():
    with open(sys.argv[1]) as f:
        json_input = f.read().replace('\r', '').replace('\n', '')

    configuration = {}
    try:
        configuration = json.loads(json_input)
    except (ValueError, KeyError, TypeError):
        print("JSON format error")
    else:
        manager = Manager(configuration)
        manager.run()


if __name__ == "__main__":
    main()