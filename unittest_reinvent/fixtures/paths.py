import os
import json

project_root = os.path.dirname(__file__)
with open(os.path.join(project_root, '../../configs/config.json'), 'r') as f:
    config = json.load(f)

MAIN_TEST_PATH = config["MAIN_TEST_PATH"]

ACTIVITY_REGRESSION = config["ACTIVITY_REGRESSION"]
ACTIVITY_CLASSIFICATION = config["ACTIVITY_CLASSIFICATION"]
SMILES_SET_PATH = config["SMILES_SET_PATH"]
PRIOR_PATH = config["PRIOR_PATH"]
LIBINVENT_PRIOR_PATH = config["LIBINVENT_PRIOR_PATH"]
SMILES_SET_LINK_INVENT_PATH = config["SMILES_SET_LINK_INVENT_PATH"]
LINK_INVENT_PRIOR_PATH = config["LINK_INVENT_PRIOR_PATH"]