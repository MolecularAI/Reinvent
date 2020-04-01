import os
import json

project_root = os.path.dirname(__file__)
with open(os.path.join(project_root, '../../configs/config.json'), 'r') as f:
    config = json.load(f)

MAIN_TEST_PATH = config["MAIN_TEST_PATH"]

RANDOM_PRIOR_PATH = os.path.join(project_root,  "../../unittest_reinvent/fixtures/test_data/augmented.prior")
SAS_MODEL_PATH = os.path.join(project_root,  "../../unittest_reinvent/fixtures/test_data/SA_score_prediction.pkl")


