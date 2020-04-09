import os
import json

project_root = os.path.dirname(__file__)
with open(os.path.join(project_root, '../../configs/config.json'), 'r') as f:
    config = json.load(f)

MAIN_TEST_PATH = config["MAIN_TEST_PATH"]

RANDOM_PRIOR_PATH = os.path.join(project_root,  "../../data/augmented.prior")
SAS_MODEL_PATH = os.path.join(project_root,  "../../data/SA_score_prediction.pkl.gz")


ACTIVITY_REGRESSION = os.path.join(project_root, "../../data/B-RAF_model.pkl")
OFFTARGET_ACTIVITY_REGRESSION = os.path.join(project_root, "../../data/Aurora_model.pkl")
ACTIVITY_CLASSIFICATION = os.path.join(project_root,  "../../data/drd2.pkl")
SMILES_SET_PATH = os.path.join(project_root,  "../../data/smiles.smi")
