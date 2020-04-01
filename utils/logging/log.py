import json
import os


def _is_development_environment() -> bool:
    try:
        project_root = os.path.dirname(__file__)
        with open(os.path.join(project_root, '../../configs/config.json'), 'r') as f:
            config = json.load(f)
        is_dev = config["DEVELOPMENT_ENVIRONMENT"]
        return is_dev
    except:
        return False
