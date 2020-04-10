import os

from .data import REINVENT_DATA_DIRECTORY

reinforcement_learning_demo_config = {
    "logging": {
        "job_id": "demo",
        "job_name": "Reinforcement learning demo",
        "logging_frequency": 10,
        "logging_path": os.path.expanduser("~/Desktop/progress.log"),
        "recipient": "local",
        "resultdir": os.path.expanduser("~/Desktop/results"),
        # "sender": "http://127.0.0.1"
    },
    "parameters": {
        "diversity_filter": {
            "minscore": 0.4,
            "minsimilarity": 0.4,
            "name": "IdenticalMurckoScaffold",
            "nbmax": 25
        },
        "inception": {
            "memory_size": 100,
            "sample_size": 10,
            "smiles": []
        },
        "reinforcement_learning": {
            "agent": os.path.join(REINVENT_DATA_DIRECTORY, "augmented.prior"),
            "batch_size": 128,
            "learning_rate": 0.0001,
            "margin_threshold": 50,
            "n_steps": 125,
            "prior": os.path.join(REINVENT_DATA_DIRECTORY, "augmented.prior"),
            "reset": 0,
            "reset_score_cutoff": 0.5,
            "sigma": 128
        },
        "scoring_function": {
            "name": "custom_product",
            "parallel": False,
            "parameters": [
                {
                    "component_type": "predictive_property",
                    "model_path": os.path.join(REINVENT_DATA_DIRECTORY, "Aurora_model.pkl"),
                    "name": "Regression model",
                    "smiles": [],
                    "specific_parameters": {
                        "descriptor_type": "ecfp_counts",
                        "high": 9,
                        "k": 0.25,
                        "low": 4,
                        "radius": 3,
                        "scikit": "regression",
                        "size": 2048,
                        "transformation": True,
                        "transformation_type": "sigmoid",
                        "use_counts": True,
                        "use_features": True
                    },
                    "weight": 2
                },
                {
                    "component_type": "matching_substructure",
                    "model_path": None,
                    "name": "Matching substructure",
                    "smiles": [
                        "c1ccccc1CC"
                    ],
                    "specific_parameters": None,
                    "weight": 1
                },
                {
                    "component_type": "custom_alerts",
                    "model_path": None,
                    "name": "Custom alerts",
                    "smiles": [
                        "[*;r8]",
                        "[*;r9]",
                        "[*;r10]",
                        "[*;r11]",
                        "[*;r12]",
                        "[*;r13]",
                        "[*;r14]",
                        "[*;r15]",
                        "[*;r16]",
                        "[*;r17]",
                        "[#8][#8]",
                        "[#6;+]",
                        "[#16][#16]",
                        "[#7;!n][S;!$(S(=O)=O)]",
                        "[#7;!n][#7;!n]",
                        "C#C",
                        "C(=[O,S])[O,S]",
                        "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                        "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                        "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                        "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                        "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                        "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
                    ],
                    "specific_parameters": None,
                    "weight": 1
                },
                {
                    "component_type": "qed_score",
                    "model_path": None,
                    "name": "QED Score",
                    "smiles": [],
                    "specific_parameters": None,
                    "weight": 1
                }
            ]
        }
    },
    "run_type": "reinforcement_learning",
    "version": 2
}
