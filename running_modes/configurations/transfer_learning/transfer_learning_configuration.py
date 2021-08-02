from dataclasses import dataclass, field

from typing import List, Optional

from running_modes.configurations.transfer_learning.adaptive_learning_rate_configuration import AdaptiveLearningRateConfiguration


@dataclass
class TransferLearningConfiguration:
    input_model_path: str
    output_model_path: str
    input_smiles_path: str
    adaptive_lr_config: AdaptiveLearningRateConfiguration
    standardization_filters: List[dict] = field(default_factory=list)
    validation_smiles_path: Optional[str] = None
    save_every_n_epochs: int = 1
    batch_size: int = 128
    clip_gradient_norm: float = 1.0
    num_epochs: int = 10
    starting_epoch: int = 1
    shuffle_each_epoch: bool = True
    collect_stats_frequency: int = 1
    standardize: bool = True
    randomize: bool = False
    validate_model_vocabulary: bool = False
