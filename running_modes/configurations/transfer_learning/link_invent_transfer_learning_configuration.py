from dataclasses import dataclass
from typing import Optional

from running_modes.configurations.transfer_learning.link_invent_learning_rate_configuration import \
    LinkInventLearningRateConfiguration


@dataclass
class LinkInventTransferLearningConfiguration:
    empty_model: str
    learning_rate: LinkInventLearningRateConfiguration
    output_path: str
    input_smiles_path: str
    sample_size: int
    batch_size: int = 128
    starting_epoch: int = 1
    num_epochs: int = 10
    clip_gradient_norm: float = 10
    collect_stats_frequency: int = 1
    save_model_frequency: int = 1
    validation_smiles_path: Optional[str] = None
    with_weights: bool = False
    model_file_name = 'trained.model'
