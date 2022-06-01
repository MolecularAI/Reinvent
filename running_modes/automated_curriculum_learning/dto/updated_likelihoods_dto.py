from dataclasses import dataclass
import torch


@dataclass
class UpdatedLikelihoodsDTO:
    agent_likelihood: torch.Tensor
    prior_likelihood: torch.Tensor
    augmented_likelihood: torch.Tensor
    loss: torch.Tensor