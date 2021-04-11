from typing import Union

import numpy as np
import statistics
import torch


class MarginGuard:

    def __init__(self, runner, margin_window=10, desirable_min_score=0.15):
        self.runner = runner
        self.margin_window = margin_window
        self._desirable_min_score = desirable_min_score
        self._run_stats = []

    def store_run_stats(self, agent_likelihood: torch.Tensor, prior_likelihood: torch.Tensor,
                        augmented_likelihood: torch.Tensor, score: np.array):
        self._run_stats.append({
            "agent_likelihood": agent_likelihood.detach().mean().item(),
            "prior_likelihood": prior_likelihood.detach().mean().item(),
            "augmented_likelihood": augmented_likelihood.detach().mean().item(),
            "score": np.mean(score).item()
        })

    def adjust_margin(self, step: int) -> None:
        if step == self.margin_window:
            if len(self._run_stats) < self.margin_window:
                raise Exception(f"self._run_stats has {len(self._run_stats)} elements. Consider storing all stats!")

            if self._is_margin_below_threshold():
                self.runner.config.sigma = self._increased_sigma()
                self._reset()

    def _reset(self):
        # self._run_stats = []
        self.runner.reset()

    def _increased_sigma(self) -> float:
        agent_likelihood = self._get_mean_stats_field("agent_likelihood")
        prior_likelihood = self._get_mean_stats_field("prior_likelihood")
        score = self._get_mean_stats_field("score")

        delta = agent_likelihood - prior_likelihood
        score = max(score, self._desirable_min_score)
        new_sigma = delta / score

        max_sigma = max(self.runner.config.sigma, new_sigma)
        max_sigma += self.runner.config.margin_threshold
        return max_sigma

    def _is_margin_below_threshold(self) -> bool:
        augmented_likelihood = self._get_mean_stats_field("augmented_likelihood")
        agent_likelihood = self._get_mean_stats_field("agent_likelihood")
        margin = augmented_likelihood - agent_likelihood
        return self.runner.config.margin_threshold > margin

    def _get_mean_stats_field(self, field: str) -> float:
        sliced = self._run_stats[:self.margin_window]
        target_fields = [s[field] for s in sliced]
        mean_data = statistics.mean(target_fields)
        return mean_data

    @torch.no_grad()
    def get_distance_to_prior(self, prior_likelihood: Union[torch.Tensor, np.ndarray],
                              distance_threshold=-100.) -> np.ndarray:
        """prior_likelihood and distance_threshold have negative values"""
        if type(prior_likelihood) == torch.Tensor:
            ones = torch.ones_like(prior_likelihood, requires_grad=False)
            mask = torch.where(prior_likelihood > distance_threshold, ones, distance_threshold / prior_likelihood)
            mask = mask.cpu().numpy()
        else:
            ones = np.ones_like(prior_likelihood)
            mask = np.where(prior_likelihood > distance_threshold, ones, distance_threshold / prior_likelihood)
        return mask
