import time

import numpy as np

from reinvent_scoring.scoring.score_summary import FinalSummary
from reinvent_chemistry.logging import fraction_valid_smiles

from running_modes.automated_curriculum_learning.dto.timestep_dto import TimestepDTO


class ConsoleMessage:

    def create(self, report_dto: TimestepDTO):
        time_message = self._time_progress(report_dto)
        score_message = self._score_profile(report_dto)
        score_breakdown = self._score_summary_breakdown(report_dto.score_summary)
        message = time_message + score_message + score_breakdown
        return message

    def _time_progress(self, report_dto: TimestepDTO):
        mean_score = np.mean(report_dto.score_summary.total_score)
        time_elapsed = int(time.time() - report_dto.start_time)
        time_left = (time_elapsed * ((report_dto.n_steps - report_dto.step) / (report_dto.step + 1)))
        valid_fraction = fraction_valid_smiles(report_dto.score_summary.scored_smiles)
        message = (f"\n Step {report_dto.step}   Fraction valid SMILES: {valid_fraction:4.1f}   Score: {mean_score:.4f}   "
                   f"Sample size: {len(report_dto.score_summary.scored_smiles)}   "
                   f"Time elapsed: {time_elapsed}   "
                   f"Time left: {time_left:.1f}\n")
        return message

    def _score_profile(self, report_dto: TimestepDTO):
        # Convert to numpy arrays so that we can print them
        augmented_likelihood = report_dto.augmented_likelihood.data.cpu().numpy()
        agent_likelihood = report_dto.agent_likelihood.data.cpu().numpy()
        message = "     ".join(["  Agent", "Prior", "Target", "Score"] + ["SMILES\n"])
        for i in range(min(10, len(report_dto.score_summary.scored_smiles))):
            message += f'{agent_likelihood[i]:6.2f}    {report_dto.prior_likelihood[i]:6.2f}    ' \
                       f'{augmented_likelihood[i]:6.2f}    {report_dto.score_summary.total_score[i]:6.2f} '
            message += f"     {report_dto.score_summary.scored_smiles[i]}\n"
        return message

    def _score_summary_breakdown(self, score_summary: FinalSummary):
        message = "   ".join([c.name for c in score_summary.profile])
        message += "\n"
        for i in range(min(10, len(score_summary.scored_smiles))):
            for summary in score_summary.profile:
                message += f"{summary.score[i]}   "
            message += "\n"
        return message
