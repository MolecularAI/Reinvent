from typing import List, Any, Dict
import pandas as pd

from running_modes.dto.scoring_table_entry_dto import ScoringTableEntryDTO
from running_modes.enums.scoring_table_enum import ScoringTableEnum


class ScoringTable:
    def __init__(self):
        self._scoring_table_enum = ScoringTableEnum()
        self.scoring_table = pd.DataFrame(columns=[self._scoring_table_enum.AGENTS,
                                                   self._scoring_table_enum.SCORES,
                                                   self._scoring_table_enum.SCORING_FUNCTIONS,
                                                   self._scoring_table_enum.COMPONENT_NAMES])
        self.constant_component_table = pd.DataFrame(columns=[self._scoring_table_enum.SCORING_FUNCTIONS,
                                                              self._scoring_table_enum.COMPONENT_NAMES])

    def add_score_for_agent(self, entry: ScoringTableEntryDTO):
        new_row = pd.DataFrame(data={self._scoring_table_enum.AGENTS: entry.agent,
                                     self._scoring_table_enum.SCORES: entry.score,
                                     self._scoring_table_enum.SCORING_FUNCTIONS: [entry.scoring_function_components],
                                     self._scoring_table_enum.COMPONENT_NAMES:
                                         entry.scoring_function_components.get('name', "unknown_name")})

        self.scoring_table = self.scoring_table.append(new_row, ignore_index=True)

    def add_constant_component(self, entry: Dict):
        new_row = pd.DataFrame(data={self._scoring_table_enum.SCORING_FUNCTIONS: [entry],
                                     self._scoring_table_enum.COMPONENT_NAMES:
                                         entry.get('name', "unknown_name")})

        self.constant_component_table = self.constant_component_table.append(new_row, ignore_index=True)

    def rank_by_score(self) -> pd.DataFrame:
        grouped_scoring_table = self.scoring_table\
            .groupby(self._scoring_table_enum.SCORES)\
            .agg({self._scoring_table_enum.SCORING_FUNCTIONS: list,
                  self._scoring_table_enum.AGENTS: lambda x: list(x)[0],
                  self._scoring_table_enum.COMPONENT_NAMES: list})\
            .reset_index()
        return grouped_scoring_table.sort_values(self._scoring_table_enum.SCORES,
                                                 ascending=False).reset_index(drop=True)

    def get_top_sf_components(self, number: int = -1) -> List:
        table = self.rank_by_score().head(number)
        components = table[self._scoring_table_enum.SCORING_FUNCTIONS].tolist()
        return sum(components, [])  # flattening the list in case of tied ranks

    def get_top_agent(self) -> Any:
        table = self.rank_by_score()
        agent = table.loc[0][self._scoring_table_enum.AGENTS]
        return agent

    def get_sf_components_by_name(self, names: List[str]) -> List:
        df = pd.concat([self.scoring_table, self.constant_component_table], axis=0, sort=False)
        components_df = [df[df[self._scoring_table_enum.COMPONENT_NAMES] == name] for name in names]
        components = [component[self._scoring_table_enum.SCORING_FUNCTIONS].item() for component in components_df]
        return components

    def get_sf_components_by_rank(self, rank: int=0) -> List:
        components = self.rank_by_score().loc[rank][self._scoring_table_enum.SCORING_FUNCTIONS]
        return components
