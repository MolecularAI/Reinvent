import pandas as pd

from ..configurations.reinforcement_learning.inception_configuration import InceptionConfiguration
from ...utils.smiles import convert_to_rdkit_smiles


class Inception:
    def __init__(self, configuration: InceptionConfiguration, scoring_function, prior):
        self.configuration = configuration
        self._initialize_memory(scoring_function, prior)

    def _initialize_memory(self, scoring_function, prior):
        self.memory: pd.DataFrame = pd.DataFrame(columns=['smiles', 'score', 'likelihood'])
        if len(self.configuration.smiles):
            standardized_and_nulls = [convert_to_rdkit_smiles(smile) for smile in self.configuration.smiles]
            standardized = [smile for smile in standardized_and_nulls if smile is not None]
            self.evaluate_and_add(standardized, scoring_function, prior)

    def _purge_memory(self):
        unique_df = self.memory.drop_duplicates(subset=["smiles"])
        sorted_df = unique_df.sort_values('score', ascending=False)
        self.memory = sorted_df.head(self.configuration.memory_size)

    def evaluate_and_add(self, smiles, scoring_function, prior):
        if len(smiles) > 0:
            score = scoring_function.get_final_score(smiles)
            likelihood = prior.likelihood_smiles(smiles)
            df = pd.DataFrame({"smiles": smiles, "score": score.total_score, "likelihood": -likelihood.detach().cpu().numpy()})
            self.memory = self.memory.append(df)
            self._purge_memory()

    def add(self, smiles, score, neg_likelihood):
        # NOTE: likelihood should be already negative
        df = pd.DataFrame({"smiles": smiles, "score": score, "likelihood": neg_likelihood.detach().cpu().numpy()})
        self.memory = self.memory.append(df)
        self._purge_memory()

    def sample(self):
        sample_size = min(len(self.memory), self.configuration.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            smiles = sampled["smiles"].values
            scores = sampled["score"].values
            prior_likelihood = sampled["likelihood"].values
            return smiles, scores, prior_likelihood
        return [], [], [] #TODO: think of better return types

    def log_out_memory(self, path):
        self.memory.to_csv(path)
        # self.logger.log_out_experience(path, self.memory)
