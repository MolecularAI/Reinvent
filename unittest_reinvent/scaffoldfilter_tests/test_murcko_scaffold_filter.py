import os
import shutil
import unittest

import numpy as np

from scaffold.scaffold_filter_factory import ScaffoldFilterFactory
from scaffold.scaffold_parameters import ScaffoldParameters
from scoring.component_parameters import ComponentParameters
from scoring.score_summary import FinalSummary, ComponentSummary
from unittest_reinvent.fixtures.paths import MAIN_TEST_PATH
from utils.enums.scaffold_filter_enum import ScaffoldFilterEnum
from utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class Test_murcko_scaffold_filter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scaffold_enum = ScaffoldFilterEnum()
        cls.sf_enum = ScoringFunctionComponentNameEnum()
        cls.workfolders = [MAIN_TEST_PATH]

    def setUp(self):
        # create a scaffold filter and fill it with a few entries
        smiles = ["O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N", "CCC", "CCCCC"]
        scores = np.array([0.7, 0.5, 0.])
        valid_idx = [0, 1, 2]
        component_parameters = ComponentParameters(component_type=self.sf_enum.TANIMOTO_SIMILARITY,
                                                   name="tanimoto_similarity",
                                                   weight=1.,
                                                   smiles=smiles,
                                                   model_path="",
                                                   specific_parameters={})
        component_score_summary = ComponentSummary(scores, component_parameters)

        final_summary = FinalSummary(scores, smiles, valid_idx,
                                     [component_score_summary], [component_score_summary])

        sf_parameters = ScaffoldParameters(name=self.scaffold_enum.IDENTICAL_MURCKO_SCAFFOLD, minscore=0.5,
                                           minsimilarity=0.4, nbmax=1)
        scaffold_filter_factory = ScaffoldFilterFactory()
        self.scaffold_filter = scaffold_filter_factory.load_scaffold_filter(sf_parameters)
        self.scaffold_filter.score(final_summary)

    @classmethod
    def tearDownClass(self):
        for path in self.workfolders:
            if os.path.isdir(path):
                shutil.rmtree(path)

    def test_save_to_csv(self):
        folder = self.workfolders[0]
        self.scaffold_filter.save_to_csv(folder)
        output_file = os.path.join(folder, "scaffold_memory.csv")
        self.assertEqual(os.path.isfile(output_file), True)

        with open(output_file, 'r') as o:
            lines = o.readlines()
        self.assertEqual(3, len(lines))

    def test_valid_addition(self):
        # add a new smile
        smiles = ["c1ccccc1CC"]
        scores = np.array([1.0])
        valid_idx = [0]
        component_parameters = ComponentParameters(component_type=self.sf_enum.TANIMOTO_SIMILARITY,
                                                   name="tanimoto_similarity",
                                                   weight=1.,
                                                   smiles=smiles,
                                                   model_path="",
                                                   specific_parameters={})
        component_score_summary = ComponentSummary(scores, component_parameters)
        final_summary = FinalSummary(scores, smiles, valid_idx,
                                     [component_score_summary], [component_score_summary])
        self.scaffold_filter.score(final_summary)
        self.assertEqual(3, len(self.scaffold_filter.scaffolds))

    def test_invalid_addition(self):
        # try to add a smile already present
        smiles = ["CCC"]
        scores = np.array([1.0])
        valid_idx = [0]
        component_parameters = ComponentParameters(component_type=self.sf_enum.TANIMOTO_SIMILARITY,
                                                   name="tanimoto_similarity",
                                                   weight=1.,
                                                   smiles=smiles,
                                                   model_path="",
                                                   specific_parameters={})
        component_score_summary = ComponentSummary(scores, component_parameters)
        final_summary = FinalSummary(scores, smiles, valid_idx,
                                     [component_score_summary], [component_score_summary])
        self.scaffold_filter.score(final_summary)
        self.assertEqual(2, len(self.scaffold_filter.scaffolds))

