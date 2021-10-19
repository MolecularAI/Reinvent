import numpy as np
import numpy.testing as nt

from unittest_reinvent.running_modes.inception_tests.test_add_base import TestInceptionAddBase
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.function import CustomSum
from unittest_reinvent.fixtures.paths import PRIOR_PATH
from unittest_reinvent.fixtures.test_data import PROPANE, BUTANE, ETHANE, ASPIRIN


class TestInceptionEvalAddTanimoto(TestInceptionAddBase):

    def setUp(self):
        super().setUp()

        self.smiles = np.array([PROPANE, BUTANE, ASPIRIN])
        scoring = ComponentParameters(component_type=self.sf_enum.TANIMOTO_SIMILARITY,
                                      name="tanimoto_similarity",
                                      weight=1.,
                                      specific_parameters={"smiles":[PROPANE, ASPIRIN]})
        self.scoring_function = CustomSum(parameters=[scoring])
        self.prior = Model.load_from_file(PRIOR_PATH)

    def test_eval_and_add_tanimoto(self):
        self.inception_model.evaluate_and_add(self.smiles, self.scoring_function, self.prior)
        self.assertEqual(len(self.inception_model.memory), 4)
        nt.assert_almost_equal(np.array(self.inception_model.memory['score'].values),
                               np.array([1, 1, 0.9464, 0.9375]), 4)
