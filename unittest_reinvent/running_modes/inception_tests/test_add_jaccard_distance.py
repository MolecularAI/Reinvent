from unittest_reinvent.running_modes.inception_tests.test_add_base import TestInceptionAddBase


class TestInceptionAddJaccardDistance(TestInceptionAddBase):

    def test_eval_add_1(self):
        self.assertEqual(len(self.inception_model.memory), 3)
