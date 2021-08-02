import unittest
from unittest.mock import Mock

from running_modes.reinforcement_learning.margin_guard import MarginGuard


class MarginGuardAdjustGuardOnEmptyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = Mock()
        self.margin_window = 10
        self.mg = MarginGuard(self.runner, self.margin_window)

    def test_empty(self):
        self.assertRaises(Exception, self.mg.adjust_margin, self.margin_window)
