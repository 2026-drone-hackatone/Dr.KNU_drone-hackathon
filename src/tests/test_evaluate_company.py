import unittest

from src.evaluation.evaluate_company import evaluate


class EvaluateCompanyTest(unittest.TestCase):
    def test_evaluate_entrypoint_is_callable(self) -> None:
        self.assertTrue(callable(evaluate))
