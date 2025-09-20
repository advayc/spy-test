"""
Spy Fairness Tests Package

Core algorithms and statistical analysis for evaluating the fairness
of spy selection algorithms in the Spy game.

Main Components:
- FairnessAnalyzer: Main class for fairness evaluation
- StatisticalTester: Advanced statistical validation
- Algorithm implementations: Fisher-Yates, LCG, System Random, Weighted

Key Results:
- Fisher-Yates Implementation: 97.7/100 Fairness Score
- Statistical Validation: Comprehensive testing with Chi-square, KS-test
- Performance: O(n) time complexity, O(1) additional space
"""

from .algorithms import (
    FairnessAnalyzer,
    StatisticalTester,
    FairnessResult,
    StatisticalTestResult,
    test_fisher_yates_fairness,
    validate_algorithm_fairness,
)

__version__ = "1.0.0"
__author__ = "Advay Chandorkar"
__email__ = "advay@example.com"

__all__ = [
    "FairnessAnalyzer",
    "StatisticalTester",
    "FairnessResult",
    "StatisticalTestResult",
    "test_fisher_yates_fairness",
    "validate_algorithm_fairness",
]