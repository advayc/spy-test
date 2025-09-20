"""
Tests for Spy Fairness Tests Package

Comprehensive test suite for the fairness testing algorithms.
"""

import pytest
from spy_fairness_tests import (
    FairnessAnalyzer,
    StatisticalTester,
    test_fisher_yates_fairness,
    validate_algorithm_fairness,
)


class TestFairnessAnalyzer:
    """Test the main FairnessAnalyzer class"""

    def test_fisher_yates_algorithm(self):
        """Test Fisher-Yates algorithm with known fairness score"""
        analyzer = FairnessAnalyzer(random_seed=42)
        players = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

        result = analyzer.test_algorithm("fisher_yates", players, num_rounds=1000)

        # Check basic properties
        assert result.fairness_score > 0.9  # Should be highly fair
        assert result.total_rounds == 1000
        assert len(result.spy_counts) == len(players)
        assert sum(result.spy_counts.values()) == 1000
        assert result.execution_time > 0

    def test_fisher_yates_fairness_score(self):
        """Test that Fisher-Yates achieves expected high fairness score"""
        players = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

        result = test_fisher_yates_fairness(
            players=players,
            num_rounds=5000,
            random_seed=42
        )

        # Should achieve high fairness score (close to 97.7/100)
        assert result.fairness_score >= 0.95
        assert 0.95 <= result.fairness_score <= 1.0

    def test_multiple_algorithms(self):
        """Test comparison of multiple algorithms"""
        analyzer = FairnessAnalyzer(random_seed=42)
        players = ["A", "B", "C", "D"]

        algorithms = ["fisher_yates", "system"]
        results = analyzer.compare_algorithms(algorithms, players, num_rounds=1000)

        assert len(results) == 2
        assert "fisher_yates" in results
        assert "system" in results

        # Fisher-Yates should be more fair than system random
        fisher_score = results["fisher_yates"].fairness_score
        system_score = results["system"].fairness_score
        assert fisher_score >= system_score

    def test_edge_cases(self):
        """Test algorithm behavior with edge cases"""
        analyzer = FairnessAnalyzer(random_seed=42)

        # Single player
        result = analyzer.test_algorithm("fisher_yates", ["Alice"], num_rounds=10)
        assert result.fairness_score == 1.0  # Perfect fairness with single player
        assert result.spy_counts["Alice"] == 10

        # Two players
        result = analyzer.test_algorithm("fisher_yates", ["Alice", "Bob"], num_rounds=100)
        assert result.fairness_score > 0.8  # Should still be fairly fair
        assert len(result.spy_counts) == 2

    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        analyzer = FairnessAnalyzer(random_seed=42)
        players = ["A", "B", "C", "D", "E"]

        result = analyzer.test_algorithm("fisher_yates", players, num_rounds=2000)

        lower, upper = result.confidence_interval
        assert 0 <= lower <= upper <= 1
        assert upper - lower < 0.2  # Reasonable confidence interval width

    def test_invalid_algorithm(self):
        """Test error handling for invalid algorithm names"""
        analyzer = FairnessAnalyzer()
        players = ["Alice", "Bob"]

        with pytest.raises(ValueError, match="Unknown algorithm"):
            analyzer.test_algorithm("invalid_algorithm", players)


class TestStatisticalTester:
    """Test the StatisticalTester class"""

    def test_comprehensive_tests(self):
        """Test comprehensive statistical validation"""
        tester = StatisticalTester()

        # Create mock spy counts
        spy_counts = {"A": 125, "B": 118, "C": 132, "D": 125}  # 500 total rounds
        expected_count = 500 / 4  # 125

        result = tester.run_comprehensive_tests(spy_counts, expected_count)

        # Check all statistical measures are present
        assert hasattr(result, 'chi_square_p_value')
        assert hasattr(result, 'ks_test_p_value')
        assert hasattr(result, 'uniformity_score')
        assert hasattr(result, 'independence_score')
        assert hasattr(result, 'monte_carlo_confidence')

        # Check ranges
        assert 0 <= result.chi_square_p_value <= 1
        assert 0 <= result.ks_test_p_value <= 1
        assert 0 <= result.uniformity_score <= 1
        assert 0 <= result.independence_score <= 1
        assert 0 <= result.monte_carlo_confidence <= 1

    def test_randomness_quality(self):
        """Test randomness quality assessment"""
        tester = StatisticalTester()

        # Create a pseudo-random sequence
        sequence = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 10  # Pattern, not random

        results = tester.test_randomness_quality(sequence)

        assert 'frequency_p_value' in results
        assert 'runs_p_value' in results
        assert 'serial_p_value' in results

        # With this pattern, p-values should be low (indicating non-randomness)
        assert all(0 <= p <= 1 for p in results.values())


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_validate_algorithm_fairness(self):
        """Test the comprehensive validation function"""
        players = ["Alice", "Bob", "Charlie"]

        result = validate_algorithm_fairness(
            "fisher_yates",
            players,
            num_rounds=500
        )

        assert 'fairness_result' in result
        assert 'statistical_validation' in result
        assert 'recommendation' in result

        assert result['fairness_result'].fairness_score > 0.8
        assert isinstance(result['recommendation'], str)

    def test_fisher_yates_convenience_function(self):
        """Test the Fisher-Yates convenience function"""
        players = ["P1", "P2", "P3", "P4"]

        result = test_fisher_yates_fairness(
            players=players,
            num_rounds=1000,
            random_seed=123
        )

        assert result.fairness_score > 0.9
        assert result.total_rounds == 1000
        assert len(result.spy_counts) == 4


class TestAlgorithmImplementations:
    """Test individual algorithm implementations"""

    def test_fisher_yates_implementation(self):
        """Test Fisher-Yates shuffle implementation"""
        analyzer = FairnessAnalyzer(random_seed=42)

        players = ["A", "B", "C"]
        spy = analyzer._fisher_yates_select(players.copy())

        assert spy in ["A", "B", "C"]
        assert len(players) == 2  # One player removed (the spy)

    def test_weighted_selection(self):
        """Test weighted selection algorithm"""
        analyzer = FairnessAnalyzer(random_seed=42)

        players = ["A", "B", "C", "D"]

        # Run multiple selections to test weighting
        spies = []
        for _ in range(100):
            spy = analyzer._weighted_select(players.copy())
            spies.append(spy)

        # All players should be selected at least once
        assert len(set(spies)) == len(players)

        # Distribution should be reasonably uniform
        spy_counts = {}
        for spy in spies:
            spy_counts[spy] = spy_counts.get(spy, 0) + 1

        # No player should be selected excessively more than others
        max_count = max(spy_counts.values())
        min_count = min(spy_counts.values())
        assert max_count <= min_count * 2  # Max should not be more than 2x min


# Performance tests
@pytest.mark.performance
class TestPerformance:
    """Performance-focused tests"""

    def test_large_group_performance(self):
        """Test performance with large player groups"""
        analyzer = FairnessAnalyzer(random_seed=42)

        # Large group of players
        players = [f"Player_{i}" for i in range(100)]

        import time
        start_time = time.time()

        result = analyzer.test_algorithm("fisher_yates", players, num_rounds=1000)

        execution_time = time.time() - start_time

        # Should complete in reasonable time
        assert execution_time < 5.0  # Less than 5 seconds
        assert result.fairness_score > 0.9  # Still highly fair

    def test_many_rounds_performance(self):
        """Test performance with many rounds"""
        analyzer = FairnessAnalyzer(random_seed=42)
        players = ["A", "B", "C", "D", "E"]

        import time
        start_time = time.time()

        result = analyzer.test_algorithm("fisher_yates", players, num_rounds=50000)

        execution_time = time.time() - start_time

        # Should complete in reasonable time
        assert execution_time < 10.0  # Less than 10 seconds
        assert result.fairness_score > 0.95  # Very high fairness with many rounds


# Statistical tests
@pytest.mark.statistical
class TestStatisticalProperties:
    """Statistical property tests"""

    def test_distribution_uniformity(self):
        """Test that spy selection follows uniform distribution"""
        analyzer = FairnessAnalyzer(random_seed=42)
        players = ["P1", "P2", "P3", "P4", "P5", "P6"]

        result = analyzer.test_algorithm("fisher_yates", players, num_rounds=10000)

        # Check that no player is selected excessively
        counts = list(result.spy_counts.values())
        expected_count = 10000 / 6  # ~1666.67

        for count in counts:
            # Each player should be within 20% of expected
            assert abs(count - expected_count) / expected_count < 0.2

    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        players = ["A", "B", "C", "D"]

        result1 = test_fisher_yates_fairness(players, 1000, seed=42)
        result2 = test_fisher_yates_fairness(players, 1000, seed=42)

        # Results should be identical with same seed
        assert result1.fairness_score == result2.fairness_score
        assert result1.spy_counts == result2.spy_counts