"""
AutoDrive RAG v2.0 — Statistical Significance Tests
Determines whether performance differences between model variants
are statistically significant, not just due to random chance.

Tests:
  - Paired t-test (for continuous metrics like F1)
  - McNemar's test (for binary correct/incorrect)
  - Bootstrap confidence intervals (non-parametric)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("chatbot.evaluation.stats")


class StatisticalTests:
    """Statistical significance testing for RAG evaluation."""

    @staticmethod
    def paired_t_test(
        scores_a: list[float],
        scores_b: list[float],
        alpha: float = 0.05,
    ) -> dict:
        """
        Paired t-test for two sets of per-question scores.

        Tests H0: mean(A) == mean(B) vs H1: mean(A) != mean(B).

        Args:
            scores_a: Per-question scores for system A.
            scores_b: Per-question scores for system B.
            alpha: Significance level (default 0.05).

        Returns:
            Dict with t-statistic, p-value, and significance verdict.
        """
        from scipy import stats

        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have equal length")

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        significant = p_value < alpha

        mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
        diff = mean_a - mean_b

        return {
            "test": "paired_t_test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "alpha": alpha,
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "mean_difference": float(diff),
            "interpretation": (
                f"{'Significant' if significant else 'Not significant'} "
                f"(p={p_value:.4f}, α={alpha}). "
                f"System {'A' if diff > 0 else 'B'} is better by {abs(diff):.4f}."
            ),
        }

    @staticmethod
    def mcnemar_test(
        correct_a: list[bool],
        correct_b: list[bool],
        alpha: float = 0.05,
    ) -> dict:
        """
        McNemar's test for paired binary (correct/incorrect) outcomes.

        Tests whether the two systems disagree in a systematic way.

        Args:
            correct_a: Per-question correctness for system A.
            correct_b: Per-question correctness for system B.
            alpha: Significance level.

        Returns:
            Dict with chi-squared statistic, p-value, and significance.
        """
        from scipy import stats

        if len(correct_a) != len(correct_b):
            raise ValueError("Lists must have equal length")

        # Build contingency table
        # b = A correct, B wrong
        # c = A wrong, B correct
        b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
        c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)

        if b + c == 0:
            return {
                "test": "mcnemar",
                "chi2": 0.0,
                "p_value": 1.0,
                "significant": False,
                "alpha": alpha,
                "interpretation": "Systems agree on all samples — no difference.",
            }

        # McNemar's with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        return {
            "test": "mcnemar",
            "chi2": float(chi2),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "alpha": alpha,
            "a_only_correct": b,
            "b_only_correct": c,
            "interpretation": (
                f"{'Significant' if p_value < alpha else 'Not significant'} "
                f"(p={p_value:.4f}). A-only correct: {b}, B-only correct: {c}."
            ),
        }

    @staticmethod
    def bootstrap_ci(
        scores: list[float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
    ) -> dict:
        """
        Bootstrap confidence interval for a metric.

        Args:
            scores: Per-question scores.
            n_bootstrap: Number of bootstrap resamples.
            confidence: Confidence level (default 0.95 = 95% CI).

        Returns:
            Dict with mean, CI lower, CI upper.
        """
        scores_arr = np.array(scores)
        n = len(scores_arr)

        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores_arr, size=n, replace=True)
            boot_means.append(np.mean(sample))

        boot_means = np.array(boot_means)
        lower_pct = (1 - confidence) / 2 * 100
        upper_pct = (1 + confidence) / 2 * 100

        return {
            "mean": float(np.mean(scores_arr)),
            "ci_lower": float(np.percentile(boot_means, lower_pct)),
            "ci_upper": float(np.percentile(boot_means, upper_pct)),
            "confidence": confidence,
            "n_bootstrap": n_bootstrap,
            "std": float(np.std(scores_arr)),
        }

    def compare_systems(
        self,
        system_a_scores: dict[str, list[float]],
        system_b_scores: dict[str, list[float]],
        system_a_name: str = "System A",
        system_b_name: str = "System B",
    ) -> dict:
        """
        Full statistical comparison of two systems across all metrics.

        Args:
            system_a_scores: {metric_name: [per_question_scores]}
            system_b_scores: {metric_name: [per_question_scores]}

        Returns:
            Dict with comparison results for each metric.
        """
        results = {}

        for metric in system_a_scores:
            if metric not in system_b_scores:
                continue

            a = system_a_scores[metric]
            b = system_b_scores[metric]

            results[metric] = {
                "t_test": self.paired_t_test(a, b),
                "bootstrap_a": self.bootstrap_ci(a),
                "bootstrap_b": self.bootstrap_ci(b),
            }

            # McNemar for EM (binary)
            if metric == "exact_match":
                correct_a = [s > 0.5 for s in a]
                correct_b = [s > 0.5 for s in b]
                results[metric]["mcnemar"] = self.mcnemar_test(correct_a, correct_b)

        return {
            "system_a": system_a_name,
            "system_b": system_b_name,
            "metrics": results,
        }
