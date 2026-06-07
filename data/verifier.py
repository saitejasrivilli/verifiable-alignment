"""
data/verifier.py

Symbolic math verifier for GSM8K and MATH answers.
Uses sympy for algebraic equivalence — not just string match.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MathVerifier:
    """
    Extracts and symbolically verifies math answers.

    Priority:
      1. \\boxed{} extraction  (MATH format)
      2. "#### N" extraction   (GSM8K format)
      3. Last number fallback
    """

    # ── Extraction patterns ────────────────────────────────────────────────────
    BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
    GSM_RE = re.compile(r"####\s*([-\d,\.]+)")
    NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")

    def extract_answer(self, text: str) -> Optional[str]:
        """Return the predicted answer string from model output."""
        # 1. \\boxed{}
        m = self.BOXED_RE.search(text)
        if m:
            return m.group(1).strip()

        # 2. GSM8K #### format
        m = self.GSM_RE.search(text)
        if m:
            return m.group(1).replace(",", "").strip()

        # 3. Last number in text
        nums = self.NUMBER_RE.findall(text)
        if nums:
            return nums[-1]

        return None

    def _to_sympy(self, expr: str):
        """Parse expression to sympy object. Returns None on failure."""
        try:
            from sympy import sympify, simplify
            from sympy.parsing.latex import parse_latex

            expr = expr.strip().replace(",", "")

            # Try direct sympify first (handles "3/4", "1.5", etc.)
            try:
                return sympify(expr)
            except Exception:
                pass

            # Try LaTeX parsing
            try:
                return parse_latex(expr)
            except Exception:
                pass

        except ImportError:
            logger.warning("sympy not available — falling back to string match")
        return None

    def verify(self, prediction: str, ground_truth: str) -> bool:
        """Return True if prediction is mathematically equivalent to ground_truth."""
        pred_str = self.extract_answer(prediction)
        true_str = self.extract_answer(ground_truth) or ground_truth.strip()

        if pred_str is None:
            return False

        # Fast path: exact string match after normalisation
        if self._normalise(pred_str) == self._normalise(true_str):
            return True

        # Symbolic equivalence via sympy
        pred_sym = self._to_sympy(pred_str)
        true_sym = self._to_sympy(true_str)

        if pred_sym is not None and true_sym is not None:
            try:
                from sympy import simplify, Abs
                diff = simplify(pred_sym - true_sym)
                return diff == 0
            except Exception:
                pass

        return False

    @staticmethod
    def _normalise(s: str) -> str:
        """Minimal normalisation for string comparison."""
        return s.strip().lower().replace(" ", "").replace(",", "")

    def batch_verify(
        self, predictions: list[str], ground_truths: list[str]
    ) -> list[bool]:
        return [self.verify(p, g) for p, g in zip(predictions, ground_truths)]
