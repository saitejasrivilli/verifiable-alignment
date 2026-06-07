"""tests/test_verifier.py"""
import pytest
from data.verifier import MathVerifier

v = MathVerifier()


class TestExtraction:
    def test_boxed(self):
        assert v.extract_answer(r"Therefore \boxed{42}") == "42"

    def test_gsm_format(self):
        assert v.extract_answer("The answer is #### 30") == "30"

    def test_last_number_fallback(self):
        assert v.extract_answer("So we get 3 + 4 = 7") == "7"

    def test_no_answer(self):
        assert v.extract_answer("There is no number here.") is None


class TestVerification:
    def test_exact_match(self):
        assert v.verify(r"\boxed{42}", "42")

    def test_fraction_equivalence(self):
        assert v.verify(r"\boxed{1/2}", r"\boxed{0.5}")

    def test_wrong_answer(self):
        assert not v.verify(r"\boxed{43}", "42")

    def test_latex_expression(self):
        assert v.verify(r"\boxed{x^2 + 2x + 1}", r"\boxed{(x+1)^2}")

    def test_gsm_format_correct(self):
        assert v.verify("Step 1 ... #### 30", "30")

    def test_format_failure(self):
        assert not v.verify("I don't know the answer.", "42")


class TestNormalisation:
    def test_whitespace(self):
        assert v._normalise("  42  ") == "42"

    def test_comma_in_number(self):
        assert v._normalise("1,000") == "1000"
