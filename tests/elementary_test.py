import pytest
import numpy as np
from expects import expect, equal, raise_error, be_true, be_within

from autodiff_team29 import elementaries
from autodiff_team29.node import Node


@pytest.fixture()
def negative_two():
    return Node("x", -2, 1)


@pytest.fixture()
def positive_two():
    return Node("x", 2, 1)


class TestDomainRestrictions:
    def test_log_domain_raises_error(self, negative_two):

        with pytest.raises(ValueError):
            expect(elementaries._check_log_domain_restrictions(negative_two)).to(
                raise_error(ValueError)
            )

    def test_sqrt_domain_raises_error(self, negative_two):

        with pytest.raises(ValueError):
            expect(elementaries._check_sqrt_domain_restrictions(negative_two)).to(
                raise_error(ValueError)
            )

    def test_arccos_domain_raises_error(self, negative_two, positive_two):

        with pytest.raises(ValueError):
            expect(elementaries._check_arccos_domain_restrictions(negative_two)).to(
                raise_error(ValueError)
            )

            expect(elementaries._check_arccos_domain_restrictions(positive_two)).to(
                raise_error(ValueError)
            )

    def test_arcsin_domain_raises_error(self, negative_two, positive_two):

        with pytest.raises(ValueError):
            expect(elementaries._check_arcsin_domain_restrictions(negative_two)).to(
                raise_error(ValueError)
            )

            expect(elementaries._check_arcsin_domain_restrictions(positive_two)).to(
                raise_error(ValueError)
            )


class TestMathFunctions:
    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (4, "sqrt(4)", 2.0, 1 / 4),
            (9, "sqrt(9)", 3.0, 1 / 6),
            (16, "sqrt(16)", 4.0, 1 / 8),
            (Node("4", 4, 1), "sqrt(4)", 2.0, 1 / 4),
            (Node("9", 9, 1), "sqrt(9)", 3.0, 1 / 6),
            (Node("16", 16, 1), "sqrt(16)", 4.0, 1 / 8),
        ],
    )
    def test_sqrt(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.sqrt(input).symbol).to(equal(expected_symbol))
        expect(elementaries.sqrt(input).value).to(equal(expected_value))
        expect(elementaries.sqrt(input).derivative).to(equal(expected_derivative))

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (1, "ln(1)", np.log(1), 1 / 1),
            (10, "ln(10)", np.log(10), 1 / 10),
        ],
    )
    def test_ln(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.ln(input).symbol).to(equal(expected_symbol))
        expect(elementaries.ln(input).value).to(equal(expected_value))
        expect(elementaries.ln(input).derivative).to(equal(expected_derivative))

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (1, "log10(1)", np.log10(1), 0.43429448190325175),
            (10, "log10(10)", np.log10(10), 0.043429448190325175),
        ],
    )
    def test_log10(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.log10(input).symbol).to(equal(expected_symbol))
        expect(elementaries.log10(input).value).to(equal(expected_value))
        expect(
            np.allclose(elementaries.log10(input).derivative, expected_derivative)
        ).to(be_true)

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (1, "log2(1)", np.log2(1), 1.4426950408889634),
            (10, "log2(10)", np.log2(10), 0.4426950408889634),
        ],
    )
    def test_log2(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.log2(input).symbol).to(equal(expected_symbol))
        expect(elementaries.log2(input).value).to(equal(expected_value))
        expect(np.allclose(elementaries.log2(input).value, expected_value)).to(be_true)

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [(1, "exp(1)", np.exp(1), np.e), (0, "exp(0)", np.exp(0), 1)],
    )
    def test_exp(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.exp(input).symbol).to(equal(expected_symbol))
        expect(elementaries.exp(input).value).to(equal(expected_value))
        expect(np.allclose(elementaries.exp(input).value, expected_value)).to(be_true)

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (10, "sin(10)", np.sin(10), np.cos(10)),
            (25, "sin(25)", np.sin(25), np.cos(25)),
        ],
    )
    def test_sin(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.sin(input).symbol).to(equal(expected_symbol))
        expect(elementaries.sin(input).value).to(equal(expected_value))
        expect(np.allclose(elementaries.sin(input).derivative, expected_derivative)).to(
            be_true
        )

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (10, "cos(10)", np.cos(10), -np.sin(10)),
            (25, "cos(25)", np.cos(25), -np.sin(25)),
        ],
    )
    def test_cos(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.cos(input).symbol).to(equal(expected_symbol))
        expect(elementaries.cos(input).value).to(equal(expected_value))
        expect(np.allclose(elementaries.cos(input).derivative, expected_derivative)).to(
            be_true
        )

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (10, "tan(10)", np.tan(10), 1.4203717625834316),
            (25, "tan(25)", np.tan(25), 1.017829301372081),
        ],
    )
    def test_tan(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.tan(input).symbol).to(equal(expected_symbol))
        expect(elementaries.tan(input).value).to(equal(expected_value))
        expect(np.allclose(elementaries.tan(input).derivative, expected_derivative)).to(
            be_true
        )

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (0.25, "arcsin(0.25)", np.arcsin(0.25), 1.0327955589886444),
            (0.5, "arcsin(0.5)", np.arcsin(0.5), 1.1547005383792517),
        ],
    )
    def test_arcsin(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.arcsin(input).symbol).to(equal(expected_symbol))
        expect(elementaries.arcsin(input).value).to(equal(expected_value))
        expect(
            np.allclose(elementaries.arcsin(input).derivative, expected_derivative)
        ).to(be_true)

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (0.25, "arccos(0.25)", np.arccos(0.25), -1.0327955589886444),
            (0.5, "arccos(0.5)", np.arccos(0.5), -1.1547005383792517),
        ],
    )
    def test_arccos(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.arccos(input).symbol).to(equal(expected_symbol))
        expect(elementaries.arccos(input).value).to(equal(expected_value))
        expect(
            np.allclose(elementaries.arccos(input).derivative, expected_derivative)
        ).to(be_true)

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (10, "tan(10)", np.tan(10), 1 / (np.cos(10)) ** 2),
            (25, "tan(25)", np.tan(25), 1 / (np.cos(25)) ** 2),
        ],
    )
    def test_tan(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.tan(input).symbol).to(equal(expected_symbol))
        expect(elementaries.tan(input).value).to(equal(expected_value))
        expect(np.allclose(elementaries.tan(input).derivative, expected_derivative)).to(
            be_true
        )

    @pytest.mark.parametrize(
        "input, expected_symbol, expected_value, expected_derivative",
        [
            (10, "arctan(10)", np.arctan(10), 0.009900990099009901),
            (25, "arctan(25)", np.arctan(25), 0.001597444089456869),
        ],
    )
    def test_arctan(self, input, expected_symbol, expected_value, expected_derivative):
        expect(elementaries.arctan(input).symbol).to(equal(expected_symbol))
        expect(elementaries.arctan(input).value).to(equal(expected_value))
        expect(
            np.allclose(elementaries.arctan(input).derivative, expected_derivative)
        ).to(be_true)
