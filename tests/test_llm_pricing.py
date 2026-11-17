import pytest

from kaprag.llm.pricing import PRICES, cost_usd
from kaprag.llm.types import Usage


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-haiku-5-5", 0.10 + 0.50),
        ("claude-sonnet-5-5", 2.00 + 10.00),
        ("claude-opus-5-5", 4.00 + 20.00),
    ],
)
def test_one_million_tokens_each_way(model, expected):
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(model, usage) == pytest.approx(expected)


def test_cache_tokens_are_priced_as_input():
    usage = Usage(
        input_tokens=2,
        output_tokens=0,
        cache_read_input_tokens=300,
        cache_creation_input_tokens=698,
    )
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(1000 * 2.00 / 1e6)


def test_small_call():
    assert cost_usd("claude-haiku-5-5", Usage(input_tokens=1000, output_tokens=200)) == (
        pytest.approx((1000 * 0.10 + 200 * 0.50) / 1e6)
    )


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        cost_usd("claude-unknown", Usage())


def test_project_models_are_priced():
    assert set(PRICES) == {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"}
