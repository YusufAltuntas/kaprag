"""The exposure guard of the chat demo (kaprag.serve.exposure): pure, no Streamlit."""

import pytest

from kaprag.serve.exposure import check_exposure


@pytest.mark.parametrize("address", ["127.0.0.1", "localhost", "::1", "127.0.0.2", "LOCALHOST"])
@pytest.mark.parametrize("backend", ["claude_code", "anthropic_api"])
def test_loopback_is_allowed_without_warning(address, backend):
    decision = check_exposure(address, backend)
    assert decision.allowed and not decision.public


@pytest.mark.parametrize("address", [None, "", "  ", "0.0.0.0", "::", "192.168.1.10", "myhost"])
def test_non_loopback_on_the_subscription_is_refused(address):
    decision = check_exposure(address, "claude_code")
    assert decision.public and not decision.allowed


@pytest.mark.parametrize("address", [None, "0.0.0.0", "192.168.1.10"])
def test_non_loopback_on_the_api_key_is_allowed_with_a_warning(address):
    decision = check_exposure(address, "anthropic_api")
    assert decision.public and decision.allowed


def test_unknown_backend_on_a_public_address_is_refused():
    assert not check_exposure("0.0.0.0", "something_else").allowed
