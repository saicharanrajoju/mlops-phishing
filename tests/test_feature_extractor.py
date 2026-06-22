"""Unit tests for the URL -> 30-feature extractor (pure logic, no network)."""

from ai_analyst.feature_extractor import URLFeatureExtractor
from phishsentinel.constant.training_pipeline import SCHEMA_COLUMNS


def feats(url: str) -> dict:
    return URLFeatureExtractor(url).get_features_dict()


def test_feature_vector_length_matches_schema():
    values = URLFeatureExtractor("https://example.com").get_features_list()
    assert len(values) == len(SCHEMA_COLUMNS) - 1  # all columns except the target "Result"


def test_ip_address_in_domain_is_flagged():
    assert feats("http://127.0.0.1/login")["having_IP_Address"] == -1


def test_regular_domain_not_flagged_as_ip():
    assert feats("https://example.com")["having_IP_Address"] == 1


def test_at_symbol_is_flagged():
    assert feats("http://legit.com@127.0.0.1/login")["having_At_Symbol"] == -1


def test_https_marks_ssl_safe():
    assert feats("https://example.com")["SSLfinal_State"] == 1


def test_http_marks_ssl_unsafe():
    assert feats("http://example.com")["SSLfinal_State"] == -1


def test_dash_in_domain_flags_prefix_suffix():
    assert feats("http://secure-paypal@127.0.0.1")["Prefix_Suffix"] == -1


def test_url_shortener_is_flagged():
    assert feats("http://bit.ly/abcd")["Shortining_Service"] == -1


def test_protocol_is_prepended_when_missing():
    # No scheme provided -> extractor should default to http and still parse.
    ext = URLFeatureExtractor("example.com")
    assert ext.url.startswith("http://")


def test_all_feature_values_are_valid_codes():
    # Every feature must be one of the three categorical codes the model expects.
    for value in feats("https://example.com").values():
        assert value in (-1, 0, 1)
