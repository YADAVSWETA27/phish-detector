import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cli
from phishing import data, features, model, synth
from phishing.features import FEATURE_NAMES, explain, extract, registrable_domain, vector


# ---------------------------------------------------------------- features ----
def test_vector_matches_feature_names():
    assert len(vector("https://example.com")) == len(FEATURE_NAMES)


def test_registrable_domain():
    assert registrable_domain("login.paypal.evil.xyz") == "evil.xyz"
    assert registrable_domain("www.bbc.co.uk") == "bbc.co.uk"
    assert registrable_domain("example.com") == "example.com"


def test_ip_host_detected():
    assert extract("http://192.0.2.4/login")["is_ip"] == 1
    assert extract("https://example.com")["is_ip"] == 0


def test_at_sign_trick():
    assert extract("http://paypal.com@evil.xyz/x")["has_at"] == 1
    assert extract("https://example.com/a?mail=me@x.com")["has_at"] == 0  # '@' in the query is fine


def test_brand_in_subdomain_but_not_real_domain():
    assert extract("http://paypal.secure-login.xyz/")["brand_in_subdomain"] == 1
    assert extract("https://www.paypal.com/signin")["brand_in_subdomain"] == 0   # real domain


def test_scheme_missing_is_handled():
    assert extract("example.com/login")["is_https"] == 0
    assert extract("example.com/login")["host_len"] == len("example.com")


def test_weird_input_never_crashes():
    for u in ["", "   ", "http://", "://", "http://[::1]:99999/x", "http://exa mple.com", "a" * 5000, "%%%", "http://xn--a.b"]:
        assert len(vector(u)) == len(FEATURE_NAMES)


def test_explain_lists_reasons():
    why = " ".join(explain("http://paypal.secure-login.xyz/verify"))
    assert "paypal" in why and ".xyz" in why
    assert not [w for w in explain("https://www.wikipedia.org/wiki/Python") if "HTTPS" not in w]
    assert explain("https://www.wikipedia.org") == []


def test_features_do_not_touch_the_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    extract("http://evil.example/login")


# -------------------------------------------------------------------- data ----
def test_synthetic_data_is_balanced_and_reproducible():
    a, b = synth.generate(400, seed=1), synth.generate(400, seed=1)
    assert a == b and len(a) == 400
    assert 0.4 < sum(l for _, l in a) / 400 < 0.6


def test_csv_loader_guesses_columns_and_skips_bad_rows(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("URL,Label\nhttp://a.xyz/login,bad\nhttps://b.com,good\nhttps://b.com,good\nhttp://c.top,unknown\n,bad\n")
    assert data.load_csv(str(p)) == [("http://a.xyz/login", 1), ("https://b.com", 0)]


def test_csv_loader_errors(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        data.load_csv(str(p))
    p.write_text("url,label\nx.com,maybe\n")
    with pytest.raises(ValueError):
        data.load_csv(str(p))


# ------------------------------------------------------------------- model ----
@pytest.fixture(scope="module")
def trained():
    return model.train(synth.generate(2000, seed=3))


def test_beats_chance_on_held_out_data(trained):
    assert trained["forest"]["f1"] > 0.85
    assert trained["n_train"] + trained["n_test"] == 2000


def test_metrics_counts_add_up(trained):
    m = trained["forest"]
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == trained["n_test"]


def test_obvious_cases(trained):
    mdl = trained["model"]
    assert model.check(mdl, "http://paypal.secure-login.xyz/verify")["verdict"] == "PHISHING"
    assert model.check(mdl, "https://www.wikipedia.org/wiki/Python")["verdict"] == "LEGITIMATE"


def test_threshold_changes_verdict(trained):
    url = "http://paypal.secure-login.xyz/verify"
    assert model.check(trained["model"], url, threshold=0.999999)["verdict"] == "LEGITIMATE"


def test_save_load_roundtrip(trained, tmp_path):
    p = str(tmp_path / "m.joblib")
    model.save(trained["model"], p)
    url = "http://192.0.2.4/bank/login.php"
    assert model.check(model.load(p), url)["phishing_probability"] == model.check(trained["model"], url)["phishing_probability"]


def test_model_with_other_features_is_refused(tmp_path):
    import joblib
    p = str(tmp_path / "old.joblib")
    joblib.dump({"model": None, "features": ["x"]}, p)
    with pytest.raises(ValueError):
        model.load(p)


def test_no_leakage_between_train_and_test():
    rows = synth.generate(600, seed=5)
    res = model.train(rows)
    # a model that memorised everything would score 1.0; an honest split on noisy labels cannot
    assert res["forest"]["f1"] < 1.0


# --------------------------------------------------------------------- CLI ----
def test_cli_train_then_check(tmp_path, capsys):
    out = str(tmp_path / "m.joblib")
    assert cli.main(["train", "--n", "800", "--out", out]) == 0
    assert "SYNTHETIC" in capsys.readouterr().out
    assert cli.main(["check", "--model", out, "http://paypal.secure-login.xyz/verify"]) == 0
    assert "PHISHING" in capsys.readouterr().out


def test_cli_check_without_model(tmp_path, capsys):
    assert cli.main(["check", "--model", str(tmp_path / "none"), "x.com"]) == 2


# --------------------------------------------------------- dataset builder ----
def test_dataset_builder_is_balanced_and_host_only():
    import build_dataset as bd
    phish = ["http://evil.xyz/login/a.php", "http://evil.xyz/other", "https://bad.top/x?y=1", "junk"]
    tranco = [(str(i), f"site{i}.com") for i in range(50)]
    rows = bd.build(phish, tranco)
    labels = [l for _, l in rows]
    assert labels.count(1) == labels.count(0) == 2          # duplicate host merged, junk dropped
    assert all(urlsplit_path == "" for urlsplit_path in (__import__("urllib.parse").parse.urlsplit(u).path for u, _ in rows))
