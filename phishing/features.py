"""Turn a URL into numeric features, and into plain-English reasons.

Only the text of the URL is looked at. Nothing is ever fetched, so checking a link is safe.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import parse_qsl, urlsplit

KEYWORDS = ("login", "signin", "verify", "secure", "account", "update", "confirm", "banking",
            "password", "wallet", "suspend", "invoice", "billing", "support", "recover", "unlock")
BRANDS = ("paypal", "apple", "google", "microsoft", "amazon", "netflix", "facebook", "instagram",
          "whatsapp", "dropbox", "linkedin", "chase", "wellsfargo", "hsbc", "sbi", "paytm", "dhl", "fedex")
RISKY_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work", "support",
              "zip", "country", "kim", "loan", "men", "icu", "cyou", "rest", "buzz"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly"}
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu"}  # co.uk, com.au, ...

FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "query_len", "dots_in_host", "hyphens_in_host",
    "digit_ratio_host", "subdomain_count", "is_ip", "has_at", "is_https", "has_port",
    "path_depth", "percent_encodings", "param_count", "host_entropy", "keyword_count",
    "keyword_in_host", "brand_in_subdomain", "risky_tld", "is_shortener", "has_punycode",
    "double_slash_in_path", "digit_ratio_url", "longest_host_label",
]


def _split(url: str):
    url = url.strip()
    if "://" not in url:
        url = "http://" + url  # people paste "example.com/login"; treat it as a web link
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().rstrip(".")
    return url, parts, host


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return bool(re.fullmatch(r"(0x[0-9a-f]+|\d+)(\.(0x[0-9a-f]+|\d+)){0,3}", host)) and "." in host


def registrable_domain(host: str) -> str:
    """'login.paypal.evil.xyz' -> 'evil.xyz'. A simple approximation (no public-suffix list needed)."""
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _brand_in_subdomain(host: str) -> str:
    """A brand name in the host but NOT in the real (registrable) domain is the classic trick."""
    real = registrable_domain(host)
    rest = host[: len(host) - len(real)] if host.endswith(real) else host
    for b in BRANDS:
        if b in rest and b not in real:
            return b
    return ""


def extract(url: str) -> dict:
    full, parts, host = _split(url)
    path, query = parts.path or "", parts.query or ""
    lowered = full.lower()
    labels = [l for l in host.split(".") if l]
    is_ip = _is_ip(host)
    tld = labels[-1] if labels else ""
    real = registrable_domain(host)
    sub_count = 0 if is_ip else max(0, len(labels) - len(real.split(".")))
    digits_host = sum(c.isdigit() for c in host)
    return {
        "url_len": len(full), "host_len": len(host), "path_len": len(path), "query_len": len(query),
        "dots_in_host": host.count("."), "hyphens_in_host": host.count("-"),
        "digit_ratio_host": digits_host / len(host) if host else 0.0,
        "subdomain_count": sub_count, "is_ip": int(is_ip), "has_at": int("@" in full.split("?")[0]),
        "is_https": int(parts.scheme == "https"), "has_port": int(parts.port not in (None, 80, 443)) if _port_ok(parts) else 0,
        "path_depth": len([p for p in path.split("/") if p]), "percent_encodings": len(re.findall(r"%[0-9a-fA-F]{2}", full)),
        "param_count": len(parse_qsl(query, keep_blank_values=True)), "host_entropy": _entropy(host),
        "keyword_count": sum(k in lowered for k in KEYWORDS),
        "keyword_in_host": int(any(k in host for k in KEYWORDS)),
        "brand_in_subdomain": int(bool(_brand_in_subdomain(host))), "risky_tld": int(tld in RISKY_TLDS),
        "is_shortener": int(real in SHORTENERS or host in SHORTENERS),
        "has_punycode": int("xn--" in host), "double_slash_in_path": int("//" in path),
        "digit_ratio_url": sum(c.isdigit() for c in full) / len(full), "longest_host_label": max((len(l) for l in labels), default=0),
    }


def _port_ok(parts) -> bool:
    try:
        parts.port
        return True
    except ValueError:
        return False


def vector(url: str) -> list[float]:
    f = extract(url)
    return [float(f[n]) for n in FEATURE_NAMES]


def explain(url: str) -> list[str]:
    """Human-readable red flags found in the URL (independent of the ML model)."""
    f, (_, _, host) = extract(url), _split(url)
    why = []
    if f["is_ip"]:
        why.append("Uses a raw IP address instead of a domain name.")
    if f["has_at"]:
        why.append("Contains '@', which can hide the real destination of the link.")
    if f["brand_in_subdomain"]:
        why.append(f"Contains the brand name '{_brand_in_subdomain(host)}' but the real domain is "
                   f"'{registrable_domain(host)}'.")
    if f["risky_tld"]:
        why.append(f"Uses the '.{host.split('.')[-1]}' ending, which is common in throw-away domains.")
    if f["keyword_in_host"]:
        why.append("The domain name itself contains words like login/verify/secure.")
    elif f["keyword_count"] >= 2:
        why.append("Contains several account-related words (login, verify, update...).")
    if f["hyphens_in_host"] >= 3:
        why.append("The domain has many hyphens.")
    if f["subdomain_count"] >= 3:
        why.append("Has many nested subdomains.")
    if f["has_punycode"]:
        why.append("Uses punycode (xn--), which can disguise look-alike characters.")
    if f["is_shortener"]:
        why.append("A link shortener hides the real destination.")
    if f["url_len"] > 90:
        why.append("The URL is unusually long.")
    if f["double_slash_in_path"]:
        why.append("Has '//' inside the path, a redirect trick.")
    if f["has_port"]:
        why.append("Uses an unusual port number.")
    if not f["is_https"]:
        why.append("Not using HTTPS (weak signal on its own: many phishing sites do use HTTPS).")
    return why
