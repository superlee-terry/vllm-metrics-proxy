"""Tests for the DB-backed, ordered loop-detection rules.

Covers both the pure rule engine in ``vllm_metrics_proxy/loop_rules``
(ordering, enable/disable, custom rules, persistence round-trip) and the
maintenance API (``/api/loop-rules`` GET/PUT, ``/api/loop-rules/reset``).

The module-level ``_live_rules`` list is reset before each test so tests
never leak state into each other or into the running service (this file is
run under the same interpreter as the service, but in a fresh process).
"""
import json
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from vllm_metrics_proxy import loop_rules
from vllm_metrics_proxy.config import Settings
from vllm_metrics_proxy.auth import create_admin_token
from vllm_metrics_proxy.main import create_app


def make_settings():
    return Settings(
        vllm_upstream="http://localhost:11434",
        proxy_port=8000,
        db_path="/tmp/unused.db",
        log_level="ERROR",
        auth_enabled=False,
        dashboard_password="testpw",
    )


def reset_live_rules(rules=None):
    """Reset the module-level live list (test isolation)."""
    loop_rules._live_rules[:] = []
    if rules is not None:
        loop_rules._live_rules[:] = rules


@pytest.fixture(autouse=True)
def _isolate_rules():
    # Snapshot & restore the live list around each test.
    snapshot = list(loop_rules._live_rules)
    yield
    reset_live_rules(snapshot)


# ---- Engine: ordering / enable-disable / custom rules -------------------

def test_default_seeding():
    reset_live_rules()
    loop_rules.seed_default_rules()
    types = [r["type"] for r in loop_rules.get_live_rules()]
    assert types == ["tail_match", "chunk_repeat", "punct_spam"]


def test_evaluate_first_hit_wins_and_order_matters():
    reset_live_rules([
        # punct_spam FIRST (custom order) — a punct run should be reported
        # as punct_spam, not tail_match.
        {"rule_id": "p1", "type": "punct_spam", "enabled": True,
         "params": {"min_chars": 10, "min_chunks": 6}},
        {"rule_id": "t1", "type": "tail_match", "enabled": True,
         "params": {"min_match": 5, "min_len": 5, "min_distinct": 2}},
    ])
    hit, reason = loop_rules.evaluate_loop_rules(["!!!!!"] * 3)  # 15-char punct run
    assert hit is True
    assert reason.startswith("punct_spam")

    # Same window with punct_spam disabled → falls through to tail_match? No:
    # '!!!!!' x3 is only 3 chunks (tail needs 5), so it should NOT trigger.
    reset_live_rules([
        {"rule_id": "p1", "type": "punct_spam", "enabled": False,
         "params": {"min_chars": 10, "min_chunks": 6}},
        {"rule_id": "t1", "type": "tail_match", "enabled": True,
         "params": {"min_match": 5, "min_len": 5, "min_distinct": 2}},
    ])
    hit, _ = loop_rules.evaluate_loop_rules(["!!!!!"] * 3)
    assert hit is False


def test_custom_tail_match_rule():
    reset_live_rules([
        {"rule_id": "c1", "type": "tail_match", "enabled": True,
         "params": {"min_match": 3, "min_len": 4, "min_distinct": 2}},
    ])
    # 3 identical chunks of 8 distinct-ish chars → custom rule fires at 3.
    hit, reason = loop_rules.evaluate_loop_rules(["hello wo"] * 3)
    assert hit is True
    assert "tail_match" in reason


def test_disabled_rule_is_skipped():
    reset_live_rules([
        {"rule_id": "t1", "type": "tail_match", "enabled": False,
         "params": {"min_match": 5, "min_len": 5, "min_distinct": 2}},
    ])
    hit, _ = loop_rules.evaluate_loop_rules(["这是一个足够长的重复句子"] * 5)
    assert hit is False


def test_unknown_type_is_skipped_not_fatal():
    reset_live_rules([
        {"rule_id": "x", "type": "not_a_real_type", "enabled": True, "params": {}},
        {"rule_id": "t1", "type": "tail_match", "enabled": True,
         "params": {"min_match": 5, "min_len": 5, "min_distinct": 2}},
    ])
    # The unknown rule is skipped; the real one still works.
    hit, _ = loop_rules.evaluate_loop_rules(["这是一个足够长的重复句子"] * 5)
    assert hit is True


# ---- Persistence round-trip --------------------------------------------

@pytest.mark.asyncio
async def test_apply_and_reload_roundtrip(tmp_path):
    db_path = str(tmp_path / "t.db")
    from vllm_metrics_proxy.db import init_db
    await init_db(db_path)

    reset_live_rules()
    await loop_rules.load_rules_from_db(db_path)  # seeds defaults + persists
    assert [r["type"] for r in loop_rules.get_live_rules()] == \
        ["tail_match", "chunk_repeat", "punct_spam"]

    # Apply a custom-ordered list with an added rule.
    new_list = [
        {"type": "punct_spam", "enabled": True,
         "params": {"min_chars": 8, "min_chunks": 5}},
        {"type": "tail_match", "enabled": True,
         "params": {"min_match": 4, "min_len": 6, "min_distinct": 2}},
        {"type": "chunk_repeat", "enabled": False,
         "params": {"threshold": 3, "min_len": 10, "recent": 10}},
    ]
    normalised = await loop_rules.apply_rules(db_path, new_list)
    assert [r["type"] for r in normalised] == ["punct_spam", "tail_match", "chunk_repeat"]
    assert normalised[0]["params"]["min_chars"] == 8
    assert normalised[2]["enabled"] is False
    # rule_id assigned where absent
    for r in normalised:
        assert r["rule_id"]

    # Simulate a restart: clear live list, reload from DB.
    reset_live_rules()
    await loop_rules.load_rules_from_db(db_path)
    reloaded = loop_rules.get_live_rules()
    assert [r["type"] for r in reloaded] == ["punct_spam", "tail_match", "chunk_repeat"]
    assert reloaded[0]["params"]["min_chars"] == 8
    assert reloaded[2]["enabled"] is False


@pytest.mark.asyncio
async def test_apply_rules_validates_params(tmp_path):
    db_path = str(tmp_path / "t.db")
    from vllm_metrics_proxy.db import init_db
    await init_db(db_path)

    reset_live_rules()
    # Out-of-range param → ValueError, live list untouched.
    with pytest.raises(ValueError):
        await loop_rules.apply_rules(db_path, [
            {"type": "tail_match", "enabled": True,
             "params": {"min_match": 999, "min_len": 5, "min_distinct": 2}},
        ])
    # Unknown type → ValueError.
    with pytest.raises(ValueError):
        await loop_rules.apply_rules(db_path, [{"type": "bogus", "enabled": True, "params": {}}])
    # Live list still has whatever it had before (seeded defaults after first init? no —
    # we reset to empty and never applied successfully, so it's empty).
    assert loop_rules.get_live_rules() == []


# ---- API tests ----------------------------------------------------------

@pytest_asyncio.fixture
async def app_and_db(tmp_path):
    db_path = str(tmp_path / "t.db")
    from vllm_metrics_proxy.db import init_db
    await init_db(db_path)
    app = create_app(settings_override=make_settings(), db_path=db_path)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        headers = {"X-Admin-Token": create_admin_token("testpw")}
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client, db_path, headers


@pytest.mark.asyncio
async def test_api_get_loop_rules(app_and_db):
    _app, client, _db, headers = app_and_db
    resp = await client.get("/api/loop-rules", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert [r["type"] for r in data["rules"]] == ["tail_match", "chunk_repeat", "punct_spam"]
    assert set(data["types"]) == {"tail_match", "chunk_repeat", "punct_spam"}


@pytest.mark.asyncio
async def test_api_put_loop_rules_appends_and_persists(app_and_db):
    _app, client, db, headers = app_and_db
    base = await client.get("/api/loop-rules", headers=headers)
    rules = base.json()["rules"]
    # Append a new punct_spam variant.
    rules.append({
        "rule_id": "p_extra",
        "type": "punct_spam",
        "enabled": True,
        "params": {"min_chars": 7, "min_chunks": 4},
    })
    resp = await client.put("/api/loop-rules", json=rules, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["count"] == 4
    # Verify live list + DB persistence.
    live = loop_rules.get_live_rules()
    assert len(live) == 4
    assert live[3]["type"] == "punct_spam"
    assert live[3]["params"]["min_chars"] == 7

    # Persistence: the settings table must contain the loop_rules blob.
    from vllm_metrics_proxy.db import get_setting
    raw = await get_setting(db, "loop_rules")
    assert raw is not None
    stored = json.loads(raw)
    assert [r["type"] for r in stored] == [
        "tail_match", "chunk_repeat", "punct_spam", "punct_spam"]


@pytest.mark.asyncio
async def test_api_put_bad_params_rejected(app_and_db):
    _app, client, _db, headers = app_and_db
    resp = await client.put("/api/loop-rules", json=[
        {"type": "tail_match", "enabled": True,
         "params": {"min_match": 99999, "min_len": 5, "min_distinct": 2}},
    ], headers=headers)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_api_reset_loop_rules(app_and_db):
    _app, client, _db, headers = app_and_db
    # First add a rule, then reset.
    base = await client.get("/api/loop-rules", headers=headers)
    rules = base.json()["rules"]
    rules.append({"rule_id": "x", "type": "punct_spam", "enabled": True,
                  "params": {"min_chars": 5, "min_chunks": 3}})
    await client.put("/api/loop-rules", json=rules, headers=headers)
    assert len((await client.get("/api/loop-rules", headers=headers)).json()["rules"]) == 4

    resp = await client.post("/api/loop-rules/reset", headers=headers)
    assert resp.status_code == 200
    data = (await client.get("/api/loop-rules", headers=headers)).json()
    assert [r["type"] for r in data["rules"]] == ["tail_match", "chunk_repeat", "punct_spam"]


@pytest.mark.asyncio
async def test_api_requires_admin_token(app_and_db):
    _app, client, _db, _headers = app_and_db
    resp = await client.get("/api/loop-rules")  # no token
    assert resp.status_code in (401, 403)
