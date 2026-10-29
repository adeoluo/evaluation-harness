"""Tests for the dependency-free HTTP evaluation service (pure handler layer)."""

from __future__ import annotations

from evaluation_harness.service import EvaluationService, handle

_SERVICE = EvaluationService.build()


def test_health() -> None:
    status, payload = handle("GET", "/health", {}, _SERVICE)
    assert status == 200
    assert payload["status"] == "ok"
    assert "arithmetic" in payload["tasks"]
    assert "arithmetic-solver" in payload["models"]


def test_evaluate_reports_accuracy_with_interval() -> None:
    status, payload = handle(
        "POST", "/evaluate", {"model": "arithmetic-solver", "tasks": ["arithmetic"]}, _SERVICE
    )
    assert status == 200
    row = payload["results"][0]
    assert row["task"] == "arithmetic"
    assert row["count"] > 0
    assert row["ci_low"] <= row["accuracy"] <= row["ci_high"]


def test_compare_reports_significance() -> None:
    status, payload = handle(
        "POST",
        "/compare",
        {"model_a": "arithmetic-solver", "model_b": "constant-zero", "task": "arithmetic"},
        _SERVICE,
    )
    assert status == 200
    assert "p_value" in payload
    assert isinstance(payload["significant"], bool)
    assert payload["mean_difference"] != 0.0


def test_evaluate_rejects_unknown_model() -> None:
    status, payload = handle(
        "POST", "/evaluate", {"model": "nope", "tasks": ["arithmetic"]}, _SERVICE
    )
    assert status == 400
    assert "model" in payload["error"]


def test_evaluate_rejects_unknown_task() -> None:
    status, _ = handle("POST", "/evaluate", {"model": "echo", "tasks": ["nope"]}, _SERVICE)
    assert status == 400


def test_evaluate_rejects_bad_confidence() -> None:
    status, _ = handle(
        "POST", "/evaluate", {"model": "echo", "tasks": ["arithmetic"], "confidence": 1.5}, _SERVICE
    )
    assert status == 400


def test_compare_rejects_unknown_task() -> None:
    status, _ = handle(
        "POST",
        "/compare",
        {"model_a": "echo", "model_b": "constant-zero", "task": "nope"},
        _SERVICE,
    )
    assert status == 400


def test_unknown_route() -> None:
    status, payload = handle("GET", "/nope", {}, _SERVICE)
    assert status == 404
    assert "error" in payload
