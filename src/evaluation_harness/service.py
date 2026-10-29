"""A dependency-free HTTP service exposing the evaluation harness.

The service evaluates built-in models on built-in tasks and compares two models
with a significance test, returning error-barred metrics. A pure handler is
separated from the socket binding for testing.

Endpoints
---------
``GET  /health``    liveness, task names, and model names.
``POST /evaluate``  body ``{"model": str, "tasks": [str], "confidence"?, "seed"?}``;
                    returns per-task accuracy with a confidence interval.
``POST /compare``   body ``{"model_a", "model_b", "task", "test"?, "seed"?}``;
                    returns both accuracies and a significance verdict.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from evaluation_harness.core import Runner, TaskResult
from evaluation_harness.models import available_models, build_model
from evaluation_harness.observability import ServiceMetrics
from evaluation_harness.significance import mcnemar_test, paired_bootstrap_test
from evaluation_harness.tasks import available_tasks, load_task

JsonObject = dict[str, Any]
_MAX_TASKS = 20


@dataclass(frozen=True)
class EvaluationService:
    """Evaluates and compares built-in models on built-in tasks."""

    tasks: tuple[str, ...]
    models: tuple[str, ...]
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def build(cls) -> EvaluationService:
        return cls(tasks=tuple(available_tasks()), models=tuple(available_models()))

    def evaluate(
        self, model_name: str, task_names: Sequence[str], *, confidence: float, seed: int
    ) -> JsonObject:
        model = build_model(model_name)
        runner = Runner(seed=seed)
        results: list[TaskResult] = [runner.run(model, load_task(name)) for name in task_names]
        return {
            "model": model_name,
            "results": [
                {
                    "task": name,
                    "count": result.count,
                    "accuracy": round(result.accuracy(confidence=confidence).mean, 6),
                    "ci_low": round(result.accuracy(confidence=confidence).low, 6),
                    "ci_high": round(result.accuracy(confidence=confidence).high, 6),
                }
                for name, result in zip(task_names, results, strict=True)
            ],
        }

    def compare(
        self, model_a: str, model_b: str, task_name: str, *, test: str, seed: int
    ) -> JsonObject:
        runner = Runner(seed=seed)
        result_a = runner.run(build_model(model_a), load_task(task_name))
        result_b = runner.run(build_model(model_b), load_task(task_name))
        comparison_test = mcnemar_test if test == "mcnemar" else paired_bootstrap_test
        comparison = comparison_test(result_a.scores, result_b.scores)
        return {
            "task": task_name,
            "count": result_a.count,
            "model_a": {"name": model_a, "accuracy": round(result_a.accuracy().mean, 6)},
            "model_b": {"name": model_b, "accuracy": round(result_b.accuracy().mean, 6)},
            "mean_difference": round(comparison.mean_difference, 6),
            "p_value": round(comparison.p_value, 6),
            "effect_size": round(comparison.effect_size, 6),
            "significant": comparison.significant,
            "test": comparison.test,
        }


def _validate_evaluate(
    body: Mapping[str, Any], service: EvaluationService
) -> tuple[str, list[str], float, int]:
    model = body.get("model")
    if not isinstance(model, str) or model not in service.models:
        raise ValueError(f"'model' must be one of: {', '.join(service.models)}")
    tasks = body.get("tasks")
    if not isinstance(tasks, list) or not tasks or len(tasks) > _MAX_TASKS:
        raise ValueError(f"'tasks' must be a non-empty list of at most {_MAX_TASKS} names")
    for name in tasks:
        if name not in service.tasks:
            raise ValueError(f"unknown task '{name}'")
    confidence = body.get("confidence", 0.95)
    if not isinstance(confidence, int | float) or not 0.0 < confidence < 1.0:
        raise ValueError("'confidence' must be in (0, 1)")
    seed = body.get("seed", 0)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("'seed' must be a non-negative integer")
    return model, tasks, float(confidence), seed


def _validate_compare(
    body: Mapping[str, Any], service: EvaluationService
) -> tuple[str, str, str, str, int]:
    for key in ("model_a", "model_b"):
        if body.get(key) not in service.models:
            raise ValueError(f"'{key}' must be one of: {', '.join(service.models)}")
    task = body.get("task")
    if task not in service.tasks:
        raise ValueError(f"'task' must be one of: {', '.join(service.tasks)}")
    test = body.get("test", "bootstrap")
    if test not in ("bootstrap", "mcnemar"):
        raise ValueError("'test' must be 'bootstrap' or 'mcnemar'")
    seed = body.get("seed", 0)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("'seed' must be a non-negative integer")
    return str(body["model_a"]), str(body["model_b"]), str(task), test, seed


def _route(
    method: str, path: str, body: Mapping[str, Any], service: EvaluationService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "tasks": list(service.tasks), "models": list(service.models)}
    if method == "POST" and path == "/evaluate":
        try:
            model, tasks, confidence, seed = _validate_evaluate(body, service)
        except ValueError as error:
            return 400, {"error": str(error)}
        return 200, service.evaluate(model, tasks, confidence=confidence, seed=seed)
    if method == "POST" and path == "/compare":
        try:
            model_a, model_b, task, test, seed = _validate_compare(body, service)
        except ValueError as error:
            return 400, {"error": str(error)}
        return 200, service.compare(model_a, model_b, task, test=test, seed=seed)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: EvaluationService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready"}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: EvaluationService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": EvaluationService.build()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"evaluation-harness service listening on http://{host}:{port}")
    server.serve_forever()
