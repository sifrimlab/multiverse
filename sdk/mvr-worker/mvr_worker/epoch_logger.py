"""Thin, framework-agnostic per-epoch metrics logger.

Streams metrics to MLflow (if reachable) and to a local JSONL sidecar so
crashed runs still keep partial curves. Designed to be copied or imported
as-is by model containers and external user code.

Minimal usage:

    from mvr_worker import EpochLogger

    with EpochLogger(jsonl_path="/output/metrics.jsonl", run_name="cobolt") as ep:
        for epoch in range(num_epochs):
            train_one_epoch(...)
            ep.log(step=epoch, loss=loss, val_loss=val_loss)

Framework adapters live at the bottom of this file as copy-pasteable
templates (Keras callback, PyTorch hook, scvi-style history replay).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EpochLogger:
    """Stream per-epoch metrics to MLflow + a local JSONL file.

    All arguments are optional. If MLflow is unavailable or no tracking URI
    is set, the logger silently falls back to JSONL-only mode.
    """

    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        run_name: Optional[str] = None,
        experiment: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        attach_active_run: bool = True,
    ) -> None:
        self._jsonl_path = jsonl_path
        self._fp = None
        self._mlflow = None
        self._owns_run = False

        self._try_attach_mlflow(
            run_name=run_name,
            experiment=experiment,
            tracking_uri=tracking_uri,
            attach_active_run=attach_active_run,
        )

        if jsonl_path:
            os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
            self._fp = open(jsonl_path, "a", encoding="utf-8")

    def _try_attach_mlflow(
        self,
        *,
        run_name: Optional[str],
        experiment: Optional[str],
        tracking_uri: Optional[str],
        attach_active_run: bool,
    ) -> None:
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if not uri:
            return
        try:
            import mlflow  # type: ignore
        except Exception as exc:
            logger.debug("mlflow import failed; JSONL-only logging. (%s)", exc)
            return

        try:
            mlflow.set_tracking_uri(uri)
            if experiment:
                mlflow.set_experiment(experiment)
            elif os.getenv("MLFLOW_EXPERIMENT_NAME"):
                mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
            if attach_active_run and mlflow.active_run() is not None:
                self._mlflow = mlflow
                return
            parent_run_id = os.getenv("MLFLOW_RUN_ID")
            if parent_run_id:
                mlflow.start_run(run_id=parent_run_id)
                self._mlflow = mlflow
                self._owns_run = False
                return
            mlflow.start_run(run_name=run_name)
            self._mlflow = mlflow
            self._owns_run = True
        except Exception as exc:
            logger.warning("EpochLogger MLflow attach failed: %s", exc)
            self._mlflow = None

    def log(self, step: int, **metrics: Any) -> None:
        """Record a row of per-epoch metrics. Non-finite values are dropped."""
        clean: Dict[str, float] = {}
        for key, value in metrics.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                clean[str(key)] = numeric
        if not clean:
            return

        ts_ms = int(time.time() * 1000)
        if self._fp is not None:
            self._fp.write(json.dumps({"step": int(step), "timestamp": ts_ms, **clean}) + "\n")
            self._fp.flush()

        if self._mlflow is not None:
            for name, value in clean.items():
                try:
                    self._mlflow.log_metric(name, value, step=int(step), timestamp=ts_ms)
                except Exception as exc:
                    logger.warning("MLflow log_metric failed for %s: %s", name, exc)

    def close(self, status: str = "FINISHED") -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            finally:
                self._fp = None
        if self._mlflow is not None and self._owns_run:
            try:
                self._mlflow.end_run(status=status)
            except Exception as exc:
                logger.warning("MLflow end_run failed: %s", exc)
        self._mlflow = None

    def __enter__(self) -> "EpochLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(status="FAILED" if exc_type else "FINISHED")


def replay_history(
    history: Dict[str, Any],
    *,
    output_dir: str,
    run_name: str,
) -> Dict[str, list]:
    """Sanitize a post-hoc training history dict and stream it through EpochLogger.

    Returns the cleaned history (floats, finite values only) so the caller can
    also write it into ``metrics.json``. Designed for frameworks (scvi-tools,
    Cobolt, Mowgli) that expose history only after ``.train()`` returns.
    """
    cleaned: Dict[str, list] = {}
    for key, values in (history or {}).items():
        if values is None:
            continue
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, (list, tuple)):
            continue
        series: list = []
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                series.append(numeric)
        if series:
            cleaned[str(key)] = series

    if cleaned:
        with EpochLogger(
            jsonl_path=os.path.join(output_dir, "metrics.jsonl"),
            run_name=run_name,
        ) as ep:
            length = max(len(v) for v in cleaned.values())
            for step in range(length):
                row = {k: v[step] for k, v in cleaned.items() if step < len(v)}
                ep.log(step=step, **row)
    return cleaned


# ---------------------------------------------------------------------------
# Reusable adapter templates. Copy into your container/run.py as needed.
# ---------------------------------------------------------------------------

# --- Keras / TensorFlow ----------------------------------------------------
#
# from tensorflow import keras
#
# class EpochLoggerKerasCallback(keras.callbacks.Callback):
#     def __init__(self, epoch_logger):
#         super().__init__()
#         self._ep = epoch_logger
#     def on_epoch_end(self, epoch, logs=None):
#         self._ep.log(step=epoch, **(logs or {}))
#
# Usage:
#     with EpochLogger(jsonl_path="/output/metrics.jsonl", run_name="my-run") as ep:
#         model.fit(..., callbacks=[EpochLoggerKerasCallback(ep)])
#
# --- PyTorch (manual loop) -------------------------------------------------
#
# with EpochLogger(jsonl_path="/output/metrics.jsonl", run_name="my-run") as ep:
#     for epoch in range(num_epochs):
#         train_loss = train_one_epoch(model, loader)
#         val_loss = evaluate(model, val_loader)
#         ep.log(step=epoch, train_loss=train_loss, val_loss=val_loss)
#
# --- scvi-tools / Cobolt (history replay after .train) ---------------------
#
# model.train(num_epochs=N)
# history = {k: list(map(float, v)) for k, v in model.history.items()}
# with EpochLogger(jsonl_path="/output/metrics.jsonl", run_name="my-run") as ep:
#     length = max((len(v) for v in history.values()), default=0)
#     for step in range(length):
#         row = {k: v[step] for k, v in history.items() if step < len(v)}
#         ep.log(step=step, **row)
