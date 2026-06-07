"""Post-training evaluation metrics for integration model outputs.

Heavy scientific dependencies (muon, anndata, h5py, numpy, scib-metrics) are
imported lazily so ``import multiverse.evaluate`` succeeds in a thin host
environment.  They are only required when evaluation functionality is actually
called.  Install them with:

    pip install "multiverse[eval]"

The evaluator runs one grouped scIB benchmark per dataset (efficient — all of a
dataset's model embeddings are scored in a single pass) and then fans the
results back out to **per-member** result files, one structured outcome per
cohort member, under the launch directory. See
:mod:`multiverse.evaluation.result` for the status vocabulary and schema.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from multiverse.evaluation import result as eval_result
from multiverse.evaluation.cohort import launch_dir
from multiverse.evaluation.result import MemberResult

logger = logging.getLogger(__name__)

_EVAL_INSTALL_HINT = (
    "Evaluation dependencies are not installed. "
    "Run: pip install \"multiverse[eval]\""
)


def _require_eval_deps():
    """Raise a helpful ImportError if eval dependencies are absent."""
    missing = []
    for pkg in ("muon", "anndata", "h5py", "numpy", "scib_metrics"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing eval dependencies: {missing}. {_EVAL_INSTALL_HINT}"
        )


def _require_worker_deps():
    """Raise a helpful ImportError if worker dependencies are absent."""
    try:
        import multiverse.worker  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"multiverse.worker not available: {exc}. "
            "Run: pip install \"multiverse[eval]\""
        ) from exc


def _now_iso() -> str:
    """UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _extract_member_metrics(full_metrics: Dict[str, Any], latent_key: str) -> Dict[str, Any]:
    """Pull one embedding's metrics out of the grouped scIB results dict.

    ``full_metrics`` is ``results_df.to_dict("dict")`` — a nested mapping whose
    orientation depends on the scIB version (latent keys may be the outer or the
    inner keys). This handles both: if ``latent_key`` is an outer key, return its
    sub-dict; otherwise collect the inner value keyed by ``latent_key`` from each
    metric column.
    """
    if latent_key in full_metrics and isinstance(full_metrics[latent_key], dict):
        return dict(full_metrics[latent_key])
    out: Dict[str, Any] = {}
    for metric, by_embedding in full_metrics.items():
        if isinstance(by_embedding, dict) and latent_key in by_embedding:
            out[metric] = by_embedding[latent_key]
    return out


class Evaluator:
    """Evaluate a dataset's model embeddings with scIB metrics.

    Each ``model_configs`` entry must carry ``member_id``, ``model_name`` and
    ``output_embedding_path``. Embeddings are stored in ``obsm`` keyed by
    ``X_<member_id>`` so that two members sharing a model (different params) on
    the same dataset never collide, and each results row maps to exactly one
    member.
    """

    def __init__(
        self,
        dataset,
        dataset_name: str,
        model_configs: List[Dict[str, Any]],
        output_dir: str,
    ):
        _require_eval_deps()
        logger.info("Initializing Evaluator")
        self.model_configs = model_configs
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.metrics_filepath = os.path.join(self.output_dir, "evaluation_metrics.json")
        os.makedirs(self.output_dir, exist_ok=True)
        # member_id -> latent key (X_<member_id>) for members that loaded cleanly.
        self.member_latent_key: Dict[str, str] = {}
        # member_id -> (status, reason) for members that failed pre-flight.
        self.member_preflight: Dict[str, "tuple[str, str]"] = {}
        logger.info(f"Evaluator initialized for {self.dataset_name}")

    def latent_key_for(self, member_id: str) -> str:
        return f"X_{member_id}"

    def load_embeddings(self) -> None:
        """Load each member's latent embedding, recording pre-flight failures.

        Detects missing ``embeddings.h5`` (``no_embeddings``) and latent/obs row
        mismatches (``obs_mismatch``) up front rather than letting anndata raise
        an opaque error mid-benchmark.
        """
        import h5py  # type: ignore[import-untyped]

        n_obs = self.dataset.n_obs
        for cfg in self.model_configs:
            member_id = cfg["member_id"]
            emb_path = cfg["output_embedding_path"]
            if not os.path.exists(emb_path):
                logger.warning("Embeddings not found for member %s at %s", member_id, emb_path)
                self.member_preflight[member_id] = (
                    eval_result.EVAL_STATUS_NO_EMBEDDINGS,
                    f"embeddings.h5 not found at {emb_path}",
                )
                continue
            try:
                with h5py.File(emb_path, "r") as f:
                    latent = f["latent"]
                    if latent.shape[0] != n_obs:
                        self.member_preflight[member_id] = (
                            eval_result.EVAL_STATUS_OBS_MISMATCH,
                            f"latent rows ({latent.shape[0]}) != dataset observations ({n_obs})",
                        )
                        continue
                    data = latent[:]
            except Exception as exc:  # noqa: BLE001
                self.member_preflight[member_id] = (
                    eval_result.EVAL_STATUS_EVALUATION_FAILED,
                    f"could not read embeddings: {exc}",
                )
                continue
            latent_key = self.latent_key_for(member_id)
            self.dataset.obsm[latent_key] = data
            self.member_latent_key[member_id] = latent_key
        logger.info("Loaded latent embeddings for: %s", list(self.member_latent_key.values()))

    def evaluate_models(
        self, batch_key: str = "batch", label_key: str = "cell_type"
    ) -> Dict[str, Any]:
        """Run scIB on all cleanly-loaded embeddings; return grouped metrics dict.

        Returns ``results_df.to_dict("dict")`` (sanitized), or ``{}`` if there is
        nothing to score. Per-member attribution is done by the caller via
        :func:`_extract_member_metrics`.
        """
        import numpy as np  # type: ignore[import-untyped]
        from scib_metrics.benchmark import (  # type: ignore[import-untyped]
            BatchCorrection, Benchmarker, BioConservation)

        from multiverse.worker import sanitize_nan_inf

        latent_keys = list(self.member_latent_key.values())
        if not latent_keys:
            logger.warning("No embeddings loaded for %s; skipping benchmark.", self.dataset_name)
            return {}

        logger.info("Evaluating model with scib-metrics.")
        if (
            batch_key not in self.dataset.obs.columns
            or self.dataset.obs[batch_key].nunique() < 2
        ):
            logger.warning(
                f"Batch key '{batch_key}' not found in .obs, assigning dummy batch labels."
            )
            rng = np.random.default_rng()
            self.dataset.obs[batch_key] = rng.choice(
                [f"batch_{i}" for i in range(10)], size=self.dataset.n_obs
            )

        if (
            label_key not in self.dataset.obs.columns
            or self.dataset.obs[label_key].nunique() < 2
        ):
            logger.warning(
                f"Label key '{label_key}' not found in .obs, assigning dummy cell type labels."
            )
            rng = np.random.default_rng()
            self.dataset.obs[label_key] = rng.choice(
                [f"cell_type_{i}" for i in range(10)], size=self.dataset.n_obs
            )

        bm = Benchmarker(
            self.dataset,
            progress_bar=False,
            batch_key=batch_key,
            label_key=label_key,
            embedding_obsm_keys=latent_keys,
            bio_conservation_metrics=BioConservation(
                isolated_labels=True,
                nmi_ari_cluster_labels_leiden=True,
                nmi_ari_cluster_labels_kmeans=True,
                silhouette_label=True,
                clisi_knn=True,
            ),
            batch_correction_metrics=BatchCorrection(
                bras=True,
                ilisi_knn=True,
                kbet_per_label=True,
                graph_connectivity=True,
                pcr_comparison=True,
            ),
        )

        bm.benchmark()
        results_df = bm.get_results(min_max_scale=False)
        bm.plot_results_table(min_max_scale=False, show=False, save_dir=self.output_dir)
        if results_df.empty:
            logger.warning(f"No results found for {self.dataset_name}.")
            return {}

        metrics = sanitize_nan_inf(results_df.to_dict("dict"))
        try:
            with open(self.metrics_filepath, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {self.metrics_filepath}")
        except IOError as e:
            logger.error(f"Could not write metrics file to {self.metrics_filepath}: {e}")
        return metrics


def _group_members_by_dataset(
    members: List[Dict[str, Any]]
) -> "tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]":
    """Group cohort members by dataset slug.

    Returns ``(members_by_dataset, dataset_cfg_by_dataset)`` where the dataset
    config carries the resolved path and the batch/label keys.
    """
    members_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    dataset_cfg: Dict[str, Dict[str, Any]] = {}
    for member in members:
        dataset_name = member.get("dataset_slug") or member.get("dataset_name") or ""
        members_by_dataset.setdefault(dataset_name, []).append(member)
        if dataset_name not in dataset_cfg:
            dataset_cfg[dataset_name] = {
                "data_path": member.get("dataset_path_resolved") or member.get("dataset_path"),
                "batch_key": member.get("batch_key", "batch"),
                "label_key": member.get("label_key", "cell_type"),
                "modalities": (member.get("job") or {}).get("omics_available"),
            }
    return members_by_dataset, dataset_cfg


def _emb_path(member: Dict[str, Any]) -> str:
    return os.path.join(member.get("artifact_dir") or "", "embeddings.h5")


def evaluate_cohort(config: Dict[str, Any], *, force: bool = False) -> List[MemberResult]:
    """Evaluate every member in ``config`` and write per-member result files.

    ``config`` is the trimmed eval cohort written by
    :func:`multiverse.evaluation.docker_runner.prepare_evaluation`. Returns the
    list of :class:`MemberResult` written (also persisted under
    ``<output_dir>/.multiverse/launches/<launch_id>/evaluations/``).

    Re-run is idempotent: a member whose ``evaluations/<member_id>.json`` already
    records ``done`` for the same artifact dir is preserved and not re-evaluated,
    unless ``force`` is set.
    """
    import muon as mu  # type: ignore[import-untyped]

    from multiverse.worker import anndata_concatenate, preprocess_mudata

    output_dir = Path(config["output_dir"])
    launch_id = config["launch_id"]
    members: List[Dict[str, Any]] = config.get("members", [])

    results: Dict[str, MemberResult] = {}
    started: Dict[str, float] = {}

    # Existing results enable idempotent re-run: keep already-``done`` members
    # (same artifact dir) instead of recomputing them.
    existing = {res.member_id: res for res in eval_result.load_member_results(output_dir, launch_id)}

    def _already_done(member: Dict[str, Any]) -> bool:
        if force:
            return False
        prev = existing.get(member["member_id"])
        return (
            prev is not None
            and prev.status == eval_result.EVAL_STATUS_DONE
            and (prev.artifact_dir or "") == (member.get("artifact_dir") or "")
        )

    to_eval: List[Dict[str, Any]] = []
    for member in members:
        mid = member["member_id"]
        if _already_done(member):
            results[mid] = existing[mid]  # preserve prior outcome
            continue
        to_eval.append(member)
        started[mid] = time.monotonic()
        res = MemberResult(
            member_id=mid,
            status=eval_result.EVAL_STATUS_RUNNING,
            artifact_dir=member.get("artifact_dir"),
            dataset_path=member.get("dataset_path_resolved") or member.get("dataset_path"),
            started_at=_now_iso(),
        )
        results[mid] = res
        eval_result.write_member_result(output_dir=output_dir, launch_id=launch_id, result=res)

    members_by_dataset, dataset_cfg = _group_members_by_dataset(to_eval)

    for dataset_name, ds_members in members_by_dataset.items():
        cfg = dataset_cfg[dataset_name]
        batch_key = cfg["batch_key"]
        label_key = cfg["label_key"]
        data_path = cfg["data_path"]

        # Per-dataset load/preprocess. A failure here fails every member of the
        # dataset with a structured status rather than aborting the run.
        try:
            mudata_obj = mu.read_h5mu(data_path)
            modalities = list(mudata_obj.mod.keys())
            mudata_obj = preprocess_mudata(
                mudata_obj,
                {
                    "n_top_genes": 2000,
                    "scale": {modality: False for modality in modalities},
                    "normalization_target_sum": None,
                    "log_normalization": False,
                },
                cell_type_key=label_key,
                batch_key=batch_key,
            )
            data_concat = anndata_concatenate(
                mdata=mudata_obj,
                selected_modalities=modalities,
                cell_type_key=label_key,
                batch_key=batch_key,
            )
        except FileNotFoundError as exc:
            _fail_members(
                results, ds_members, started, output_dir, launch_id,
                eval_result.EVAL_STATUS_MISSING_DATASET, str(exc),
            )
            continue
        except Exception as exc:  # noqa: BLE001
            _fail_members(
                results, ds_members, started, output_dir, launch_id,
                eval_result.EVAL_STATUS_EVALUATION_FAILED,
                f"dataset load/preprocess failed: {exc}",
            )
            continue

        model_configs = [
            {
                "member_id": m["member_id"],
                "model_name": m.get("model_slug") or "",
                "output_embedding_path": _emb_path(m),
            }
            for m in ds_members
        ]
        plot_dir = str(launch_dir(output_dir, launch_id) / "plots" / f"dataset_{dataset_name}")
        evaluator = Evaluator(
            dataset=data_concat,
            dataset_name=dataset_name,
            model_configs=model_configs,
            output_dir=plot_dir,
        )
        evaluator.load_embeddings()
        try:
            full_metrics = evaluator.evaluate_models(batch_key=batch_key, label_key=label_key)
            benchmark_error: Optional[str] = None
        except Exception as exc:  # noqa: BLE001
            full_metrics = {}
            benchmark_error = f"scIB benchmark raised: {exc}"

        for member in ds_members:
            mid = member["member_id"]
            if mid in evaluator.member_preflight:
                status, reason = evaluator.member_preflight[mid]
                _finalize(results, mid, started, output_dir, launch_id, status, reason)
                continue
            latent_key = evaluator.member_latent_key.get(mid)
            member_metrics = (
                _extract_member_metrics(full_metrics, latent_key) if latent_key else {}
            )
            if benchmark_error is not None:
                _finalize(
                    results, mid, started, output_dir, launch_id,
                    eval_result.EVAL_STATUS_EVALUATION_FAILED, benchmark_error,
                )
            elif member_metrics:
                _finalize(
                    results, mid, started, output_dir, launch_id,
                    eval_result.EVAL_STATUS_DONE, "",
                    metrics={"evaluation": member_metrics},
                )
            else:
                _finalize(
                    results, mid, started, output_dir, launch_id,
                    eval_result.EVAL_STATUS_EVALUATION_FAILED,
                    "scIB produced no metrics for this member",
                )

    return list(results.values())


def _finalize(
    results: Dict[str, MemberResult],
    member_id: str,
    started: Dict[str, float],
    output_dir: Path,
    launch_id: str,
    status: str,
    reason: str,
    *,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    res = results[member_id]
    res.status = status
    res.reason = reason
    res.finished_at = _now_iso()
    res.duration_seconds = round(time.monotonic() - started[member_id], 3)
    if metrics is not None:
        res.metrics = metrics
    if status in {
        eval_result.EVAL_STATUS_EVALUATION_FAILED,
        eval_result.EVAL_STATUS_OBS_MISMATCH,
        eval_result.EVAL_STATUS_NO_EMBEDDINGS,
        eval_result.EVAL_STATUS_MISSING_DATASET,
    } and reason:
        res.error = {"message": reason}
    eval_result.write_member_result(output_dir=output_dir, launch_id=launch_id, result=res)


def _fail_members(
    results: Dict[str, MemberResult],
    ds_members: List[Dict[str, Any]],
    started: Dict[str, float],
    output_dir: Path,
    launch_id: str,
    status: str,
    reason: str,
) -> None:
    for member in ds_members:
        _finalize(results, member["member_id"], started, output_dir, launch_id, status, reason)


def main():
    _require_eval_deps()
    _require_worker_deps()

    from multiverse.evaluation.cohort import resolve_cohort_readiness
    from multiverse.worker import load_config

    parser = argparse.ArgumentParser(description="Run cohort evaluation")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/app/config_alldatasets.json",
        help="Path to the trimmed evaluation config (eval_config.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate members even if a prior 'done' result exists.",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    member_results = evaluate_cohort(config, force=args.force)

    # Write a launch-level report from the members the container saw. The host
    # GUI rebuilds the authoritative report with live readiness on render.
    output_dir = Path(config["output_dir"])
    launch_id = config["launch_id"]
    members_with_status = resolve_cohort_readiness(config)
    report = eval_result.build_evaluation_report(
        cohort=config,
        members_with_status=members_with_status,
        member_results=member_results,
    )
    eval_result.write_evaluation_report(
        output_dir=output_dir, launch_id=launch_id, report=report
    )
    logger.info(
        "Evaluation complete: %s members, status counts %s",
        report["total"], report["status_counts"],
    )


if __name__ == "__main__":
    main()
