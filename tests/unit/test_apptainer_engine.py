"""Unit tests for the Apptainer engine adapters (STRATEGY M2)."""

from __future__ import annotations

import pytest

from multiverse.apptainer import (
    InMemoryApptainerEngine,
    classify_image_ref,
    compute_sif_digest,
)
from multiverse.apptainer.images import ApptainerImageKind
from multiverse.docker_supervisor.client import ContainerEngine, ContainerState
from multiverse.docker_supervisor.errors import NoSuchContainerError


pytestmark = pytest.mark.control_plane


# ---------------------------------------------------------------------------
# image classification
# ---------------------------------------------------------------------------


def test_classify_oci_registry():
    ref = classify_image_ref("docker://registry/foo:tag")
    assert ref.kind is ApptainerImageKind.OCI_REGISTRY
    assert ref.locator == "docker://registry/foo:tag"


def test_classify_sif_file_by_extension(tmp_path):
    sif = tmp_path / "x.sif"
    sif.write_bytes(b"sif")
    ref = classify_image_ref(str(sif))
    assert ref.kind is ApptainerImageKind.SIF_FILE


def test_classify_bare_tag_falls_back_to_docker_daemon():
    ref = classify_image_ref("myimage:v1")
    assert ref.kind is ApptainerImageKind.DOCKER_DAEMON
    assert ref.locator == "docker-daemon://myimage:v1"


def test_classify_empty_rejected():
    with pytest.raises(ValueError):
        classify_image_ref("")


# ---------------------------------------------------------------------------
# compute_sif_digest
# ---------------------------------------------------------------------------


def test_sif_digest_is_content_addressed(tmp_path):
    a = tmp_path / "a.sif"
    b = tmp_path / "b.sif"
    a.write_bytes(b"identical")
    b.write_bytes(b"identical")
    assert compute_sif_digest(a) == compute_sif_digest(b)
    b.write_bytes(b"different")
    assert compute_sif_digest(a) != compute_sif_digest(b)


# ---------------------------------------------------------------------------
# InMemoryApptainerEngine satisfies ContainerEngine
# ---------------------------------------------------------------------------


def test_in_memory_engine_satisfies_protocol():
    e = InMemoryApptainerEngine()
    assert isinstance(e, ContainerEngine)


def test_launch_and_inspect_lifecycle():
    e = InMemoryApptainerEngine()
    info = e.launch(
        image="docker://foo:latest",
        labels={"multiverse.run_id": "r1", "multiverse.image_digest": "sha256:abc"},
    )
    assert info.state is ContainerState.RUNNING
    assert e.sif_digest_for(info.container_id) is not None
    assert e.source_oci_digest_for(info.container_id) == "sha256:abc"

    # natural exit
    e.simulate_natural_exit(info.container_id, exit_code=0)
    again = e.inspect(info.container_id)
    assert again.state is ContainerState.EXITED
    assert again.exit_code == 0


def test_list_by_labels_filters_removed():
    e = InMemoryApptainerEngine()
    a = e.launch(image="img:1", labels={"k": "v"})
    b = e.launch(image="img:1", labels={"k": "v"})
    assert {c.container_id for c in e.list_by_labels(labels={"k": "v"})} == {
        a.container_id,
        b.container_id,
    }
    e.remove(a.container_id)
    assert {c.container_id for c in e.list_by_labels(labels={"k": "v"})} == {
        b.container_id
    }


def test_kill_reports_137():
    e = InMemoryApptainerEngine()
    info = e.launch(image="img:1")
    e.kill(info.container_id)
    finished = e.inspect(info.container_id)
    assert finished.state is ContainerState.EXITED
    assert finished.exit_code == 137


def test_inspect_unknown_raises():
    e = InMemoryApptainerEngine()
    with pytest.raises(NoSuchContainerError):
        e.inspect("nope")


def test_oom_simulation():
    e = InMemoryApptainerEngine()
    info = e.launch(image="img:1")
    e.simulate_natural_exit(info.container_id, exit_code=137, oom_killed=True)
    after = e.inspect(info.container_id)
    assert after.oom_killed is True


# ---------------------------------------------------------------------------
# sif digest is deterministic across engine instances
# ---------------------------------------------------------------------------


def test_sif_digest_deterministic_across_engines():
    a = InMemoryApptainerEngine()
    b = InMemoryApptainerEngine()
    ia = a.launch(image="docker://foo", labels={"multiverse.image_digest": "sha256:xyz"})
    ib = b.launch(image="docker://foo", labels={"multiverse.image_digest": "sha256:xyz"})
    assert a.sif_digest_for(ia.container_id) == b.sif_digest_for(ib.container_id)
