"""Smoke tests for deploy/signoz/ artifacts.

These tests run without any external services — they just validate that the
deploy files are syntactically valid and contain the expected structure.
"""

from __future__ import annotations
import os
import stat

import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNOZ_DIR = os.path.join(REPO_ROOT, "deploy", "signoz")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> object:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------


def test_docker_compose_parses_as_valid_yaml() -> None:
    path = os.path.join(SIGNOZ_DIR, "docker-compose.yml")
    doc = _load_yaml(path)
    assert isinstance(doc, dict), "docker-compose.yml must be a YAML mapping"


def test_docker_compose_has_expected_services() -> None:
    path = os.path.join(SIGNOZ_DIR, "docker-compose.yml")
    doc = _load_yaml(path)
    assert isinstance(doc, dict)
    services = doc.get("services", {})
    required = {
        "zookeeper-1",
        "clickhouse",
        "signoz",
        "otel-collector",
        "signoz-telemetrystore-migrator",
        "init-clickhouse",
    }
    missing = required - set(services.keys())
    assert not missing, f"Missing services in docker-compose.yml: {missing}"


def test_docker_compose_signoz_ui_port() -> None:
    """SigNoz UI should be on host port 3301 (avoids clash with app's :8080)."""
    path = os.path.join(SIGNOZ_DIR, "docker-compose.yml")
    doc = _load_yaml(path)
    assert isinstance(doc, dict)
    signoz_ports = doc["services"]["signoz"].get("ports", [])
    assert any("3301" in str(p) for p in signoz_ports), f"SigNoz UI port 3301 not found in: {signoz_ports}"


def test_docker_compose_otlp_grpc_port_exposed() -> None:
    """OTel collector must expose gRPC port 4317 for robot_comic traces."""
    path = os.path.join(SIGNOZ_DIR, "docker-compose.yml")
    doc = _load_yaml(path)
    assert isinstance(doc, dict)
    collector_ports = doc["services"]["otel-collector"].get("ports", [])
    assert any("4317" in str(p) for p in collector_ports), (
        f"OTLP gRPC port 4317 not found in otel-collector ports: {collector_ports}"
    )


def test_docker_compose_signoz_version_pinned() -> None:
    """SigNoz image should have a pinned version tag, not 'latest'."""
    path = os.path.join(SIGNOZ_DIR, "docker-compose.yml")
    doc = _load_yaml(path)
    assert isinstance(doc, dict)
    image = doc["services"]["signoz"].get("image", "")
    assert "latest" not in image, f"SigNoz image should be pinned, not 'latest': {image}"
    assert "signoz/signoz:" in image, f"Unexpected SigNoz image name: {image}"


# ---------------------------------------------------------------------------
# Config YAML files
# ---------------------------------------------------------------------------


def test_otel_collector_config_parses() -> None:
    path = os.path.join(SIGNOZ_DIR, "config", "otel-collector-config.yaml")
    doc = _load_yaml(path)
    assert isinstance(doc, dict)
    assert "receivers" in doc
    assert "exporters" in doc
    assert "service" in doc


def test_otel_collector_config_has_otlp_receiver() -> None:
    path = os.path.join(SIGNOZ_DIR, "config", "otel-collector-config.yaml")
    doc = _load_yaml(path)
    assert isinstance(doc, dict)
    receivers = doc.get("receivers", {})
    assert "otlp" in receivers, "OTel collector config must include an 'otlp' receiver"


def test_otel_collector_opamp_config_parses() -> None:
    path = os.path.join(SIGNOZ_DIR, "config", "otel-collector-opamp-config.yaml")
    doc = _load_yaml(path)
    assert isinstance(doc, dict)
    assert "server_endpoint" in doc


# ---------------------------------------------------------------------------
# Shell scripts — existence and executable bit
# ---------------------------------------------------------------------------


def test_install_sh_exists() -> None:
    path = os.path.join(SIGNOZ_DIR, "install.sh")
    assert os.path.isfile(path), f"install.sh not found at {path}"


def test_uninstall_sh_exists() -> None:
    path = os.path.join(SIGNOZ_DIR, "uninstall.sh")
    assert os.path.isfile(path), f"uninstall.sh not found at {path}"


def _git_mode(path: str) -> int:
    """Return the git index mode for a file (works on Windows where os.stat won't show +x)."""
    import subprocess

    rel = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
    result = subprocess.run(
        ["git", "ls-files", "-s", rel],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    # Output: "<mode> <hash> <stage>\t<path>"
    line = result.stdout.strip()
    if not line:
        return 0
    return int(line.split()[0], 8)


def test_install_sh_is_executable() -> None:
    path = os.path.join(SIGNOZ_DIR, "install.sh")
    if os.name == "nt":
        # Windows filesystem has no POSIX execute bits; check git index instead.
        mode = _git_mode(path)
        assert mode & stat.S_IXUSR, f"install.sh is not marked +x in git (git mode={oct(mode)})"
    else:
        mode = os.stat(path).st_mode
        assert mode & stat.S_IXUSR, f"install.sh is not user-executable (mode={oct(mode)})"


def test_uninstall_sh_is_executable() -> None:
    path = os.path.join(SIGNOZ_DIR, "uninstall.sh")
    if os.name == "nt":
        mode = _git_mode(path)
        assert mode & stat.S_IXUSR, f"uninstall.sh is not marked +x in git (git mode={oct(mode)})"
    else:
        mode = os.stat(path).st_mode
        assert mode & stat.S_IXUSR, f"uninstall.sh is not user-executable (mode={oct(mode)})"


def test_install_sh_references_docker_compose() -> None:
    path = os.path.join(SIGNOZ_DIR, "install.sh")
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "docker compose up" in content, "install.sh must call 'docker compose up'"


def test_uninstall_sh_removes_volumes() -> None:
    path = os.path.join(SIGNOZ_DIR, "uninstall.sh")
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "docker compose down -v" in content, "uninstall.sh must call 'docker compose down -v' to remove volumes"


# ---------------------------------------------------------------------------
# XML config files — existence only (not parsed, to avoid lxml dep)
# ---------------------------------------------------------------------------


def test_clickhouse_config_xml_exists() -> None:
    path = os.path.join(SIGNOZ_DIR, "config", "clickhouse-config.xml")
    assert os.path.isfile(path), f"clickhouse-config.xml not found at {path}"


def test_clickhouse_users_xml_exists() -> None:
    path = os.path.join(SIGNOZ_DIR, "config", "clickhouse-users.xml")
    assert os.path.isfile(path), f"clickhouse-users.xml not found at {path}"


def test_clickhouse_cluster_xml_exists() -> None:
    path = os.path.join(SIGNOZ_DIR, "config", "clickhouse-cluster.xml")
    assert os.path.isfile(path), f"clickhouse-cluster.xml not found at {path}"


def test_clickhouse_pi_memory_tuning_present() -> None:
    """Pi-tuned config must cap ClickHouse RAM to ≤ 50% of system RAM."""
    path = os.path.join(SIGNOZ_DIR, "config", "clickhouse-config.xml")
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "max_server_memory_usage_to_ram_ratio" in content, (
        "clickhouse-config.xml must set max_server_memory_usage_to_ram_ratio for Pi tuning"
    )
    # Ratio must be ≤ 0.5 for Pi 5 suitability
    import re

    matches = re.findall(
        r"<max_server_memory_usage_to_ram_ratio>([\d.]+)</max_server_memory_usage_to_ram_ratio>", content
    )
    assert matches, "Could not parse max_server_memory_usage_to_ram_ratio value"
    ratio = float(matches[0])
    assert ratio <= 0.5, f"ClickHouse memory ratio {ratio} is too high for Pi 5 (should be ≤ 0.5)"
