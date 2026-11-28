"""Companion package discovery and compatibility checking."""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import annotations

import os
import re
import sys
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from jupyter_client.kernelspec import KernelSpecManager
from jupyter_core.utils import ensure_async
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


async def get_all_kernelspecs(
    manager: KernelSpecManager | None = None,
) -> dict[str, dict[str, Any]]:
    """Return available kernelspecs keyed by kernelspec name."""
    try:
        manager = manager or KernelSpecManager()
        specs = await ensure_async(manager.get_all_specs())
        if type(manager) is not KernelSpecManager:
            # Custom managers may resolve kernels on a different host (e.g. a gateway).
            specs = {name: {**info, "_companion_local": False} for name, info in specs.items()}
        return specs
    except Exception:
        return {}


def _parse_package_entry(raw: object) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    entry: dict[str, str] = {"name": name.strip()}
    version = raw.get("version")
    if isinstance(version, str) and version.strip():
        entry["version"] = version.strip()
    return entry


def _parse_overrides(raw: object, base: dict[str, str]) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, dict[str, str]] = {}
    for manager, value in raw.items():
        if not isinstance(manager, str):
            continue
        entry = _parse_package_entry({**base, **value}) if isinstance(value, dict) else None
        if entry is not None:
            parsed[manager] = entry
    return parsed


def _parse_managers(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item.strip()]


def _parse_kernel_spec_filter(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for key in ("language", "display_name"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            result[key] = value
    return result


def _candidate_names(base: dict[str, str], overrides: dict[str, dict[str, str]]) -> list[str]:
    pip_override = overrides.get("pip")
    return [(pip_override or base)["name"]]


def _required_version(base: dict[str, str], overrides: dict[str, dict[str, str]]) -> str | None:
    pip_override = overrides.get("pip")
    if pip_override is not None and "version" in pip_override:
        return pip_override["version"]
    return base.get("version")


def _find_installed_version(
    package_names: list[str], search_paths: list[str] | None = None
) -> str | None:
    for name in package_names:
        try:
            if search_paths is None:
                return distribution(name).version
            dist = next(iter(Distribution.discover(name=name, path=search_paths)), None)
            if dist is not None:
                return dist.version
        except (PackageNotFoundError, OSError, ValueError):
            continue
    return None


def _check_version(installed_version: str | None, required_version: str | None) -> dict[str, Any]:
    try:
        specifiers = SpecifierSet(required_version or "")
    except InvalidSpecifier:
        return {
            "status": "unknown",
            "installedVersion": installed_version,
            "requiredVersion": required_version,
            "message": f"Invalid version specifier: {required_version}",
        }

    if installed_version is None:
        return {
            "status": "missing",
            "installedVersion": None,
            "requiredVersion": required_version,
            "message": None,
        }

    if not required_version:
        return {
            "status": "ok",
            "installedVersion": installed_version,
            "requiredVersion": None,
            "message": None,
        }

    try:
        parsed_version = Version(installed_version)
    except InvalidVersion:
        return {
            "status": "unknown",
            "installedVersion": installed_version,
            "requiredVersion": required_version,
            "message": f"Invalid installed version: {installed_version}",
        }

    if specifiers.contains(parsed_version, prereleases=True):
        return {
            "status": "ok",
            "installedVersion": installed_version,
            "requiredVersion": required_version,
            "message": None,
        }

    return {
        "status": "incompatible",
        "installedVersion": installed_version,
        "requiredVersion": required_version,
        "message": f"Installed version {installed_version} does not satisfy {required_version}",
    }


def _resolve_python_site_packages(spec: dict[str, Any]) -> tuple[bool, list[str] | None]:
    """Determine whether a kernelspec's Python environment can be inspected on disk."""
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
        return False, None

    # Custom launchers, provisioners and environment overrides may use a different sys.path.
    metadata = spec.get("metadata") or {}
    if (
        not isinstance(metadata, dict)
        or metadata.get("kernel_provisioner")
        or spec.get("env")
        or argv[1:3] != ["-m", "ipykernel_launcher"]
    ):
        return False, None

    executable = argv[0]
    exe_name = Path(executable).name.lower()
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?(?:\.exe)?", exe_name):
        return False, None

    # Match jupyter_client's substitution, which does not resolve these names on PATH.
    if executable in {
        "python",
        f"python{sys.version_info.major}",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
    }:
        executable = sys.executable
    exe_path = Path(executable)
    if not exe_path.is_absolute() or not exe_path.is_file():
        return False, None

    if os.path.normcase(os.path.abspath(exe_path)) == os.path.normcase(
        os.path.abspath(sys.executable)
    ):
        return True, None

    paths = _isolated_site_packages(exe_path)
    return (True, paths) if paths else (False, None)


def _isolated_site_packages(exe_path: Path) -> list[str] | None:
    """Inspect only a standard venv whose package search path is unambiguous."""
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        return None
    root = (
        exe_path.parent.parent
        if exe_path.parent.name.lower() in {"bin", "scripts"}
        else exe_path.parent
    )
    try:
        config = dict(
            line.lower().strip().split("=", 1)
            for line in (root / "pyvenv.cfg").read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
        config = {key.strip(): value.strip() for key, value in config.items()}
        if config.get("include-system-site-packages") != "false":
            return None
        version = config.get("version", "")
        match = re.match(r"^(\d+\.\d+)\.", version)
        if match is None:
            return None
        site = (
            root / "Lib" / "site-packages"
            if os.name == "nt"
            else (root / "lib" / f"python{match[1]}" / "site-packages")
        )
        # Path files can add inherited/editable distributions or execute startup code.
        if not site.is_dir() or any(site.glob("*.pth")) or (site / "sitecustomize.py").exists():
            return None
    except (OSError, ValueError):
        return None
    return [str(site)]


def _matches_pattern(pattern: str | None, value: object) -> bool:
    if not pattern:
        return True
    if not isinstance(value, str):
        return False
    try:
        return bool(re.search(pattern, value))
    except re.error:
        return False


def _matches_kernelspec(kernel_spec_filter: dict[str, str], spec: dict[str, Any]) -> bool:
    return _matches_pattern(kernel_spec_filter.get("language"), spec.get("language")) and (
        _matches_pattern(kernel_spec_filter.get("display_name"), spec.get("display_name"))
    )


def _inspect_server_companion(raw_server: object) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(raw_server, dict):
        return None, False
    base = _parse_package_entry(raw_server.get("base"))
    if base is None:
        return None, False
    overrides = _parse_overrides(raw_server.get("overrides"), base)
    managers = _parse_managers(raw_server.get("managers"))
    required = _required_version(base, overrides)
    installed = _find_installed_version(_candidate_names(base, overrides))
    check = _check_version(installed, required)
    has_issue = check["status"] in ("missing", "incompatible")
    server_entry: dict[str, Any] = {
        "base": base,
        "managers": managers,
        "check": check,
    }
    if overrides:
        server_entry["overrides"] = overrides
    return server_entry, has_issue


def _inspect_kernel_entry(
    raw_kernel: dict[str, Any], specs: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any] | None, bool]:
    base = _parse_package_entry(raw_kernel.get("base"))
    if base is None:
        return None, False
    kernel_spec_filter = _parse_kernel_spec_filter(raw_kernel.get("kernel_spec"))
    overrides = _parse_overrides(raw_kernel.get("overrides"), base)
    managers = _parse_managers(raw_kernel.get("managers"))
    required = _required_version(base, overrides)
    names = _candidate_names(base, overrides)

    checks: list[dict[str, Any]] = []
    has_issue = False
    for ks_name, ks_info in specs.items():
        spec = ks_info.get("spec", {}) if isinstance(ks_info, dict) else {}
        if not isinstance(spec, dict) or not _matches_kernelspec(kernel_spec_filter, spec):
            continue
        can_inspect, search_paths = _resolve_python_site_packages(spec)
        if can_inspect and ks_info.get("_companion_local", True):
            installed = _find_installed_version(names, search_paths=search_paths)
            check = _check_version(installed, required)
            if check["status"] in ("missing", "incompatible"):
                has_issue = True
        else:
            check = {
                "status": "unknown",
                "installedVersion": None,
                "requiredVersion": required,
                "message": None,
            }
        checks.append(
            {
                "kernelName": ks_name,
                "displayName": spec.get("display_name") or ks_name,
                "language": spec.get("language") or "",
                **check,
            }
        )

    kernel_entry: dict[str, Any] = {
        "kernel_spec": kernel_spec_filter,
        "base": base,
        "managers": managers,
        "checks": checks,
    }
    if overrides:
        kernel_entry["overrides"] = overrides
    return kernel_entry, has_issue


def _inspect_kernel_companions(
    raw_kernels: object, kernelspecs: dict[str, dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(raw_kernels, list):
        return [], False
    specs = kernelspecs or {}
    entries: list[dict[str, Any]] = []
    has_issue = False
    for raw_kernel in raw_kernels:
        if not isinstance(raw_kernel, dict):
            continue
        entry, entry_issue = _inspect_kernel_entry(raw_kernel, specs)
        if entry is not None:
            entries.append(entry)
            has_issue = has_issue or entry_issue
    return entries, has_issue


def inspect_companions(
    data: dict[str, Any],
    kernelspecs: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Parse ``jupyterlab.discovery`` metadata and check companion compatibility.

    Args:
        data: Extension ``package.json`` metadata dictionary.
        kernelspecs: Mapping of kernelspec names to kernelspec info dictionaries.

    Returns:
        A tuple of ``(companions_metadata, has_issue)``.
    """
    jlab_meta = data.get("jupyterlab")
    if not isinstance(jlab_meta, dict):
        return None, False

    discovery = jlab_meta.get("discovery")
    if not isinstance(discovery, dict):
        return None, False

    result: dict[str, Any] = {}
    server_entry, server_issue = _inspect_server_companion(discovery.get("server"))
    if server_entry is not None:
        result["server"] = server_entry

    kernel_entries, kernel_issue = _inspect_kernel_companions(discovery.get("kernel"), kernelspecs)
    if kernel_entries:
        result["kernel"] = kernel_entries

    if not result:
        return None, False

    return result, (server_issue or kernel_issue)
