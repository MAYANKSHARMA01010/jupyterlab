# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import sys
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from jupyter_client.kernelspec import KernelSpecManager
from traitlets.config import Configurable

from jupyterlab.extensions import ReadOnlyExtensionManager
from jupyterlab.extensions.companions import (
    _check_version,
    _resolve_python_site_packages,
    get_all_kernelspecs,
    inspect_companions,
)
from jupyterlab.extensions.manager import ExtensionPackage


@pytest.mark.parametrize(
    "extra",
    [
        {"env": {"PYTHONPATH": "/another/environment"}},
        {"metadata": {"kernel_provisioner": {"provisioner_name": "remote"}}},
        {"argv": [sys.executable, "-S", "-m", "ipykernel_launcher"]},
        {"argv": [sys.executable, "custom_launcher.py"]},
    ],
)
def test_kernel_launch_customizations_are_unverified(extra):
    spec = {"argv": [sys.executable, "-m", "ipykernel_launcher"], **extra}
    assert _resolve_python_site_packages(spec) == (False, None)


def test_relative_python_uses_server_interpreter(monkeypatch):
    monkeypatch.setenv("PATH", "")
    spec = {"argv": ["python3", "-m", "ipykernel_launcher"]}
    assert _resolve_python_site_packages(spec) == (True, None)


def test_inherited_site_packages_are_unverified(tmp_path):
    root = tmp_path / "env"
    (root / "bin").mkdir(parents=True)
    executable = root / "bin" / "python"
    executable.touch()
    (root / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    assert _resolve_python_site_packages(
        {"argv": [str(executable), "-m", "ipykernel_launcher"]}
    ) == (False, None)


def test_pip_override_does_not_fall_back_to_different_distribution(monkeypatch):
    def distribution(name):
        if name == "other-package":
            raise PackageNotFoundError(name)
        return type("Distribution", (), {"version": "1.0"})()

    monkeypatch.setattr("jupyterlab.extensions.companions.distribution", distribution)
    companions, has_issue = inspect_companions(
        {
            "jupyterlab": {
                "discovery": {
                    "server": {
                        "base": {"name": "base-package"},
                        "overrides": {"pip": {"name": "other-package"}},
                    }
                }
            }
        }
    )
    assert companions["server"]["check"]["status"] == "missing"
    assert has_issue


def test_version_only_override_inherits_package_name(monkeypatch):
    monkeypatch.setattr(
        "jupyterlab.extensions.companions._find_installed_version", lambda *a, **k: "1.0"
    )
    companions, has_issue = inspect_companions(
        {
            "jupyterlab": {
                "discovery": {
                    "server": {
                        "base": {"name": "base-package", "version": ">=1"},
                        "overrides": {"pip": {"version": ">=2"}},
                    }
                }
            }
        }
    )
    assert companions["server"]["check"]["requiredVersion"] == ">=2"
    assert has_issue


def test_invalid_requirement_is_unverified_even_if_package_missing():
    assert _check_version(None, "not a specifier")["status"] == "unknown"


def test_positional_extension_package_arguments_remain_compatible():
    package = ExtensionPackage("example", "", "", "prebuilt", True, False, None, True)
    assert package.core is True
    assert package.companions is None


async def test_configured_kernel_manager_is_used(monkeypatch):
    kernel_manager = KernelSpecManager()
    kernel_manager.get_all_specs = lambda: {"configured": {"spec": {}}}
    parent = Configurable()
    monkeypatch.setattr(
        parent, "serverapp", SimpleNamespace(kernel_spec_manager=kernel_manager), raising=False
    )
    manager = ReadOnlyExtensionManager(parent=parent)
    specs = await manager._get_companion_kernelspecs(
        {
            "federated_extensions": {"example": {"jupyterlab": {"discovery": {"kernel": []}}}},
            "extensions": {},
            "shadowed_exts": [],
        }
    )
    assert set(specs) == {"configured"}


async def test_async_remote_kernel_manager_is_not_checked_locally(monkeypatch):
    manager = SimpleNamespace(
        get_all_specs=AsyncMock(
            return_value={
                "remote": {
                    "spec": {
                        "argv": [sys.executable, "-m", "ipykernel_launcher"],
                        "language": "python",
                    }
                }
            }
        )
    )
    specs = await get_all_kernelspecs(manager)
    monkeypatch.setattr(
        "jupyterlab.extensions.companions._find_installed_version", lambda *a, **k: "1.0"
    )
    companions, has_issue = inspect_companions(
        {
            "jupyterlab": {
                "discovery": {
                    "kernel": [
                        {"base": {"name": "example-package"}, "kernel_spec": {"language": "python"}}
                    ]
                }
            }
        },
        specs,
    )
    assert companions["kernel"][0]["checks"][0]["status"] == "unknown"
    assert not has_issue


@pytest.mark.parametrize("customization", ["inherited", "pth", "multiple-versions", "no-config"])
def test_ambiguous_external_environment_is_unverified(tmp_path, customization):
    root = tmp_path / "env"
    (root / "bin").mkdir(parents=True)
    executable = root / "bin" / "python"
    executable.touch()
    site = root / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    config = root / "pyvenv.cfg"
    if customization != "no-config":
        config.write_text("version = 3.12.0\ninclude-system-site-packages = false\n")
    if customization == "inherited":
        config.write_text("version = 3.12.0\ninclude-system-site-packages = true\n")
    elif customization == "pth":
        (site / "extra.pth").write_text("/another/environment\n")
    elif customization == "multiple-versions":
        config.write_text("version = 3.13.0\ninclude-system-site-packages = false\n")
    assert _resolve_python_site_packages(
        {"argv": [str(executable), "-m", "ipykernel_launcher"]}
    ) == (False, None)
