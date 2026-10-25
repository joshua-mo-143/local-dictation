from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from local_dictation import service_units
from local_dictation.install_distro import (
    DistroFamily,
    InstallPlan,
    InstallPlanError,
    detect_distro_family,
    discover_install_paths,
    main,
    parse_os_release,
    resolve_device,
    resolve_extra,
    resolve_install_plan,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = PROJECT_ROOT / "scripts" / "install.sh"
ARCH_RELEASE = 'ID=arch\nNAME="Arch Linux"\n'
UBUNTU_RELEASE = 'ID=ubuntu\nID_LIKE=debian\nNAME="Ubuntu"\n'
FEDORA_RELEASE = 'ID=fedora\nNAME="Fedora Linux"\n'
OS_RELEASES = {
    DistroFamily.ARCH: ARCH_RELEASE,
    DistroFamily.DEBIAN: UBUNTU_RELEASE,
    DistroFamily.FEDORA: FEDORA_RELEASE,
}


def _plan(
    family: DistroFamily,
    root: Path,
    *,
    device: str = "cpu",
    cuda_available: bool = False,
    install_packages: bool = True,
) -> InstallPlan:
    home = root / "home"
    return resolve_install_plan(
        device=device,
        cuda_available=cuda_available,
        checkout=root / "checkout",
        os_release=parse_os_release(OS_RELEASES[family]),
        home=home,
        environment={"HOME": str(home)},
        install_packages=install_packages,
    )


class DeviceTests(unittest.TestCase):
    def test_requested_device_selects_the_extra(self) -> None:
        cases = {
            ("cpu", False): "cpu",
            ("cpu", True): "cpu",
            ("cuda", False): "cuda",
            ("cuda", True): "cuda",
            ("auto", False): "cpu",
            ("auto", True): "cuda",
        }
        for (device, cuda_available), extra in cases.items():
            with self.subTest(device=device, cuda_available=cuda_available):
                self.assertEqual(
                    resolve_extra(
                        resolve_device(device), cuda_available=cuda_available
                    ),
                    extra,
                )

    def test_device_defaults_to_auto_and_ignores_case_and_spacing(self) -> None:
        self.assertEqual(resolve_device(None), "auto")
        self.assertEqual(resolve_device(""), "auto")
        self.assertEqual(resolve_device(" CUDA "), "cuda")

    def test_unknown_device_fails_closed(self) -> None:
        with self.assertRaisesRegex(InstallPlanError, "auto, cpu, or cuda"):
            resolve_device("gpu")

    def test_plan_keeps_the_requested_device_for_the_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = _plan(
                DistroFamily.ARCH, Path(temporary), device="auto", cuda_available=True
            )

        self.assertEqual((plan.device, plan.extra), ("auto", "cuda"))


class DistroDetectionTests(unittest.TestCase):
    def test_detects_arch_ubuntu_and_fedora_families(self) -> None:
        for family, release in OS_RELEASES.items():
            with self.subTest(family=family.value):
                self.assertEqual(
                    detect_distro_family(parse_os_release(release)), family
                )
        self.assertEqual(
            detect_distro_family({"ID": "endeavouros", "ID_LIKE": "arch"}),
            DistroFamily.ARCH,
        )

    def test_unknown_distro_is_rejected(self) -> None:
        with self.assertRaisesRegex(InstallPlanError, "unsupported distribution"):
            detect_distro_family({"ID": "alpine"})


class PackageTests(unittest.TestCase):
    def test_cuda_runtime_is_installed_only_with_the_cuda_extra(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cpu = _plan(DistroFamily.ARCH, root, device="auto")
            cuda = _plan(DistroFamily.ARCH, root, device="auto", cuda_available=True)

        self.assertEqual((cpu.extra, cpu.cuda_packages), ("cpu", ()))
        self.assertEqual((cuda.extra, cuda.cuda_packages), ("cuda", ("cuda",)))
        self.assertNotIn("cuda", cuda.packages)

    def test_package_names_are_mapped_per_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arch = _plan(DistroFamily.ARCH, root)
            debian = _plan(DistroFamily.DEBIAN, root)
            fedora = _plan(DistroFamily.FEDORA, root)

        self.assertEqual(arch.package_manager, "paru")
        self.assertEqual(arch.install_command[0], "paru")
        self.assertIn("uv", arch.packages)
        self.assertFalse(arch.uv_bootstrap)
        self.assertEqual(debian.package_manager, "apt-get")
        self.assertIn("libnotify-bin", debian.packages)
        self.assertTrue(debian.uv_bootstrap)
        self.assertEqual(debian.skipped_packages, ("uv",))
        self.assertEqual(fedora.package_manager, "dnf")
        self.assertIn("libnotify", fedora.packages)
        self.assertTrue(fedora.uv_bootstrap)

    def test_each_family_installs_exactly_the_tools_dictation_runs(self) -> None:
        tools = ("xdotool", "xclip", "wl-clipboard", "wtype", "git", "curl")
        expected = {
            DistroFamily.ARCH: (
                "pipewire",
                "pipewire-audio",
                "wireplumber",
                "libnotify",
                *tools,
                "uv",
            ),
            DistroFamily.DEBIAN: (
                "pipewire",
                "pipewire-bin",
                "wireplumber",
                "libnotify-bin",
                *tools,
            ),
            DistroFamily.FEDORA: (
                "pipewire",
                "pipewire-utils",
                "wireplumber",
                "libnotify",
                *tools,
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            for family, packages in expected.items():
                with self.subTest(family=family.value):
                    self.assertEqual(_plan(family, Path(temporary)).packages, packages)

    def test_cuda_without_a_package_adapter_fails_closed(self) -> None:
        for family in (DistroFamily.DEBIAN, DistroFamily.FEDORA):
            with (
                self.subTest(family=family.value),
                tempfile.TemporaryDirectory() as temporary,
            ):
                with self.assertRaisesRegex(
                    InstallPlanError,
                    "no supported package adapter for: cuda;.*"
                    "SKIP_SYSTEM_PACKAGES=1.*--device cpu",
                ):
                    _plan(family, Path(temporary), device="cuda")

                skipped = _plan(
                    family, Path(temporary), device="cuda", install_packages=False
                )
                self.assertEqual(skipped.extra, "cuda")
                self.assertEqual(skipped.cuda_packages, ())
                self.assertIn("cuda", skipped.skipped_packages)

                cpu = _plan(family, Path(temporary), device="auto")
                self.assertEqual(cpu.extra, "cpu")

    def test_resolving_the_same_inputs_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = _plan(DistroFamily.DEBIAN, Path(temporary))
            second = _plan(DistroFamily.DEBIAN, Path(temporary))
        self.assertEqual(first, second)


class InstallPathTests(unittest.TestCase):
    def test_paths_follow_checkout_home_and_xdg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "src" / "checkout"
            checkout.mkdir(parents=True)
            home = root / "user"
            paths = discover_install_paths(
                checkout=checkout,
                home=home,
                environment={
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(root / "xdg-config"),
                },
            )

        self.assertEqual(paths.checkout, checkout.resolve())
        self.assertEqual(paths.service_checkout, home / "local-dictation")
        self.assertEqual(
            service_units.INSTALL_ROOT,
            "%h/" + paths.service_checkout.relative_to(home).as_posix(),
        )
        self.assertEqual(paths.user_bin, home / ".local" / "bin")
        self.assertEqual(paths.dictate, home / ".local" / "bin" / "dictate")
        self.assertEqual(paths.systemd_user_dir, root / "xdg-config/systemd/user")
        self.assertEqual(paths.config_dir, root / "xdg-config" / "local-dictation")

    def test_relative_xdg_config_home_falls_back_to_home(self) -> None:
        home = Path("/home/example")
        paths = discover_install_paths(
            checkout=Path("/srv/checkout"),
            home=home,
            environment={"HOME": str(home), "XDG_CONFIG_HOME": "relative"},
        )

        self.assertEqual(paths.systemd_user_dir, home / ".config/systemd/user")
        self.assertEqual(paths.config_dir, home / ".config/local-dictation")

    def test_runtime_systemd_dir_matches_the_planned_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                **os.environ,
                "HOME": str(root / "home"),
                "XDG_CONFIG_HOME": str(root / "xdg-config"),
            }
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from local_dictation.config import SYSTEMD_USER_DIR; "
                    "print(SYSTEMD_USER_DIR)",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            planned = discover_install_paths(
                checkout=root, environment=environment
            ).systemd_user_dir

        self.assertEqual(result.stdout.strip(), str(planned))


class PlannerCliTests(unittest.TestCase):
    def run_planner(
        self, root: Path, release: str, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        os_release = root / "os-release"
        os_release.write_text(release)
        # -S keeps site-packages off sys.path: the planner must run on a bare
        # system Python before any project environment exists.
        return subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "local_dictation.install_distro",
                "--checkout",
                str(root / "checkout"),
                "--os-release",
                str(os_release),
                *arguments,
            ],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(root / "home"),
                "PYTHONPATH": str(PROJECT_ROOT / "src"),
            },
        )

    def test_env_output_is_shell_evaluable_on_a_bare_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "with space"
            root.mkdir()
            planned = self.run_planner(
                root, ARCH_RELEASE, "--device", "auto", "--cuda-available"
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            evaluated = subprocess.run(
                [
                    "sh",
                    "-c",
                    'eval "$1"; printf "%s|%s|%s|%s|%s\\n" "$INSTALL_DEVICE" '
                    '"$INSTALL_EXTRA" "$INSTALL_DISTRO_CUDA_PACKAGES" '
                    '"$INSTALL_SERVICE_CHECKOUT" "$INSTALL_DICTATE"',
                    "sh",
                    planned.stdout,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        home = root / "home"
        self.assertEqual(
            evaluated.stdout.strip(),
            f"auto|cuda|cuda|{home / 'local-dictation'}|{home / '.local/bin/dictate'}",
        )

    def test_json_output_and_skip_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            refused = self.run_planner(root, UBUNTU_RELEASE, "--device", "cuda")
            skipped = self.run_planner(
                root,
                UBUNTU_RELEASE,
                "--device",
                "cuda",
                "--skip-system-packages",
                "--format",
                "json",
            )

        self.assertEqual(refused.returncode, 1)
        self.assertIn("error: debian has no supported package adapter", refused.stderr)
        self.assertEqual(skipped.returncode, 0, skipped.stderr)
        payload = json.loads(skipped.stdout)
        self.assertEqual(payload["extra"], "cuda")
        self.assertEqual(
            payload["paths"]["config_dir"], str(root / "home/.config/local-dictation")
        )

    def test_cli_rejects_unknown_device_and_unreadable_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unknown = self.run_planner(root, ARCH_RELEASE, "--device", "gpu")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                unreadable = main(
                    ["--checkout", str(root), "--os-release", str(root / "missing")]
                )

        self.assertEqual(unknown.returncode, 1)
        self.assertIn("auto, cpu, or cuda", unknown.stderr)
        self.assertEqual(unreadable, 1)
        self.assertIn("could not read", stderr.getvalue())


class InstallerScriptTests(unittest.TestCase):
    installer = INSTALLER.read_text()

    def position(self, text: str) -> int:
        self.assertIn(text, self.installer)
        return self.installer.index(text)

    def test_installer_plans_then_syncs_the_service_environment(self) -> None:
        venv = service_units.EXPECTED_EXECSTART.removeprefix(
            service_units.INSTALL_ROOT + "/"
        ).split("/bin/")[0]

        self.assertTrue(self.installer.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("\nset -euo pipefail\n", self.installer)
        self.assertIn(f'SERVICE_VENV="$PROJECT_DIR/{venv}"', self.installer)
        self.assertIn("-m local_dictation.install_distro", self.installer)
        self.assertIn('plan="$(resolve_install_plan', self.installer)
        self.assertIn("|| exit 1", self.installer)
        self.assertIn("plan_args+=(--skip-system-packages)", self.installer)
        self.assertIn("timeout 10 nvidia-smi", self.installer)
        self.assertIn(
            'UV_PROJECT_ENVIRONMENT="$SERVICE_VENV" \\\n'
            '    uv sync --python "$PYTHON_VERSION" --extra "$INSTALL_EXTRA" --no-dev',
            self.installer,
        )
        self.assertNotIn("pacman", self.installer)

    def test_installer_configures_and_starts_the_service_in_order(self) -> None:
        steps = [
            self.position("bootstrap_installer_python\n\n"),
            self.position('plan="$(resolve_install_plan'),
            self.position('link_service_checkout "$INSTALL_SERVICE_CHECKOUT"'),
            self.position("install_distro_packages $INSTALL_DISTRO_PACKAGES"),
            self.position("uv sync --python"),
            self.position(
                'link_dictate "$INSTALL_DICTATE" "$SERVICE_VENV/bin/dictate"'
            ),
            self.position('"$INSTALL_DICTATE" config set stt.device "$INSTALL_DEVICE"'),
            self.position('"$INSTALL_DICTATE" services install\n'),
            self.position('"$INSTALL_DICTATE" services audit\n'),
            self.position('"$INSTALL_DICTATE" services start\n'),
            self.position("dictate config set dictation.source"),
            self.position("dictate doctor"),
        ]
        self.assertEqual(steps, sorted(steps))

    def test_installer_only_fetches_uv_and_runs_dictate_setup(self) -> None:
        self.assertEqual(self.installer.count("curl "), 1)
        self.assertIn("curl -LsSf https://astral.sh/uv/install.sh | sh", self.installer)
        self.assertEqual(
            re.findall(r'"\$INSTALL_DICTATE" (\S+ \S+)', self.installer),
            ["config set", "services install", "services audit", "services start"],
        )


class InstallerHelperTests(unittest.TestCase):
    """Source the installer and exercise its link helpers in a scratch home."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "with space"
        self.home = self.root / "home"
        self.home.mkdir(parents=True)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for tool in ("dirname", "ln", "mkdir", "readlink"):
            located = shutil.which(tool)
            assert located is not None
            (self.bin / tool).symlink_to(located)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()

    def run_helper(self, *command: str) -> subprocess.CompletedProcess[str]:
        bash = shutil.which("bash")
        assert bash is not None
        process = subprocess.run(
            [bash, "-c", 'source "$1"; shift; "$@"', "bash", str(INSTALLER), *command],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": str(self.bin),
                "HOME": str(self.home),
                "SKIP_SYSTEM_PACKAGES": "1",
            },
        )
        self.assertNotIn("installer", process.stdout)
        return process

    def test_service_checkout_link_is_created_and_retargeted(self) -> None:
        link = self.home / "local-dictation"

        created = self.run_helper(
            "link_service_checkout", str(link), str(self.checkout)
        )
        self.assertEqual(created.returncode, 0, created.stderr)
        self.assertEqual(os.readlink(link), str(self.checkout))

        other = self.root / "other"
        other.mkdir()
        retargeted = self.run_helper("link_service_checkout", str(link), str(other))
        self.assertEqual(retargeted.returncode, 0, retargeted.stderr)
        self.assertEqual(os.readlink(link), str(other))

    def test_service_checkout_link_refuses_real_paths(self) -> None:
        link = self.home / "local-dictation"
        link.mkdir()
        (link / "keep").write_text("data")

        refused = self.run_helper(
            "link_service_checkout", str(link), str(self.checkout)
        )

        self.assertEqual(refused.returncode, 1)
        self.assertIn("refusing to replace it", refused.stderr)
        self.assertFalse(link.is_symlink())
        self.assertEqual((link / "keep").read_text(), "data")

    def test_checkout_already_at_the_service_path_is_left_alone(self) -> None:
        real_home = self.root / "real-home"
        checkout = real_home / "local-dictation"
        checkout.mkdir(parents=True)
        linked_home = self.root / "linked-home"
        linked_home.symlink_to(real_home, target_is_directory=True)

        for link in (checkout, linked_home / "local-dictation"):
            with self.subTest(link=str(link)):
                result = self.run_helper(
                    "link_service_checkout", str(link), str(checkout.resolve())
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(checkout.is_symlink())

    def test_dictate_link_is_created_or_retargeted_but_never_clobbers_files(
        self,
    ) -> None:
        link = self.home / ".local" / "bin" / "dictate"
        target = self.checkout / ".venv-service" / "bin" / "dictate"

        created = self.run_helper("link_dictate", str(link), str(target))
        self.assertEqual(created.returncode, 0, created.stderr)
        self.assertEqual(os.readlink(link), str(target))

        link.unlink()
        moved = self.root / "old-checkout" / ".venv-service" / "bin" / "dictate"
        link.symlink_to(moved)
        retargeted = self.run_helper("link_dictate", str(link), str(target))
        self.assertEqual(retargeted.returncode, 0, retargeted.stderr)
        self.assertEqual(os.readlink(link), str(target))

        link.unlink()
        link.symlink_to(self.root / "another-tool")
        refused = self.run_helper("link_dictate", str(link), str(target))
        self.assertEqual(refused.returncode, 1)
        self.assertIn("refusing to replace it", refused.stderr)
        self.assertEqual(os.readlink(link), str(self.root / "another-tool"))

        link.unlink()
        link.write_text("#!/bin/sh\necho another tool\n")
        refused = self.run_helper("link_dictate", str(link), str(target))
        self.assertEqual(refused.returncode, 1)
        self.assertIn("refusing to replace it", refused.stderr)
        self.assertEqual(link.read_text(), "#!/bin/sh\necho another tool\n")

        link.unlink()
        link.mkdir()
        refused = self.run_helper("link_dictate", str(link), str(target))
        self.assertEqual(refused.returncode, 1)
        self.assertTrue(link.is_dir())


if __name__ == "__main__":
    unittest.main()
