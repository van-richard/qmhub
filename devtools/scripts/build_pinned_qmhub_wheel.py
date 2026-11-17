#!/usr/bin/env python3
"""Build a verified QMHub wheel from an existing commit without changing HEAD.

Run with a deployment interpreter that already provides pip, setuptools, wheel,
and versioningit 2.3.0. Output directories must be new and beneath repo/build.
The native helPME extension is intentionally handled separately from this wheel.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import tomllib
import zipfile


class BuildError(RuntimeError):
    """A source, build, or wheel identity check failed."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False,
    )
    if result.returncode:
        raise BuildError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def is_native(path: str) -> bool:
    return bool(re.search(r"\.(?:so(?:\.[^/]*)?|pyd|dll|dylib|a|o)$", path))


def pinned_version(repo: Path, stage: Path, commit: str, epoch: int) -> tuple[str, str]:
    # Reuse versioningit's actual tag/distance/format rules, while replacing
    # its implicit HEAD with the requested immutable commit and clean state.
    from versioningit import Versioningit
    from versioningit.git import DescribeOpts, GitRepo, describe_git_core

    class PinnedRepo(GitRepo):
        def describe(self, opts):
            return self.read("describe", "--long", "--always", *opts.as_args(), commit)

        def get_branch(self):
            return None

        def read(self, *args, **kwargs):
            return super().read(*(commit if arg == "HEAD" else arg for arg in args), **kwargs)

    config = tomllib.loads((stage / "pyproject.toml").read_text())
    vcs = config["tool"]["versioningit"]["vcs"]
    if vcs.get("method", "git") != "git":
        raise BuildError("Only the repository's git versioning method is supported.")
    opts = DescribeOpts(tags=True, match=vcs.get("match", []), exclude=vcs.get("exclude", []))
    pinned = PinnedRepo(repo)
    description = describe_git_core(
        pinned, datetime.fromtimestamp(epoch, timezone.utc), vcs.get("default-tag"), opts,
    )
    versioning = Versioningit.from_project_dir(stage)
    base = versioning.do_tag2version(description.tag)
    version = base if description.state == "exact" else versioning.do_format(
        description=description,
        base_version=base,
        next_version=versioning.do_next_version(base, None),
    )
    if "unknown" in version or "dirty" in version:
        raise BuildError(f"Refusing an ambiguous source version: {version}")
    return version, pinned.describe(opts)


def source_entries(repo: Path, commit: str) -> dict[str, tuple[str, str]]:
    entries = {}
    for record in git(repo, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        path = raw_path.decode("utf-8")
        parts = PurePosixPath(path)
        if parts.is_absolute() or ".." in parts.parts or kind != "blob":
            raise BuildError(f"Unsupported source entry: {path}")
        if mode not in ("100644", "100755", "120000") or is_native(path):
            raise BuildError(f"Unexpected source mode or native artifact: {path}")
        entries[path] = (mode, oid)
    return entries


def verify_archive(stage: Path, entries: dict[str, tuple[str, str]]) -> dict[str, str]:
    actual = {
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*") if path.is_symlink() or not path.is_dir()
    }
    if actual != entries.keys():
        raise BuildError(
            f"Archive file mismatch; missing={sorted(entries.keys() - actual)}, "
            f"unexpected={sorted(actual - entries.keys())}"
        )
    hashes = {}
    for name, (mode, expected_oid) in entries.items():
        path = stage / name
        if path.is_symlink() != (mode == "120000"):
            raise BuildError(f"Archive symlink mode mismatch: {name}")
        data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        if path.is_symlink() and not path.resolve().is_relative_to(stage):
            raise BuildError(f"Archive symlink escapes stage: {name}")
        # Git's object hash includes a header; checking it detects changed
        # archive contents independently of the deployment SHA-256 manifest.
        oid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if oid != expected_oid:
            raise BuildError(f"Archive content differs from the pinned Git blob: {name}")
        hashes[name] = sha256(data)
    return hashes


def verify_wheel(wheel: Path, stage: Path, package_files: set[str], version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise BuildError("Wheel contains duplicate paths.")
        actual = {name for name in names if name.startswith("qmhub/") and not name.endswith("/")}
        if actual != package_files:
            raise BuildError(
                f"Wheel package mismatch; missing={sorted(package_files - actual)}, "
                f"unexpected={sorted(actual - package_files)}"
            )
        dist_infos = {
            name.split("/", 1)[0] for name in names if name.split("/", 1)[0].endswith(".dist-info")
        }
        if len(dist_infos) != 1:
            raise BuildError("Expected exactly one wheel metadata directory.")
        dist_info = next(iter(dist_infos))
        for name in names:
            if is_native(name) or name.split("/", 1)[0] not in ("qmhub", dist_info):
                raise BuildError(f"Unexpected wheel path or native artifact: {name}")
        for name in package_files:
            if archive.read(name) != (stage / name).read_bytes():
                raise BuildError(f"Wheel content differs from the source stage: {name}")
        metadata = BytesParser().parsebytes(archive.read(f"{dist_info}/METADATA"))
        if metadata["Name"] != "qmhub" or metadata["Version"] != version:
            raise BuildError("Wheel package name/version differs from the pinned source.")


def build(args: argparse.Namespace) -> Path:
    repo = args.repo.resolve()
    if Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve() != repo:
        raise BuildError("--repo must identify the repository root, not a subdirectory.")
    commit = args.commit
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise BuildError("--commit must be a full 40-character lowercase Git commit SHA.")
    if git(repo, "rev-parse", f"{commit}^{{commit}}").decode().strip() != commit:
        raise BuildError("The requested commit does not exist in this repository.")
    output = args.output_dir.resolve()
    build_root = (repo / "build").resolve()
    if not build_root.is_relative_to(repo) or output == build_root or not output.is_relative_to(build_root):
        raise BuildError("--output-dir must be a new directory below repo/build.")
    if output.exists():
        raise BuildError(f"Refusing to reuse or overwrite an output directory: {output}")
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("pip", "setuptools", "wheel", "versioningit")
    }
    if dependencies["versioningit"] != "2.3.0":
        raise BuildError("This builder requires the inspected versioningit==2.3.0 implementation.")
    entries = source_entries(repo, commit)
    if "PKG-INFO" in entries or "qmhub/_version.py" in entries:
        raise BuildError("Pinned sources already contain generated version metadata.")
    epoch = int(git(repo, "show", "-s", "--format=%ct", commit).decode().strip())
    archive_data = git(repo, "archive", "--format=tar", commit)
    output.mkdir(parents=True, exist_ok=False)
    stage = output / "source"
    stage.mkdir()
    archive_path = output / "source.tar"
    archive_path.write_bytes(archive_data)
    with tarfile.open(fileobj=io.BytesIO(archive_data)) as archive:
        # The data filter also rejects traversal and links outside the fresh
        # stage; GNU patch symlinks inside the archived tree remain intact.
        archive.extractall(stage, filter="data")
    hashes = verify_archive(stage, entries)
    version, description = pinned_version(repo, stage, commit, epoch)
    (stage / "PKG-INFO").write_text(f"Metadata-Version: 2.1\nName: qmhub\nVersion: {version}\n\n")
    # versioningit 2.3's sdist fallback reads PKG-INFO but skips its write step.
    generated_version = f'__version__ = "{version}"\n'.encode()
    (stage / "qmhub/_version.py").write_bytes(generated_version)
    runtime_hashes = {
        name: value for name, value in hashes.items()
        if name.startswith("qmhub/") and name.endswith(".py") and not name.startswith("qmhub/tests/")
    }
    package_files = {name for name in hashes if name.startswith("qmhub/")}
    package_files.add("qmhub/_version.py")
    wheels = output / "wheels"
    wheels.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "staged",
        "source_commit": commit,
        "source_description": description,
        "package_version": version,
        "source_date_epoch": epoch,
        "source_stage": str(stage),
        "source_archive": {"path": str(archive_path), "sha256": sha256(archive_data)},
        "build_interpreter": sys.executable,
        "build_dependencies": dependencies,
        "source_file_sha256": hashes,
        "runtime_file_sha256": runtime_hashes,
        "generated_version_sha256": sha256((stage / "qmhub/_version.py").read_bytes()),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        GIT_CEILING_DIRECTORIES=str(repo), SOURCE_DATE_EPOCH=str(epoch),
    )
    command = [
        sys.executable, "-I", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
        "--no-cache-dir", "--wheel-dir", str(wheels), str(stage),
    ]
    try:
        log_path = output / "build.log"
        with log_path.open("w") as log:
            log.write(shlex.join(command) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=stage, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise BuildError(f"pip wheel failed with return code {result.returncode}; see {log_path}")
        built = list(wheels.glob("*.whl"))
        if len(built) != 1:
            raise BuildError("Expected exactly one wheel from the pinned build.")
        if (stage / "qmhub/_version.py").read_bytes() != generated_version:
            raise BuildError("Build changed the pinned generated version module.")
        verify_wheel(built[0], stage, package_files, version)
        for name in package_files - {"qmhub/_version.py"}:
            if sha256((stage / name).read_bytes()) != hashes[name]:
                raise BuildError(f"Build changed archived package source: {name}")
        manifest.update(status="built", wheel={"path": str(built[0]), "sha256": sha256(built[0].read_bytes())})
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "version": version, **manifest["wheel"]}))
    return built[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        build(args)
    except (BuildError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"Pinned QMHub wheel build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
