"""Locate implementation and harness sources in flat or organized snapshots.

These helpers only inspect paths. They never import a requested snapshot, add it
to sys.path, or let a missing baseline dependency fall back to the live checkout.
"""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def runtime_directory(directory):
    """Accept a repository snapshot root or its actual runtime directory."""
    directory = Path(directory).resolve()
    packaged = directory / "virtualads"
    return packaged if packaged.is_dir() else directory


def runtime_sources(directory):
    """Hash implementations, including package wiring, rather than compat aliases."""
    directory = runtime_directory(directory)
    if directory.name == "virtualads":
        return sorted(directory.glob("*.py"))
    paths = set(directory.glob("paste*.py"))
    for name in ("shared_paste_example.py", "dynamic_paste_example.py"):
        path = directory / name
        if path.is_file():
            paths.add(path)
    return sorted(paths)


def source_file(directory, filename):
    """Resolve a known filename in one snapshot without using another checkout."""
    directory = Path(directory).resolve()
    for parent in ("virtualads", "benchmarks", "tools", "examples", ""):
        path = directory / parent / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"source file {filename} is missing from {directory}")


def source_key(path, directory):
    """Use portable repository-relative paths in provenance JSON."""
    return Path(path).resolve().relative_to(Path(directory).resolve()).as_posix()
