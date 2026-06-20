#!/usr/bin/env python3
"""Manage Wh0 external weights under one root directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_ROOT = REPO_ROOT / "weights"
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "weights_manifest.json"
DEFAULT_HFD_SCRIPT = REPO_ROOT / "tools" / "weights" / "hfd.sh"
PROXY_ENV_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_local_paths(items: Iterable[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --local-path '{item}', expected <id>=<path>")
        key, value = item.split("=", 1)
        parsed[key.strip()] = Path(value).expanduser().resolve()
    return parsed


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def sync_link(target: Path, source: Path) -> None:
    source = source.resolve()
    ensure_parent(target)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source:
            return
        safe_remove(target)
    target.symlink_to(source, target_is_directory=source.is_dir())


def download_file(url: str, destination: Path) -> None:
    ensure_parent(destination)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(destination)


def ensure_hfd_script(path: Path = DEFAULT_HFD_SCRIPT) -> Path:
    configured = os.environ.get("WH0_HFD_SCRIPT", "").strip()
    if configured:
        path = Path(configured).expanduser()
    if path.exists():
        path.chmod(path.stat().st_mode | 0o111)
        return path
    if not shutil.which("wget"):
        raise RuntimeError("wget is required to bootstrap hfd.sh from https://hf-mirror.com/hfd/hfd.sh")
    url = os.environ.get("WH0_HFD_URL", "https://hf-mirror.com/hfd/hfd.sh")
    subprocess.run(["wget", url, "-O", str(path)], check=True)
    path.chmod(0o755)
    return path


def hfd_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in PROXY_ENV_VARS:
        env.pop(name, None)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return env


def hfd_base_cmd(repo_id: str, local_dir: Path) -> list[str]:
    hfd = ensure_hfd_script()
    tool = os.environ.get("WH0_HFD_TOOL", "aria2c" if shutil.which("aria2c") else "wget")
    cmd = [
        "bash",
        str(hfd),
        repo_id,
        "--local-dir",
        str(local_dir),
        "--tool",
        tool,
    ]
    if tool == "aria2c":
        cmd.extend(["-x", os.environ.get("WH0_HFD_CONNECTIONS", "8")])
        cmd.extend(["-j", os.environ.get("WH0_HFD_JOBS", "5")])
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        cmd.extend(["--hf_token", token])
    return cmd


def download_hf_snapshot_with_hfd(repo_id: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(hfd_base_cmd(repo_id, destination), check=True, env=hfd_env())


def parse_hf_resolve_url(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    if parsed.netloc not in {"huggingface.co", "hf-mirror.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if "resolve" not in parts:
        return None
    resolve_idx = parts.index("resolve")
    if resolve_idx < 2 or resolve_idx + 2 >= len(parts):
        return None
    repo_id = "/".join(parts[:resolve_idx])
    revision = parts[resolve_idx + 1]
    repo_file = "/".join(parts[resolve_idx + 2 :])
    return repo_id, revision, repo_file


def download_hf_file_with_hfd(url: str, destination: Path) -> bool:
    parsed = parse_hf_resolve_url(url)
    if parsed is None:
        return False
    repo_id, revision, repo_file = parsed
    scratch = destination.parent / f".{destination.name}.hfd"
    scratch.mkdir(parents=True, exist_ok=True)
    cmd = hfd_base_cmd(repo_id, scratch)
    cmd.extend(["--revision", revision, "--include", repo_file])
    subprocess.run(cmd, check=True, env=hfd_env())
    source = scratch / repo_file
    if not source.exists():
        raise FileNotFoundError(f"hfd did not produce expected file: {source}")
    ensure_parent(destination)
    shutil.copy2(source, destination)
    return True


def download_hf_snapshot_with_hub(repo_id: str, destination: Path) -> None:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(destination), local_dir_use_symlinks=False)


def download_hf_file(repo_id: str, filename: str, destination: Path, downloader: str) -> None:
    if downloader == "hfd":
        scratch = destination.parent / f".{destination.name}.hfd"
        scratch.mkdir(parents=True, exist_ok=True)
        cmd = hfd_base_cmd(repo_id, scratch)
        cmd.extend(["--include", filename])
        subprocess.run(cmd, check=True, env=hfd_env())
        source = scratch / filename
        if not source.exists():
            raise FileNotFoundError(f"hfd did not produce expected file: {source}")
        ensure_parent(destination)
        shutil.copy2(source, destination)
        return
    if downloader == "huggingface_hub":
        from huggingface_hub import hf_hub_download

        source = hf_hub_download(repo_id=repo_id, filename=filename)
        ensure_parent(destination)
        shutil.copy2(source, destination)
        return
    raise ValueError(f"Unsupported Hugging Face downloader: {downloader}")


def download_hf_snapshot(repo_id: str, destination: Path, downloader: str) -> None:
    if downloader == "hfd":
        download_hf_snapshot_with_hfd(repo_id, destination)
        return
    if downloader == "huggingface_hub":
        download_hf_snapshot_with_hub(repo_id, destination)
        return
    raise ValueError(f"Unsupported Hugging Face downloader: {downloader}")


def item_env_name(item_id: str) -> str:
    return f"WH0_{item_id.upper()}_LOCAL"


def install_item(item: dict, weights_root: Path, local_paths: dict[str, Path], hf_downloader: str) -> str:
    destination = weights_root / item["target"]
    local_source = local_paths.get(item["id"])
    env_source = os.environ.get(item_env_name(item["id"]))
    source = local_source or (Path(env_source).expanduser().resolve() if env_source else None)

    if source is not None:
        if not source.exists():
            raise FileNotFoundError(f"Local source for {item['id']} does not exist: {source}")
        sync_link(destination, source)
        return f"linked  {item['id']} -> {source}"

    kind = item["kind"]
    if kind == "manual":
        return (
            f"manual  {item['id']} -> {destination} "
            f"(download yourself from {item['source_url']} or pass --local-path {item['id']}=/abs/path)"
        )
    if kind == "url":
        if destination.exists():
            return f"exists   {item['id']} -> {destination}"
        if not download_hf_file_with_hfd(item["url"], destination):
            download_file(item["url"], destination)
        return f"fetched  {item['id']} -> {destination}"
    if kind == "hf_snapshot":
        if destination.exists() and any(destination.iterdir()):
            return f"exists   {item['id']} -> {destination}"
        download_hf_snapshot(item["repo_id"], destination, hf_downloader)
        return f"fetched  {item['id']} -> {destination} ({hf_downloader})"
    if kind == "hf_file":
        if destination.exists():
            return f"exists   {item['id']} -> {destination}"
        download_hf_file(item["repo_id"], item["filename"], destination, hf_downloader)
        return f"fetched  {item['id']} -> {destination} ({hf_downloader})"
    raise ValueError(f"Unsupported manifest kind: {kind}")


def ensure_compatibility_links(repo_root: Path, weights_root: Path) -> None:
    links = {
        repo_root / "WM-H" / "models": weights_root / "models",
        repo_root / "WM-H" / "weights" / "external" / "detector.pt": weights_root / "hawor" / "external" / "detector.pt",
        repo_root / "vitra-wh0" / "weights" / "mano" / "MANO_RIGHT.pkl": weights_root / "mano" / "MANO_RIGHT.pkl",
        repo_root / "vitra-wh0" / "weights" / "hawor" / "checkpoints" / "hawor.ckpt": weights_root / "hawor" / "checkpoints" / "hawor.ckpt",
        repo_root / "vitra-wh0" / "weights" / "hawor" / "external" / "detector.pt": weights_root / "hawor" / "external" / "detector.pt",
        repo_root / "vitra-wh0" / "checkpoints" / "vitra-vla-3b.pt": weights_root / "checkpoints" / "vitra-vla-3b.pt",
    }
    for target, source in links.items():
        if source.exists():
            sync_link(target, source)


def print_list(manifest: list[dict], weights_root: Path) -> None:
    for item in manifest:
        destination = weights_root / item["target"]
        required_by = ", ".join(item.get("required_by", []))
        source = item.get("url") or item.get("repo_id") or item.get("source_url", "")
        print(f"{item['id']}\n  kind: {item['kind']}\n  target: {destination}\n  source: {source}\n  required_by: {required_by}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Wh0 weights and model assets.")
    parser.add_argument("command", choices=["list", "sync"])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--weights-root", default=str(DEFAULT_WEIGHTS_ROOT))
    parser.add_argument(
        "--hf-downloader",
        choices=["hfd", "huggingface_hub"],
        default=os.environ.get("WH0_HF_DOWNLOADER", "hfd"),
        help="Downloader for Hugging Face snapshot items. Default: hfd via hf-mirror.com.",
    )
    parser.add_argument("--only", nargs="*", default=None, help="Only process the listed manifest ids")
    parser.add_argument(
        "--local-path",
        action="append",
        default=[],
        help="Link a local file/dir instead of downloading, format: <id>=</abs/path>",
    )
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    weights_root = Path(args.weights_root).expanduser().resolve()
    selected = {item["id"] for item in manifest} if not args.only else set(args.only)
    manifest = [item for item in manifest if item["id"] in selected]
    local_paths = parse_local_paths(args.local_path)

    if args.command == "list":
        print_list(manifest, weights_root)
        return 0

    for item in manifest:
        print(install_item(item, weights_root, local_paths, args.hf_downloader))

    ensure_compatibility_links(REPO_ROOT, weights_root)
    print(f"\nWeights root: {weights_root}")
    print("Compatibility links refreshed for WM-H and vitra-wh0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
