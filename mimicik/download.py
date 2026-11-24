"""Download one pinned, checksummed model bundle without the historical zoo."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from importlib.resources import files
from pathlib import Path

from .checkpoint import load_bundle, sha256

BUNDLE_FILES = {"config.json", "normalizer.json", "model.safetensors", "metadata.json"}


def catalog():
    return json.loads(files("mimicik").joinpath("models.json").read_text())


def unpack_verified(archive, target, entry):
    """Extract only the four expected regular files after checksum validation."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    if sha256(archive) != entry["sha256"]:
        raise ValueError("Downloaded archive SHA256 mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        with zipfile.ZipFile(archive) as source:
            names = [member.filename for member in source.infolist()]
            if set(names) != BUNDLE_FILES or len(names) != len(BUNDLE_FILES):
                raise ValueError(
                    "Model archive must contain exactly four root-level bundle files"
                )
            if (
                sum(member.file_size for member in source.infolist())
                > entry["unpacked_bytes"]
            ):
                raise ValueError("Archive exceeds declared uncompressed size")
            for member in source.infolist():
                if stat.S_ISLNK(member.external_attr >> 16) or member.is_dir():
                    raise ValueError("Archive entries must be regular files")
                with (
                    source.open(member) as reader,
                    (staging / member.filename).open("wb") as writer,
                ):
                    shutil.copyfileobj(reader, writer)
        # Validate manifests, architecture and tensors before installing it.
        load_bundle(staging)
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def download_model(model, output):
    records = catalog()
    canonical = records.get("aliases", {}).get(model, model)
    if canonical not in records["models"]:
        raise ValueError(f"Unknown model {model!r}; use --list")
    entry = records["models"][canonical]
    output = Path(output).expanduser()
    if output.exists():
        if not output.is_dir() or not all(
            (output / name).is_file() for name in BUNDLE_FILES
        ):
            raise FileExistsError(f"Refusing to overwrite {output}")
        if any(
            sha256(output / name) != expected
            for name, expected in entry["files"].items()
        ):
            raise FileExistsError(
                f"Existing output differs from {canonical}; use a different path"
            )
        load_bundle(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with tempfile.TemporaryDirectory(
        prefix="mimicik-download-", dir=output.parent
    ) as scratch:
        archive = Path(scratch) / "model.zip"
        for url in entry["urls"]:
            if not url.startswith("https://"):
                raise ValueError("Official downloads require HTTPS")
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "MimicIK/0.1.0"}
                )
                digest, count = hashlib.sha256(), 0
                with (
                    urllib.request.urlopen(request, timeout=60) as response,
                    archive.open("wb") as writer,
                ):
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        count += len(chunk)
                        if count > entry["bytes"]:
                            raise ValueError("Download exceeds the published file size")
                        digest.update(chunk)
                        writer.write(chunk)
                if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise ValueError(
                        "Downloaded file differs from its published size or SHA256"
                    )
                return unpack_verified(archive, output, entry)
            except (OSError, urllib.error.URLError, ValueError) as exc:
                failures.append(f"{url}: {exc}")
        raise RuntimeError("Could not download the model:\n" + "\n".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="airbot")
    parser.add_argument("--output", default="checkpoints/airbot")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        records = catalog()
        for name, entry in records["models"].items():
            print(f"{name}: {entry['description']}")
        print("Aliases:", records["aliases"])
        return
    print(f"Verified model: {download_model(args.model, args.output)}")


if __name__ == "__main__":
    main()
