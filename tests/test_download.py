import hashlib
import zipfile

import pytest

from mimicik.download import unpack_verified


def test_archive_path_traversal_is_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("../outside", "unsafe")
    entry = {
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "unpacked_bytes": 100,
    }
    with pytest.raises(ValueError, match="exactly four"):
        unpack_verified(archive, tmp_path / "model", entry)
    assert not (tmp_path / "outside").exists()
    assert not (tmp_path / "model").exists()


def test_bad_download_hash_does_not_install(tmp_path):
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="SHA256"):
        unpack_verified(archive, tmp_path / "model", {"sha256": "0" * 64})
    assert not (tmp_path / "model").exists()
