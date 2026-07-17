"""The `.vp` model file format.

A `.vp` file is a zip archive bundling everything that makes up one trained
"virtual person": the checkpoint weights, that person's persistent memory (if
any), a manifest of the training data that produced the checkpoint, and
metadata. It exists so a trained person can be moved, backed up, or shared as
a single file rather than a scattered checkpoint + memory + corpus.

Layout inside the zip::

    metadata.json           - name, created_at, architecture, provenance
    model.pt                - the NodeLinkSpikeModel checkpoint, unmodified
    memory.sqlite3          - the agent's MemoryStore database, if supplied
    training_data/manifest.json   - per-source category, record/char counts, sha256
    training_data/<files>          - copies of the source files, unless data was
                                     packed with embed_data=False (manifest only)

Nothing here changes how checkpoints or memory databases are read elsewhere;
`model.pt` and `memory.sqlite3` are read and written with the exact same
`spike_training.save_checkpoint` / `sqlite3` formats already used everywhere
else in the project, so a `.vp` archive's contents are also usable directly,
outside the archive, with existing tools.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

VP_FORMAT_VERSION = 1


@dataclass(slots=True, frozen=True)
class TrainingSourceManifestEntry:
    category: str
    path: str
    records: int
    characters: int
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_records_and_chars(path: Path) -> tuple[int, int]:
    if path.is_dir():
        records = characters = 0
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".txt", ".jsonl"}:
                r, c = _count_records_and_chars(child)
                records += r
                characters += c
        return records, characters

    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        records = sum(1 for line in text.splitlines() if line.strip())
    else:
        records = sum(1 for para in text.split("\n\n") if para.strip())
    return records, len(text)


def build_training_manifest(sources: list[tuple[str, str]]) -> list[TrainingSourceManifestEntry]:
    """Build manifest entries from a list of (path, category) pairs."""
    entries = []
    for raw_path, category in sources:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Training source not found: {path}")
        records, characters = _count_records_and_chars(path)
        # Directories don't hash cleanly as one file; hash their sorted file list instead.
        if path.is_dir():
            digest = hashlib.sha256()
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    digest.update(_sha256_file(child).encode("ascii"))
            sha = digest.hexdigest()
        else:
            sha = _sha256_file(path)
        entries.append(
            TrainingSourceManifestEntry(
                category=category, path=str(path), records=records,
                characters=characters, sha256=sha,
            )
        )
    return entries


def pack(
    output: str | Path,
    *,
    checkpoint_path: str | Path,
    name: str,
    memory_path: str | Path | None = None,
    training_sources: list[tuple[str, str]] | None = None,
    embed_data: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Package a checkpoint (+ optional memory + training-data manifest) into a `.vp` file.

    ``training_sources`` is a list of ``(path, category)`` pairs, matching the
    trainer workspace's registered sources. If ``embed_data`` is True, source
    files are copied into the archive; otherwise only the manifest (with
    hashes) is stored, keeping the archive small while still recording exactly
    what data produced this checkpoint.
    """
    output = Path(output)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    manifest_entries = build_training_manifest(training_sources or [])

    metadata = {
        "vp_format_version": VP_FORMAT_VERSION,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_filename": "model.pt",
        "has_memory": memory_path is not None,
        "training_data_embedded": embed_data,
        **(extra_metadata or {}),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        archive.write(checkpoint_path, "model.pt")

        if memory_path is not None:
            memory_path = Path(memory_path).expanduser().resolve()
            if memory_path.is_file():
                # Snapshot via sqlite's own backup API rather than a raw file
                # copy, so an in-use database is captured in a consistent state.
                with tempfile.TemporaryDirectory() as tmp:
                    snapshot_path = Path(tmp) / "memory.sqlite3"
                    source_conn = sqlite3.connect(str(memory_path))
                    dest_conn = sqlite3.connect(str(snapshot_path))
                    with dest_conn:
                        source_conn.backup(dest_conn)
                    source_conn.close()
                    dest_conn.close()
                    archive.write(snapshot_path, "memory.sqlite3")

        manifest_payload = [
            {
                "category": entry.category,
                "path": entry.path,
                "records": entry.records,
                "characters": entry.characters,
                "sha256": entry.sha256,
            }
            for entry in manifest_entries
        ]
        archive.writestr(
            "training_data/manifest.json",
            json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        )

        if embed_data:
            for entry in manifest_entries:
                source_path = Path(entry.path)
                if source_path.is_dir():
                    for child in sorted(source_path.rglob("*")):
                        if child.is_file():
                            arcname = f"training_data/files/{source_path.name}/{child.relative_to(source_path)}"
                            archive.write(child, arcname)
                else:
                    arcname = f"training_data/files/{source_path.name}"
                    archive.write(source_path, arcname)

    return output


@dataclass(slots=True, frozen=True)
class VpContents:
    metadata: dict[str, Any]
    checkpoint_path: Path
    memory_path: Path | None
    training_manifest: list[dict[str, Any]]
    extracted_dir: Path


def unpack(vp_path: str | Path, destination: str | Path) -> VpContents:
    """Extract a `.vp` archive into ``destination`` and return paths to its parts."""
    vp_path = Path(vp_path).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with ZipFile(vp_path, "r") as archive:
        archive.extractall(destination)

    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    checkpoint_path = destination / "model.pt"
    memory_candidate = destination / "memory.sqlite3"
    memory_path = memory_candidate if memory_candidate.is_file() else None

    manifest_path = destination / "training_data" / "manifest.json"
    training_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else []
    )

    return VpContents(
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        memory_path=memory_path,
        training_manifest=training_manifest,
        extracted_dir=destination,
    )


def inspect(vp_path: str | Path) -> dict[str, Any]:
    """Read a `.vp` archive's metadata and manifest without extracting the checkpoint."""
    vp_path = Path(vp_path).expanduser().resolve()
    with ZipFile(vp_path, "r") as archive:
        metadata = json.loads(archive.read("metadata.json"))
        try:
            manifest = json.loads(archive.read("training_data/manifest.json"))
        except KeyError:
            manifest = []
        names = archive.namelist()
    return {
        "metadata": metadata,
        "training_manifest": manifest,
        "has_memory": "memory.sqlite3" in names,
        "archive_entries": len(names),
    }
