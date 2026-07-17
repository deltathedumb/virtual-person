import sqlite3

import pytest
import torch

from virtual_person.spiking import NodeLinkSpikeModel, SpikingModelConfig
from virtual_person.spike_training import save_checkpoint
from virtual_person.vp_package import inspect, pack, unpack


@pytest.fixture()
def tiny_checkpoint(tmp_path):
    config = SpikingModelConfig(hidden_size=8, layer_count=1, ticks_per_token=1)
    model = NodeLinkSpikeModel(config)
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, model, step=7, metadata={"stage": "test"})
    return path


@pytest.fixture()
def sample_memory_db(tmp_path):
    path = tmp_path / "memory.sqlite3"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY, sim_time REAL, kind TEXT, "
        "summary TEXT, importance REAL, metadata_json TEXT)"
    )
    conn.execute(
        "INSERT INTO memories (sim_time, kind, summary, importance, metadata_json) "
        "VALUES (0.0, 'goal', 'Test memory row', 0.5, '{}')"
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def training_source(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("First paragraph.\n\nSecond paragraph here.\n", encoding="utf-8")
    return path


def test_pack_and_inspect_without_memory_or_data(tmp_path, tiny_checkpoint):
    output = tmp_path / "model.vp"
    result = pack(output, checkpoint_path=tiny_checkpoint, name="TestPerson")

    assert result == output
    assert output.is_file()

    info = inspect(output)
    assert info["metadata"]["name"] == "TestPerson"
    assert info["metadata"]["vp_format_version"] == 1
    assert info["has_memory"] is False
    assert info["training_manifest"] == []


def test_pack_embeds_memory_as_valid_sqlite(tmp_path, tiny_checkpoint, sample_memory_db):
    output = tmp_path / "model.vp"
    pack(output, checkpoint_path=tiny_checkpoint, name="TestPerson", memory_path=sample_memory_db)

    result = unpack(output, tmp_path / "extracted")
    assert result.memory_path is not None
    assert result.memory_path.is_file()

    conn = sqlite3.connect(str(result.memory_path))
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()
    assert count == 1


def test_pack_records_training_manifest_with_hash(tmp_path, tiny_checkpoint, training_source):
    output = tmp_path / "model.vp"
    pack(
        output,
        checkpoint_path=tiny_checkpoint,
        name="TestPerson",
        training_sources=[(str(training_source), "English")],
        embed_data=True,
    )

    info = inspect(output)
    assert len(info["training_manifest"]) == 1
    entry = info["training_manifest"][0]
    assert entry["category"] == "English"
    assert entry["records"] == 2
    assert len(entry["sha256"]) == 64


def test_no_embed_data_keeps_manifest_but_skips_file_copy(tmp_path, tiny_checkpoint, training_source):
    embedded_output = tmp_path / "embedded.vp"
    manifest_only_output = tmp_path / "manifest_only.vp"

    pack(
        embedded_output, checkpoint_path=tiny_checkpoint, name="A",
        training_sources=[(str(training_source), "English")], embed_data=True,
    )
    pack(
        manifest_only_output, checkpoint_path=tiny_checkpoint, name="A",
        training_sources=[(str(training_source), "English")], embed_data=False,
    )

    assert manifest_only_output.stat().st_size < embedded_output.stat().st_size
    manifest_info = inspect(manifest_only_output)
    assert manifest_info["metadata"]["training_data_embedded"] is False
    assert len(manifest_info["training_manifest"]) == 1


def test_unpacked_checkpoint_loads_correctly(tmp_path, tiny_checkpoint):
    output = tmp_path / "model.vp"
    pack(output, checkpoint_path=tiny_checkpoint, name="TestPerson")

    result = unpack(output, tmp_path / "extracted")
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["step"] == 7
    assert payload["metadata"]["stage"] == "test"


def test_pack_raises_on_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        pack(tmp_path / "out.vp", checkpoint_path=tmp_path / "missing.pt", name="X")
