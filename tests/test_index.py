import json
import shutil
from pathlib import Path

import pytest

from conftest import PDF
from tcb import build, conf, index
from tcb.cli import main
from tcb.index import IndexError_

EMB = dict(embed_model="TEST-HASH", embed_rev="test")


def clone(built, tmp_path) -> Path:
    d = tmp_path / "idx"
    shutil.copytree(built[0], d)
    return d


def test_validate_ok(built):
    m = index.validate(built[0], **EMB)
    assert m["counts"]["nodes"] == sum(m["counts"]["by_type"].values())
    assert m["source"]["pdf_pages"] == 197 and m["source"]["printed_pages_inferred"] == 13
    assert m["embedding"]["dim"] == 64 and len(m["source"]["sha256"]) == 64


def test_manifest_fields(built):
    m = built[1]
    for k in ("index_format_version", "source", "embedding", "reranker_expected", "chunking", "bm25",
              "counts", "runtime_requirements", "files"):
        assert k in m
    assert m["reranker_expected"] == {"model": conf.RERANK_MODEL, "revision": conf.RERANK_REV}


def test_real_embed_model_mismatch(built):
    with pytest.raises(IndexError_) as e:
        index.validate(built[0])           # code expects BAAI/bge-m3, index says TEST-HASH
    assert e.value.code == "embedding"


def test_missing_dir_and_manifest(tmp_path):
    with pytest.raises(IndexError_) as e: index.validate(tmp_path / "nope")
    assert e.value.code == "missing"


def test_corrupt_manifest(built, tmp_path):
    d = clone(built, tmp_path); (d / "manifest.json").write_text("{not json")
    with pytest.raises(IndexError_) as e: index.validate(d, **EMB)
    assert e.value.code == "corrupt"


def test_missing_file(built, tmp_path):
    d = clone(built, tmp_path); (d / "gloss.json").unlink()
    with pytest.raises(IndexError_) as e: index.validate(d, **EMB)
    assert e.value.code == "missing"


def test_tampered_file(built, tmp_path):
    d = clone(built, tmp_path)
    p = d / "nodes.jsonl"; p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(IndexError_) as e: index.validate(d, **EMB)
    assert e.value.code == "corrupt" and "sha256" in str(e.value)


def test_unlisted_file(built, tmp_path):
    d = clone(built, tmp_path); (d / "extra.bin").write_bytes(b"x")
    with pytest.raises(IndexError_) as e: index.validate(d, **EMB)
    assert "unlisted" in str(e.value)


def test_format_version(built, tmp_path):
    d = clone(built, tmp_path)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8")); m["index_format_version"] = 99
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(IndexError_) as e: index.validate(d, **EMB)
    assert e.value.code == "format"


def test_count_mismatch(built, tmp_path):
    d = clone(built, tmp_path)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8")); m["counts"]["nodes"] += 1
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(IndexError_) as e: index.validate(d, **EMB)
    assert e.value.code == "corrupt"


def test_snapshot_check_fails_without_models(built):
    with pytest.raises(IndexError_) as e: index.validate(built[0], check_snapshot=True, **EMB)
    assert e.value.code == "snapshot"


def test_error_message_has_remediation(tmp_path):
    with pytest.raises(IndexError_) as e: index.validate(tmp_path / "x")
    assert "tcb build" in str(e.value) and "git pull" in str(e.value)


def test_cli_exit_3(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "nope")]) == 3
    assert "INDEX ERROR" in capsys.readouterr().err


def test_refuses_overwrite(built, embedder):
    with pytest.raises(FileExistsError):
        build.build(str(PDF), str(built[0]), embedder=embedder)


def test_deterministic_rebuild(built, tmp_path, embedder):
    out = tmp_path / "again"
    m2 = build.build(str(PDF), str(out), embedder=embedder, embed_name="TEST-HASH", embed_rev="test")
    for rel in ("nodes.jsonl", "bm25/ids.json", "tables.jsonl", "gloss.json", "pages.json"):
        assert m2["files"][rel] == built[1]["files"][rel], rel
    # lexical index identical (vocab + ids); dense vectors identical for the deterministic test embedder
    assert m2["files"]["bm25/vocab.index.json"] == built[1]["files"]["bm25/vocab.index.json"]


def test_load_and_retrieve(built, embedder):
    ix = index.Index.load(built[0], embedder, **EMB)
    from tcb import vi
    res, sc = ix.bm25.retrieve([vi.tokens("Tỷ lệ CASA 40,4%")], k=5)
    top = [ix.nodes[ix.ids[i]] for i in res[0]]
    assert any(n["printed_page_number"] == 5 for n in top)       # BM25 finds the KPI page
    dense = ix.vector.as_retriever(similarity_top_k=5).retrieve("Tổng tài sản 1.192 nghìn tỷ đồng")
    assert dense and dense[0].node.node_id in ix.nodes
