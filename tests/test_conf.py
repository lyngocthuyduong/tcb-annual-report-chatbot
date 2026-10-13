import importlib


def test_models_not_env_configurable(monkeypatch):
    from tcb import conf
    monkeypatch.setenv("EMBED_MODEL", "x/y"); monkeypatch.setenv("FINAL_EVIDENCE", "3")
    monkeypatch.setenv("EMBED_REV", "deadbeef"); monkeypatch.setenv("RRF_K", "1")
    c = importlib.reload(conf)
    assert (c.EMBED_MODEL, c.FINAL_EVIDENCE, c.EMBED_REV, c.RRF_K) == (
        "BAAI/bge-m3", 12, "5617a9f61b028005a4858fdac845db406aefb181", 60)
    assert not hasattr(c, "RERANK_MODEL") and not hasattr(c, "TAU_LOW")


def test_published_questions_present_and_untouched():
    import json
    from pathlib import Path
    q = json.loads((Path(__file__).parent.parent / "data" / "q10.json").read_text(encoding="utf-8"))
    assert [x["id"] for x in q] == [f"sq-{i:02d}" for i in range(1, 11)]
