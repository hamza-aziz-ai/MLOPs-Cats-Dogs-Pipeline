"""Drop builtin-exception shadow nodes graphify's AST extractor collapses
across files (e.g. every ``raise ValueError(...)`` in the repo resolves to one
shared ``valueerror`` node, fabricating cross-file/cross-community edges).

Root cause is graphify's ``ensure_named_node`` fallback (shared by 20+ language
extractors in the vendored `graphify` package), so this is a repo-side
post-process instead of a package patch -- it survives `pip install -U
graphifyy` and any --update/--watch rebuild.

Usage:
    python scripts/graphify_clean.py            # clean once
    python scripts/graphify_clean.py --watch     # run graphify's watcher, re-clean after every rebuild
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

GRAPH_PATH = Path("graphify-out/graph.json")

# ponytail: name-based denylist of Python builtin exceptions, not a general
# fix for ensure_named_node collapse in other languages. Extend this set if
# another language's builtins show up the same way.
BUILTIN_EXCEPTIONS = {
    "valueerror", "typeerror", "keyerror", "indexerror", "attributeerror",
    "importerror", "filenotfounderror", "oserror", "notimplementederror",
    "assertionerror", "runtimeerror", "stopiteration", "zerodivisionerror",
    "arithmeticerror", "lookuperror", "nameerror", "unboundlocalerror",
    "recursionerror", "memoryerror", "overflowerror", "floatingpointerror",
    "referenceerror", "connectionerror", "timeouterror", "permissionerror",
    "exception",
}


def clean_graph(graph: dict) -> tuple[dict, int]:
    """Return (cleaned_graph, dropped_count). Pure function, no I/O."""
    nodes = graph["nodes"]
    drop_ids = {
        n["id"] for n in nodes
        if not n.get("source_file") and n["id"] in BUILTIN_EXCEPTIONS
    }
    if not drop_ids:
        return graph, 0
    graph["nodes"] = [n for n in nodes if n["id"] not in drop_ids]
    graph["links"] = [
        l for l in graph["links"]
        if l.get("source") not in drop_ids and l.get("target") not in drop_ids
    ]
    return graph, len(drop_ids)


def clean_and_reexport() -> int:
    if not GRAPH_PATH.exists():
        return 0
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    graph, dropped = clean_graph(graph)
    if not dropped:
        return 0
    GRAPH_PATH.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    subprocess.run(["graphify", "cluster-only", "."], check=True)
    if Path("graphify-out/obsidian").exists():
        subprocess.run(["graphify", "export", "obsidian"], check=True)
    subprocess.run(["graphify", "export", "html"], check=True)
    return dropped


def watch(watch_path: str = ".", debounce: int = 3) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "graphify.watch", watch_path, "--debounce", str(debounce)],
    )
    last_mtime = GRAPH_PATH.stat().st_mtime if GRAPH_PATH.exists() else 0.0
    try:
        while proc.poll() is None:
            time.sleep(2)
            if not GRAPH_PATH.exists():
                continue
            mtime = GRAPH_PATH.stat().st_mtime
            if mtime != last_mtime:
                time.sleep(1)  # let graphify finish writing report/html
                dropped = clean_and_reexport()
                if dropped:
                    print(f"[graphify_clean] dropped {dropped} builtin-exception shadow node(s) after rebuild", flush=True)
                last_mtime = GRAPH_PATH.stat().st_mtime
    finally:
        proc.terminate()


def _demo() -> None:
    """ponytail: smallest runnable check for clean_graph's branch logic."""
    sample = {
        "nodes": [
            {"id": "valueerror", "source_file": ""},
            {"id": "real_module_foo", "source_file": "foo.py"},
        ],
        "links": [
            {"source": "real_module_foo", "target": "valueerror"},
            {"source": "real_module_foo", "target": "real_module_foo"},
        ],
    }
    cleaned, dropped = clean_graph(sample)
    assert dropped == 1, dropped
    assert {n["id"] for n in cleaned["nodes"]} == {"real_module_foo"}
    assert len(cleaned["links"]) == 1
    empty, dropped2 = clean_graph({"nodes": [{"id": "real_module_foo", "source_file": "foo.py"}], "links": []})
    assert dropped2 == 0
    print("ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    elif "--watch" in sys.argv:
        watch()
    else:
        n = clean_and_reexport()
        print(f"dropped {n} builtin-exception shadow node(s)" if n else "graph clean, nothing to do")
