"""Remove built-in exception shadow nodes introduced by the graph extractor.

The extractor can collapse every reference to the same Python exception into a
single node without a source. For example, separate ``raise ValueError(...)``
calls can create false cross-file and cross-community edges.

The package-level cause is the shared ``ensure_named_node`` fallback used by
multiple language extractors. This repository applies a post-processing step so
package upgrades and graph rebuilds do not overwrite the correction.

Run the script without arguments to clean once, or pass ``--watch`` to clean
after every graph rebuild.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

GRAPH_PATH = Path("graphify-out/graph.json")

# Name-based denylist of Python built-in exceptions, not a general
# fix for ensure_named_node collapse in other languages. Extend this set if
# another language's builtins show up the same way.
BUILTIN_EXCEPTION_IDS = {
    exception_type.__name__.casefold()
    for exception_type in (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        FileNotFoundError,
        OSError,
        NotImplementedError,
        AssertionError,
        RuntimeError,
        StopIteration,
        ZeroDivisionError,
        ArithmeticError,
        LookupError,
        NameError,
        UnboundLocalError,
        RecursionError,
        MemoryError,
        OverflowError,
        FloatingPointError,
        ReferenceError,
        ConnectionError,
        TimeoutError,
        PermissionError,
        Exception,
    )
}


def clean_graph(graph: dict) -> tuple[dict, int]:
    """Return (cleaned_graph, dropped_count). Pure function, no I/O."""
    nodes = graph["nodes"]
    drop_ids = {
        node["id"]
        for node in nodes
        if not node.get("source_file") and node["id"] in BUILTIN_EXCEPTION_IDS
    }
    if not drop_ids:
        return graph, 0
    graph["nodes"] = [node for node in nodes if node["id"] not in drop_ids]
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
    """Exercise both branches of ``clean_graph`` with a minimal graph."""
    value_error_id = ValueError.__name__.casefold()
    sample = {
        "nodes": [
            {"id": value_error_id, "source_file": ""},
            {"id": "real_module_foo", "source_file": "foo.py"},
        ],
        "links": [
            {"source": "real_module_foo", "target": value_error_id},
            {"source": "real_module_foo", "target": "real_module_foo"},
        ],
    }
    cleaned, dropped = clean_graph(sample)
    assert dropped == 1, dropped
    assert {node["id"] for node in cleaned["nodes"]} == {"real_module_foo"}
    assert len(cleaned["links"]) == 1
    empty, dropped2 = clean_graph({"nodes": [{"id": "real_module_foo", "source_file": "foo.py"}], "links": []})
    assert dropped2 == 0
    print("ok")


def main() -> None:
    if "--selftest" in sys.argv:
        _demo()
    elif "--watch" in sys.argv:
        watch()
    else:
        dropped = clean_and_reexport()
        message = (
            f"dropped {dropped} built-in exception shadow node(s)"
            if dropped
            else "graph clean, nothing to do"
        )
        print(message)


if __name__ == "__main__":
    main()
