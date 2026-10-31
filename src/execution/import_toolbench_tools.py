from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.registry import SQLiteRegistry
from generation.tool_generator import register_tool_from_spec
from generation.toolbench_catalog import (
    endpoint_to_code,
    endpoint_to_spec,
    load_toolbench_endpoints,
    select_toolbench_endpoints,
)


def import_toolbench_tools(
    *,
    db_path: str,
    tools_dir: str,
    toolbench_dir: str,
    query: str = "",
    max_tools: int = 0,
    reset_db: bool = False,
) -> int:
    if reset_db and os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(tools_dir, exist_ok=True)

    if query:
        endpoints = select_toolbench_endpoints(
            toolbench_dir,
            request_text=query,
            count=max_tools or len(load_toolbench_endpoints(toolbench_dir)),
        )
    else:
        endpoints = load_toolbench_endpoints(toolbench_dir)
        if max_tools > 0:
            endpoints = endpoints[:max_tools]

    with SQLiteRegistry(db_path) as registry:
        for endpoint in endpoints:
            register_tool_from_spec(
                registry,
                endpoint_to_spec(endpoint),
                endpoint_to_code(endpoint),
                tool_id=endpoint["tool_id"],
                save_dir=tools_dir,
                skip_existing=False,
            )
    return len(endpoints)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import real ToolBench tools into generated_tools")
    parser.add_argument("--db", default="demo_registry.sqlite")
    parser.add_argument("--tools_dir", default="generated_tools")
    parser.add_argument("--toolbench_dir", default="tools")
    parser.add_argument("--query", default="")
    parser.add_argument("--max_tools", type=int, default=0)
    parser.add_argument("--reset_db", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = import_toolbench_tools(
        db_path=args.db,
        tools_dir=args.tools_dir,
        toolbench_dir=args.toolbench_dir,
        query=args.query,
        max_tools=args.max_tools,
        reset_db=args.reset_db,
    )
    print(f"imported_toolbench_tools: {count}")


if __name__ == "__main__":
    main()
