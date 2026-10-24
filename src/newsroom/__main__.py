"""Command line: ``python -m newsroom demo | cycle | serve``."""

from __future__ import annotations

import argparse
import sys

from newsroom.config import get_settings
from newsroom.demo import run_demo


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="newsroom", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Offline end-to-end walkthrough on fixture articles.")
    demo.add_argument("--community", default="lapland-seasonal")
    demo.add_argument(
        "--no-approve",
        action="store_true",
        help="Stop at the human gate instead of simulating named reviewers.",
    )

    cycle = sub.add_parser("cycle", help="Fetch the live feeds once and prepare stories.")
    cycle.add_argument("--community", action="append", dest="communities")

    serve = sub.add_parser("serve", help="Run the HTTP API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)

    args = parser.parse_args(argv)

    if args.command == "demo":
        return run_demo(community_id=args.community, approve=not args.no_approve)

    if args.command == "cycle":
        from newsroom.service import Newsroom

        newsroom = Newsroom(persistent_checkpoints=True)
        report = newsroom.run_cycle(args.communities)
        print(
            f"run {report.run_id}: {report.new_articles} new articles, {report.events} events, "
            f"{report.pitches} pitches"
        )
        for run in report.runs:
            print(f"  [{run.status}] {run.community_id}: {run.working_title}")
        return 0

    if args.command == "serve":
        import uvicorn

        settings = get_settings()
        print(f"LLM provider: {settings.llm_provider}")
        uvicorn.run("newsroom.api.app:app", host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
