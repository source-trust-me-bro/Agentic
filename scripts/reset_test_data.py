"""Empty the test databases so a test run starts clean. Destructive, so it is opt-in.

    python scripts/reset_test_data.py                      # preview: counts only, deletes nothing
    python scripts/reset_test_data.py --wipe agentic_newsroom,multilingual_news

The databases come from the two projects' .env files (MONGODB_URI / MONGODB_DATABASE). Nothing is
deleted unless ``--wipe`` names exactly the databases to empty. Collections are emptied, not
dropped, so indexes stay. Clerk sign-in accounts are not in these databases: after a wipe, people
can still sign in and get a fresh empty profile.

``--checkpoints`` also deletes the newsroom's LangGraph checkpoint file, because paused reviews in
it point at stories that no longer exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import dotenv_values
from pymongo import MongoClient

PROJECTS = {
    "newsroom": Path(r"D:\Agentic\.env"),
    "backend": Path(r"D:\Backend\.env"),
}
CHECKPOINTS = Path(r"D:\Agentic\data\checkpoints.sqlite")


def targets() -> dict[str, tuple[str, str]]:
    """database name -> (project, uri)"""

    found: dict[str, tuple[str, str]] = {}
    for project, path in PROJECTS.items():
        values = dotenv_values(path)
        uri, name = values.get("MONGODB_URI"), values.get("MONGODB_DATABASE")
        if uri and name:
            found[name] = (project, uri)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--wipe", default="", help="comma-separated database names to empty")
    parser.add_argument("--checkpoints", action="store_true", help="also delete the LangGraph checkpoint file")
    parser.add_argument(
        "--keep",
        default="",
        help="comma-separated collections to leave alone, e.g. users (keeps accounts and their editor/admin roles)",
    )
    args = parser.parse_args()
    keep = {c.strip() for c in args.keep.split(",") if c.strip()}

    dbs = targets()
    wipe = {n.strip() for n in args.wipe.split(",") if n.strip()}
    unknown = wipe - set(dbs)
    if unknown:
        print(f"Unknown database(s): {', '.join(sorted(unknown))}. Known: {', '.join(dbs)}")
        return 2

    for name, (project, uri) in dbs.items():
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        db = client[name]
        print(f"\n{name}  ({project})")
        total = 0
        for collection in sorted(db.list_collection_names()):
            count = db[collection].count_documents({})
            total += count
            kept = collection in keep
            print(f"  {collection:28s} {count:>8}" + ("   (kept)" if kept and name in wipe else ""))
            if name in wipe and count and not kept:
                db[collection].delete_many({})
        print(f"  {'TOTAL':28s} {total:>8}" + ("   -> emptied" if name in wipe else "   (kept)"))

    if args.checkpoints and wipe:
        if CHECKPOINTS.exists():
            CHECKPOINTS.unlink()
            print(f"\nDeleted {CHECKPOINTS}")
    if not wipe:
        print("\nPreview only. Nothing was deleted. To empty databases, name them with --wipe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
