from pathlib import Path
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyglot_atlas.client import ProductClient
from polyglot_atlas.data import load_bundle
from polyglot_atlas.service import running_service
from polyglot_atlas.store import ArtifactStore
from polyglot_atlas.workflow import publish_overview


def main():
    root = Path(__file__).resolve().parents[1]
    bundle = load_bundle(root / "fixtures/bundle.json")
    with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(str(Path(directory) / "state.sqlite3"), bundle.get("policy"))
        try:
            with running_service(bundle, store) as base:
                client = ProductClient(base)
                published = publish_overview(client, "2026-01-01", "overview-demo")
                restored = client.artifact(published["artifact_id"])
                assert restored == published["artifact"]
                print(json.dumps({"产品": bundle["product"], "发布编号": published["artifact_id"], "原始资料概览": restored["result"]}, ensure_ascii=False, indent=2))
        finally:
            store.close()


if __name__ == "__main__":
    main()
