from pathlib import Path
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyglot_atlas.client import ProductClient
from polyglot_atlas.crosslingual import CrossLingualRejected
from polyglot_atlas.data import load_bundle
from polyglot_atlas.registry import AlignmentRegistry
from polyglot_atlas.service import running_service
from polyglot_atlas.store import ArtifactStore
from polyglot_atlas.workflow import (
    cross_lingual_material_search,
    publish_cross_lingual_alignment,
    publish_overview,
)


def main():
    root = Path(__file__).resolve().parents[1]
    bundle = load_bundle(root / "fixtures/bundle.json")
    with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(str(Path(directory) / "state.sqlite3"), bundle.get("policy"))
        registry = AlignmentRegistry(str(Path(directory) / "alignments.sqlite3"))
        try:
            with running_service(bundle, store, registry) as base:
                client = ProductClient(base)
                published = publish_overview(client, "2026-01-01", "overview-demo")
                restored = client.artifact(published["artifact_id"])
                assert restored == published["artifact"]

                # 构建并原子激活跨语种对齐方案。
                receipt = publish_cross_lingual_alignment(client, "2026-01-01")
                plan_summary = receipt["plan_summary"]
                accepted = [p["pair"] for p in plan_summary["pairs"] if p["status"] == "accepted"]
                rejected = {
                    p["pair"]: p["rejection_reasons"]
                    for p in plan_summary["pairs"]
                    if p["status"] == "rejected"
                }

                # 单跳 zh->en 与两跳 zh->en->ar 检索。
                direct = cross_lingual_material_search(client, "zh-2", "en", "2026-01-01", limit=3)
                two_hop = cross_lingual_material_search(client, "zh-2", "ar", "2026-01-01", limit=3)

                # 无可靠连接的语种必须明确拒配（fr 同时不在权限内，先展示权限拒配）。
                unreachable = None
                try:
                    cross_lingual_material_search(client, "zh-2", "fr", "2026-01-01")
                except CrossLingualRejected as error:
                    unreachable = {"code": error.code, "error": error.reason}

                output = {
                    "产品": bundle["product"],
                    "概览发布编号": published["artifact_id"],
                    "对齐版本": receipt["alignment_version"],
                    "对齐编号": receipt["alignment_id"],
                    "已接受语种对": accepted,
                    "可达性": plan_summary["reachability"],
                    "拒配语种对示例": dict(list(rejected.items())[:3]),
                    "单跳检索_zh到en": {
                        "路径": direct["alignment_path"],
                        "可信阈值": direct["trust_band"]["credible_threshold"],
                        "候选": [
                            {"文档": r["document"], "分数": r["score"], "可信": r["credible"],
                             "对齐版本": r["alignment_version"]}
                            for r in direct["results"]
                        ],
                    },
                    "两跳检索_zh到ar": {
                        "路径": two_hop["alignment_path"],
                        "候选": [{"文档": r["document"], "分数": r["score"]} for r in two_hop["results"]],
                    },
                    "无连接拒配": unreachable,
                }
                print(json.dumps(output, ensure_ascii=False, indent=2))
        finally:
            store.close()
            registry.close()


if __name__ == "__main__":
    main()
