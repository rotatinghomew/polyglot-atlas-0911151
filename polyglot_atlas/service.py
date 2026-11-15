from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import parse_qs, urlparse

from .data import canonical, load_bundle, visible_snapshot
from .store import ArtifactStore, Conflict
from .registry import AlignmentRegistry, RegistryConflict
from .alignment import build_alignment_plan
from .crosslingual import cross_language_search, CrossLingualRejected


class Server(ThreadingHTTPServer):
    daemon_threads = True


def make_server(bundle, store, registry=None, host="127.0.0.1", port=0):
    registry = registry if registry is not None else AlignmentRegistry(":memory:")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def respond(self, status, value):
            payload = canonical(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            target = urlparse(self.path)
            try:
                if target.path == "/health":
                    self.respond(200, {"status": "ok", "product": bundle["product"]})
                elif target.path == "/snapshot":
                    as_of = parse_qs(target.query).get("as_of", ["2026-01-01"])[0]
                    date.fromisoformat(as_of)
                    self.respond(200, visible_snapshot(bundle, as_of))
                elif target.path == "/policy":
                    self.respond(200, store.current_policy())
                elif target.path == "/alignments/active":
                    active = registry.active()
                    if active is None:
                        self.respond(404, {"error": "尚无生效的对齐版本", "code": "no-active-alignment"})
                    else:
                        self.respond(200, active)
                elif target.path.startswith("/alignments/"):
                    raw = target.path.rsplit("/", 1)[-1]
                    active_version, _ = registry.active_version()
                    if raw == "versions":
                        self.respond(200, {"versions": registry.versions(), "active_version": active_version})
                    else:
                        value = registry.get(int(raw))
                        self.respond(404 if value is None else 200,
                                      {"error": "对齐版本不存在"} if value is None else value)
                elif target.path == "/search/cross-lingual":
                    query = parse_qs(target.query)
                    as_of = query.get("as_of", ["2026-01-01"])[0]
                    date.fromisoformat(as_of)
                    active = registry.active()
                    if active is None:
                        self.respond(409, {"error": "尚无生效的对齐版本，拒绝跨语种检索", "code": "no-active-alignment"})
                    else:
                        snapshot = visible_snapshot(bundle, as_of)
                        if active["plan"]["snapshot_id"] != snapshot["snapshot_id"]:
                            self.respond(409, {
                                "error": "生效对齐版本基于其他资料快照，不能混用空间",
                                "code": "snapshot-mismatch",
                            })
                        else:
                            result = cross_language_search(
                                snapshot, active, query["document"][0], query["target_language"][0],
                                limit=int(query.get("limit", ["5"])[0]),
                                blocked_items=store.current_policy().get("blocked_items", []),
                            )
                            self.respond(200, result)
                elif target.path.startswith("/artifacts/"):
                    value = store.get(target.path.rsplit("/", 1)[-1])
                    self.respond(404 if value is None else 200, {"error": "成果不存在"} if value is None else value)
                else:
                    self.respond(404, {"error": "接口不存在"})
            except CrossLingualRejected as error:
                self.respond(422, {"error": str(error), "code": error.code})
            except RegistryConflict as error:
                self.respond(409, {"error": str(error)})
            except ValueError as error:
                self.respond(400, {"error": str(error)})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 16_000_000:
                    raise ValueError("请求大小不合适")
                value = json.loads(self.rfile.read(length))
                if self.path == "/artifacts":
                    artifact = value["artifact"]
                    expected = visible_snapshot(bundle, artifact["as_of"])
                    if expected["snapshot_id"] != artifact["snapshot_id"]:
                        raise ValueError("资料指纹与来源不一致")
                    self.respond(201, store.publish(value["request_key"], artifact, value["policy_revision"]))
                elif self.path == "/policy":
                    self.respond(200, store.update_policy(value["expected_revision"], value["policy"]))
                elif self.path == "/alignments":
                    # 构建、保存并在同一事务内原子切换；检索侧永远只看到完整版本。
                    as_of = value.get("as_of", "2026-01-01")
                    date.fromisoformat(as_of)
                    snapshot = visible_snapshot(bundle, as_of)
                    plan = build_alignment_plan(snapshot)
                    receipt = registry.save(plan, activate=bool(value.get("activate", True)))
                    self.respond(201, {**receipt, "plan_summary": _plan_summary(plan)})
                elif self.path == "/alignments/activate":
                    receipt = registry.activate(
                        int(value["alignment_version"]), value.get("expected_active_version")
                    )
                    self.respond(200, {"active_version": receipt["alignment_version"],
                                       "alignment_id": receipt["alignment_id"]})
                else:
                    self.respond(404, {"error": "接口不存在"})
            except Conflict as error:
                self.respond(409, {"error": str(error)})
            except RegistryConflict as error:
                self.respond(409, {"error": str(error)})
            except (ValueError, KeyError, TypeError) as error:
                self.respond(400, {"error": str(error)})
    return Server((host, port), Handler)


def _plan_summary(plan):
    return {
        "method": plan["method"],
        "snapshot_id": plan["snapshot_id"],
        "reachability": plan["reachability"],
        "pairs": [
            {
                "pair": f"{row['source_language']}->{row['target_language']}",
                "status": row["status"],
                "rejection_reasons": row.get("rejection_reasons", []),
                "selected_candidate": row.get("selected_candidate"),
                "validation": row.get("validation"),
                "diagnostics": row.get("diagnostics"),
            }
            for row in plan["pairs"]
        ],
    }


@contextmanager
def running_service(bundle, store, registry=None):
    server = make_server(bundle, store, registry)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description="启动本地资料与成果服务")
    parser.add_argument("--bundle", default="fixtures/bundle.json")
    parser.add_argument("--state", default="local-state.sqlite3")
    parser.add_argument("--alignment-state", default="local-alignments.sqlite3")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    store = ArtifactStore(args.state, bundle.get("policy"))
    registry = AlignmentRegistry(args.alignment_state)
    server = make_server(bundle, store, registry, port=args.port)
    print("本地服务已启动", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()
        registry.close()


if __name__ == "__main__":
    main()
