from __future__ import annotations

from .product import summarize
from .crosslingual import CrossLingualRejected


def publish_overview(client, as_of, request_key):
    snapshot = client.snapshot(as_of)
    policy = client.policy()
    artifact = {
        "as_of": as_of,
        "snapshot_id": snapshot["snapshot_id"],
        "method": "observed-overview-v1",
        "result": summarize(snapshot),
    }
    return client.publish(request_key, artifact, policy["revision"])


def publish_cross_lingual_alignment(client, as_of, activate=True):
    """构建并（默认原子）激活跨语种对齐方案，返回保存回执与方案摘要。"""
    return client.build_alignment(as_of, activate=activate)


def cross_lingual_material_search(client, document_id, target_language, as_of, limit=5):
    """跨语种检索入口：无可靠连接等拒配以 CrossLingualRejected 显式返回。"""
    try:
        return client.cross_lingual_search(document_id, target_language, as_of, limit=limit)
    except Exception as error:  # 保留远端状态码，包装为本地拒配类型
        status = getattr(error, "status", None)
        if status in (409, 422):
            body = getattr(error, "body", {}) or {}
            raise CrossLingualRejected(
                body.get("error", str(error)), code=body.get("code", "rejected")
            ) from error
        raise
