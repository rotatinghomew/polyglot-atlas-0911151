"""对齐方案的版本化存储与原子切换。

一次对齐更新在单个数据库事务内完成“写入新版本 + 切换生效指针”，检索侧
读取的是激活版本的完整快照；任何查询看到的所有语种对都属于同一个对齐
版本，不会出现一半查询用旧空间、另一半用新空间的情况。旧版本保持不可
变，候选对应可以一直回溯到它所依赖的对齐版本。
"""
from __future__ import annotations

import json
import sqlite3
import threading

from .data import canonical, fingerprint
from .alignment import METHOD_VERSION


class RegistryConflict(ValueError):
    pass


class AlignmentRegistry:
    def __init__(self, path=":memory:"):
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS alignments (
            alignment_id TEXT PRIMARY KEY,
            alignment_version INTEGER UNIQUE NOT NULL,
            digest TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS active_slot (
            slot INTEGER PRIMARY KEY CHECK (slot=1),
            alignment_version INTEGER,
            alignment_id TEXT,
            FOREIGN KEY(alignment_version) REFERENCES alignments(alignment_version)
        );
        """)
        with self.lock, self.connection:
            self.connection.execute("INSERT OR IGNORE INTO active_slot(slot) VALUES (1)")
            row = self.connection.execute("SELECT COALESCE(MAX(alignment_version),0) AS v FROM alignments").fetchone()
            self._next_version = row["v"] + 1

    def save(self, plan, activate=True):
        """保存对齐方案；activate=True 时在同一事务内切生效指针。"""
        if plan.get("method") != METHOD_VERSION:
            raise ValueError("对齐方法版本不匹配，拒绝保存未知方法的方案")
        digest = fingerprint(plan)
        encoded = canonical(plan)
        with self.lock, self.connection:
            existing = self.connection.execute(
                "SELECT alignment_id,alignment_version FROM alignments WHERE digest=?", (digest,)
            ).fetchone()
            if existing is not None:
                alignment_id = existing["alignment_id"]
                version = existing["alignment_version"]
                if activate:
                    self.connection.execute(
                        "UPDATE active_slot SET alignment_version=?, alignment_id=? WHERE slot=1",
                        (version, alignment_id),
                    )
                return self._receipt(alignment_id, version, digest, replayed=True, activated=activate)
            version = self._next_version
            self._next_version += 1
            alignment_id = "alignment-v" + str(version) + "-" + digest[:16]
            self.connection.execute(
                "INSERT INTO alignments(alignment_id,alignment_version,digest,payload) VALUES (?,?,?,?)",
                (alignment_id, version, digest, encoded),
            )
            if activate:
                self.connection.execute(
                    "UPDATE active_slot SET alignment_version=?, alignment_id=? WHERE slot=1",
                    (version, alignment_id),
                )
            return self._receipt(alignment_id, version, digest, replayed=False, activated=activate)

    def _receipt(self, alignment_id, version, digest, replayed, activated):
        return {
            "alignment_id": alignment_id,
            "alignment_version": version,
            "digest": digest,
            "replayed": replayed,
            "activated": activated,
        }

    def activate(self, version, expected_active_version=None):
        """比较交换式切换生效版本，避免并发更新互相覆盖。"""
        with self.lock, self.connection:
            current = self.connection.execute(
                "SELECT alignment_version FROM active_slot WHERE slot=1"
            ).fetchone()["alignment_version"]
            if expected_active_version is not None and current != expected_active_version:
                raise RegistryConflict("生效对齐版本已被其他更新改变")
            row = self.connection.execute(
                "SELECT alignment_id FROM alignments WHERE alignment_version=?", (version,)
            ).fetchone()
            if row is None:
                raise RegistryConflict("目标对齐版本不存在")
            self.connection.execute(
                "UPDATE active_slot SET alignment_version=?, alignment_id=? WHERE slot=1",
                (version, row["alignment_id"]),
            )
        return self.active()

    def active_version(self):
        with self.lock:
            row = self.connection.execute(
                "SELECT alignment_version,alignment_id FROM active_slot WHERE slot=1"
            ).fetchone()
            return row["alignment_version"], row["alignment_id"]

    def active(self):
        version, alignment_id = self.active_version()
        if version is None:
            return None
        return self.get(version)

    def get(self, version):
        with self.lock:
            row = self.connection.execute(
                "SELECT payload,alignment_id FROM alignments WHERE alignment_version=?", (version,)
            ).fetchone()
        if row is None:
            return None
        plan = json.loads(row["payload"])
        return {
            "alignment_id": row["alignment_id"],
            "alignment_version": version,
            "plan": plan,
        }

    def versions(self):
        with self.lock:
            rows = self.connection.execute(
                "SELECT alignment_version,alignment_id,digest FROM alignments ORDER BY alignment_version"
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self):
        with self.lock:
            self.connection.close()
