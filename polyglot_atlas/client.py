from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class RemoteError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(str(body))
        self.status = status
        self.body = body


class ProductClient:
    def __init__(self, base_url, timeout=5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, path, value=None):
        body = None if value is None else json.dumps(value, allow_nan=False).encode()
        request = Request(self.base_url + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise RemoteError(error.code, json.load(error)) from error

    def snapshot(self, as_of):
        return self.request("/snapshot?as_of=" + as_of)

    def policy(self):
        return self.request("/policy")

    def publish(self, key, artifact, revision):
        return self.request("/artifacts", {"request_key": key, "artifact": artifact, "policy_revision": revision})

    def artifact(self, identifier):
        return self.request("/artifacts/" + identifier)

    def build_alignment(self, as_of, activate=True):
        return self.request("/alignments", {"as_of": as_of, "activate": activate})

    def active_alignment(self):
        return self.request("/alignments/active")

    def alignment_versions(self):
        return self.request("/alignments/versions")

    def alignment(self, version):
        return self.request("/alignments/" + str(version))

    def activate_alignment(self, version, expected_active_version=None):
        return self.request(
            "/alignments/activate",
            {"alignment_version": version, "expected_active_version": expected_active_version},
        )

    def cross_lingual_search(self, document, target_language, as_of, limit=5):
        from urllib.parse import quote

        query = (
            "?document=" + quote(document)
            + "&target_language=" + quote(target_language)
            + "&as_of=" + quote(as_of)
            + "&limit=" + str(limit)
        )
        return self.request("/search/cross-lingual" + query)
