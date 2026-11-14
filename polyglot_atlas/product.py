import math


def cosine(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("单一索引的向量维数不一致")
    denominator = math.sqrt(sum(value*value for value in left) * sum(value*value for value in right))
    return None if denominator == 0 else sum(a*b for a,b in zip(left,right))/denominator


def same_language_search(snapshot, document_id, limit=5):
    documents = {row["id"]: row for row in snapshot["tables"]["documents"]}
    vectors = {row["id"]: row for row in snapshot["tables"]["vectors"]}
    if document_id not in documents or document_id not in vectors:
        raise ValueError("检索资料不存在")
    query = documents[document_id]
    permission = snapshot["tables"]["permissions"][0]
    allowed_languages = set(permission["allowed_languages"])
    collections = set(permission["allowed_collections"])
    if query["language"] not in allowed_languages or query["collection"] not in collections:
        raise ValueError("没有访问该资料的权限")
    results = []
    for identifier, document in documents.items():
        if identifier == document_id or document["language"] != query["language"] or document["collection"] not in collections:
            continue
        vector = vectors[identifier]
        if vector["index_version"] != vectors[document_id]["index_version"]:
            continue
        score = cosine(vectors[document_id]["values"], vector["values"])
        if score is not None:
            results.append({"document": identifier, "score": score, "index_version": vector["index_version"]})
    return sorted(results, key=lambda row: (-row["score"], row["document"]))[:limit]


def summarize(snapshot):
    languages = sorted({row["language"] for row in snapshot["tables"]["documents"]})
    return {"languages": languages, "within_language_neighbors": same_language_search(snapshot, "zh-0"), "interpretation": "不同语种索引没有共同坐标，当前只返回单语相似资料"}
