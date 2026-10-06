"""Conservative local repair of explicitly stated event ordering.

No embeddings, network requests, new nodes or inferred causality are involved.
The matching subject is each event's own action, never its contextual premise.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

from hamgf.core.edges import MemoryEdge
from backend.extraction import DATE_RE, MAX_RELATIONS, _statement_sentences, normalize_time


ACTION_RE = re.compile(r"完成|提交|制作|研究|学习|阅读|召开|参加|出发|到达|启动|结束|发布|发送|收到|购买|支付|准备|开始|吃|喝|写|做|买|读|考|练|跑|睡")
ENGLISH_ACTION_RE = re.compile(r"\b(finish(?:ing|ed)?|complet(?:e|ing|ed)|submit(?:ting|ted)?|eat(?:ing)?|ate|writ(?:e|ing|ten)|wrote|do(?:ing|ne)?|did|read(?:ing)?|study(?:ing)?|studied|start(?:ing|ed)?|arriv(?:e|ing|ed)|leave|leaving|left)\b", re.I)
NEGATED_RE = re.compile(r"不|没|未|并非|\b(?:not|never|didn't|won't|isn't)\b", re.I)
BACKGROUND_RE = re.compile(r"喜欢|偏好|希望|想象|提到|讨论|关于|听说|据说|他说|她说|\b(?:prefer|like|hope|imagine|mention|discuss)\b", re.I)


def _trim(value: str) -> str:
    return value.strip(" ，,；;：:。.!！\t\r\n")


def _signature(value: str) -> tuple[str, str, str, str] | None:
    value = DATE_RE.sub("", value)
    value = re.sub(r"^(?:请记住|记住|更正|纠正|接着)[：:]?\s*", "", _trim(value))
    if NEGATED_RE.search(value) or BACKGROUND_RE.search(value):
        return None
    english = ENGLISH_ACTION_RE.search(value)
    if english:
        token = english.group().lower()
        verb = next((base for base, forms in {
            "finish": {"finish", "finishing", "finished", "complete", "completing", "completed"},
            "submit": {"submit", "submitting", "submitted"}, "eat": {"eat", "eating", "ate"},
            "write": {"write", "writing", "written", "wrote"}, "do": {"do", "doing", "done", "did"},
            "read": {"read", "reading"}, "study": {"study", "studying", "studied"},
            "start": {"start", "starting", "started"}, "arrive": {"arrive", "arriving", "arrived"},
            "leave": {"leave", "leaving", "left"}}.items() if token in forms), token)
        actor = re.sub(r"\b(?:will|would|have|has|had|am|is|are|was|were|plan to|want to|going to|the)\b", "", value[:english.start()].lower()).strip()
        obj = re.sub(r"\b(?:a|an|the|my|some)\b", "", value[english.end():].lower())
        obj = re.sub(r"[^\w]+", " ", obj).strip()
        return (actor, verb, obj, "") if len(obj) >= 2 else None
    action = ACTION_RE.search(value)
    if action is None:
        return None
    prefix, obj = value[:action.start()], value[action.end():]
    location_match = re.search(r"在([^，,]{1,16})$", prefix)
    location = location_match.group(1) if location_match else ""
    if location_match:
        prefix = prefix[:location_match.start()]
    actor = re.sub(r"(?:今天|明天|昨天|后天|前天|今晚|现在|接下来|已经|正在|打算|计划|准备|将要|将|会|要|在|先|再|刚刚|随后|之后|之前|请)", "", prefix)
    actor = re.sub(r"[\s，,:：]", "", actor)
    obj = re.sub(r"^(?:完|过|了|一下|一顿|一餐|一次|一遍|一些)+", "", obj)
    obj = re.sub(r"(?:了|过)$", "", _trim(obj))
    obj = re.sub(r"[\s的，,:：]", "", obj)
    if not obj or len(actor) > 20:
        return None
    return actor, action.group(), obj, location


def temporal_clauses(text: str) -> list[dict[str, str]]:
    """Return ordered phrases and the grammatical main action of each clause."""
    result = []
    for sentence in _statement_sentences(text):
        for fragment in re.split(r"[；;]", sentence):
            quote = _trim(fragment)
            if not quote or len(quote) > 1200 or NEGATED_RE.search(quote):
                continue
            left = right = label = focus = None
            # After A, B / Before A, B.
            match = re.match(r"^(after|before)\s+(.+?)[,，]\s*(.+)$", quote, re.I)
            if match:
                label, left, right = match.groups()
                before, after = (left, right) if label.lower() == "after" else (right, left)
                focus = right
            else:
                # A before B / A after B. A remains the grammatical focus.
                match = re.match(r"^(.+?)\s+(before|after)\s+(.+)$", quote, re.I)
                if match:
                    left, label, right = match.groups()
                    before, after = (left, right) if label.lower() == "before" else (right, left)
                    focus = left
                else:
                    # A 在 B 之前/之后.
                    match = re.match(r"^(.+?)在(.+?)(之前|之后|以前|以后|前|后)$", quote)
                    if match:
                        left, right, label = match.groups()
                        before, after = (left, right) if label in {"之前", "以前", "前"} else (right, left)
                        focus = left
                    else:
                        match = re.match(r"^先(.+?)[，,]?\s*(再|然后|随后|接着)(.+)$", quote)
                        if match:
                            left, label, right = match.groups()
                            before, after, focus = left, right, right
                        else:
                            match = re.match(r"^(.+?)(之后|之前|以后|以前|然后|随后|接着|后|前)(.+)$", quote)
                            if match:
                                left, label, right = match.groups()
                                # A comma before the temporal premise belongs
                                # to a previous clause, not the event's actor.
                                left = re.split(r"[，,]", left)[-1]
                                before, after = (right, left) if label in {"之前", "以前", "前"} else (left, right)
                                focus = right
            if not left or not right:
                continue
            before, after, focus = map(_trim, (before, after, focus))
            if _signature(before) is None or _signature(after) is None:
                continue
            result.append({"before": before, "after": after, "focus": focus, "label": label,
                           "evidence": quote})
    return result[:MAX_RELATIONS]


def event_body(content: str) -> str:
    pairs = temporal_clauses(content)
    return pairs[-1]["focus"] if len(pairs) == 1 else content


def dates_compatible(reference: Mapping[str, Any], phrase: str) -> bool:
    body = event_body(str(reference.get("content", "")))
    stated_dates = DATE_RE.findall(phrase)
    recorded_dates = DATE_RE.findall(body)
    recorded_time = reference.get("temporal") or reference.get("metadata", {}).get("temporal") or {}
    if not recorded_dates and recorded_time.get("normalized"):
        recorded_dates = [recorded_time["raw"]]
    if stated_dates:
        if not recorded_dates:
            return False
        try:
            explicit = [normalize_time(value)["normalized"] for value in stated_dates]
            recorded = [normalize_time(value)["normalized"] for value in recorded_dates]
        except ValueError:
            return False
        if not all(any(a == b or ((a.startswith("--") or b.startswith("--")) and a[-5:] == b[-5:])
                       for b in recorded if b) for a in explicit if a):
            return False
    return True


def event_matches(reference: Mapping[str, Any], phrase: str) -> bool:
    category = reference.get("category") or reference.get("metadata", {}).get("category")
    if category != "event" or not dates_compatible(reference, phrase):
        return False
    body = event_body(str(reference.get("content", "")))
    a, b = _signature(body), _signature(phrase)
    if a is None or b is None:
        return False
    actor_a, verb_a, object_a, location_a = a
    actor_b, verb_b, object_b, location_b = b
    return (verb_a == verb_b and object_a == object_b
            and (not actor_a or not actor_b or actor_a == actor_b)
            and (not location_a or not location_b or location_a == location_b))


def align_temporal_references(source: Mapping[str, Any], target: Mapping[str, Any], quote: str) -> dict[str, Any] | None:
    """Resolve model titles against explicit temporal clauses, without new facts."""
    matched = []
    for pair in temporal_clauses(quote):
        if event_matches(source, pair["before"]) and event_matches(target, pair["after"]):
            matched.append({**pair, "reverse": False})
        if event_matches(target, pair["before"]) and event_matches(source, pair["after"]):
            matched.append({**pair, "reverse": True})
    return matched[0] if len(matched) == 1 else None


def would_create_temporal_cycle(application: Any, source: str, target: str) -> bool:
    successors: dict[str, set[str]] = {}
    for _, edge in application.graph.iter_edges(statuses=["active"]):
        if edge.relation.value == "temporal":
            successors.setdefault(edge.source, set()).add(edge.target)
    pending, visited = [target], set()
    while pending:
        current = pending.pop()
        if current == source:
            return True
        if current not in visited:
            visited.add(current)
            pending.extend(successors.get(current, set()) - visited)
    return False


def repair_explicit_relations(application: Any, messages: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Idempotently add missing grounded time edges between existing events."""
    nodes = [n.to_dict() for n in application.graph.iter_nodes()
             if n.status.value not in {"archived", "superseded"}
             and n.metadata.get("kind") == "logical_memory" and n.metadata.get("category") == "event"]
    chronology = {m["id"]: index for index, m in enumerate(messages) if isinstance(m.get("id"), str)}
    result = {"created_count": 0, "temporal_edges": 0, "node_ids": [], "warnings": []}
    for message in messages:
        if message.get("role", "user") != "user" or not isinstance(message.get("content"), str):
            continue
        for pair in temporal_clauses(message["content"]):
            eligible = [node for node in nodes if not node["metadata"].get("source_message_ids") or
                        any(mid not in chronology or chronology[mid] <= chronology.get(message["id"], 0)
                            for mid in node["metadata"].get("source_message_ids", []))]
            before = [node for node in eligible if event_matches(node, pair["before"])]
            after = [node for node in eligible if event_matches(node, pair["after"])]
            if len(before) != 1 or len(after) != 1 or before[0]["node_id"] == after[0]["node_id"]:
                continue
            source, target = before[0]["node_id"], after[0]["node_id"]
            existing = [edge for _, edge in application.graph.iter_edges(statuses=["active"])
                        if edge.relation.value == "temporal"]
            if any(edge.source == source and edge.target == target for edge in existing):
                continue
            if would_create_temporal_cycle(application, source, target):
                result["warnings"].append("明确先后顺序与已有时序边冲突，已保留原图以供核对。")
                continue
            label = "明确先后：" + pair["label"]
            edge = MemoryEdge.create(source, target, relation="temporal", label=label, weight=0.95)
            application.graph.add_edge(edge)
            current = application.graph.get_node(source)
            metadata = deepcopy(dict(current.metadata))
            audit = {"target_node_id": target, "relation": "temporal", "label": label,
                     "evidence": pair["evidence"], "evidence_sources": [{"message_id": message["id"], "quote": pair["evidence"]}],
                     "method": "explicit_temporal_parser", "model": None,
                     "source_mention": pair["before"], "target_mention": pair["after"]}
            metadata.setdefault("relations", []).append(audit)
            application.graph.update_node(source, metadata=metadata)
            application._record_change("logical_relation_repaired", edge.to_dict())
            result["created_count"] += 1
            result["temporal_edges"] += 1
            result["node_ids"].extend([source, target])
    result["node_ids"] = list(dict.fromkeys(result["node_ids"]))
    result["warnings"] = list(dict.fromkeys(result["warnings"]))
    return result
