from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hamgf.adapters.base import BaseMemoryAdapter, UnifiedEvent
from hamgf.core.edges import EdgeRelation
from hamgf.core.nodes import CredibilitySource, NodeType


TIMESTAMP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "iso_or_slash",
        re.compile(r"(?P<timestamp>\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?)"),
    ),
    (
        "chinese",
        re.compile(r"(?P<timestamp>\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}(?::\d{2})?)"),
    ),
    (
        "month_day",
        re.compile(r"(?P<timestamp>\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)"),
    ),
)
SPEAKER_DELIMITERS = ("：", ":", "\t", "|")
MEDIA_WORDS = ("图片", "视频", "文件", "语音", "表情", "image", "video", "file", "audio", "media")
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:\+?\d[\d -]{8,}\d)(?![A-Za-z0-9_])")


class LowConfidenceFormatError(ValueError):
    def __init__(self, report: "ChatProbeReport") -> None:
        super().__init__(
            f"chat format confidence {report.confidence:.2f} is below the required threshold"
        )
        self.report = report


@dataclass(frozen=True, slots=True)
class ChatProbeReport:
    source_name: str
    line_count: int
    nonempty_line_count: int
    header_count: int
    timestamp_pattern: str | None
    message_mode: str | None
    speaker_delimiter: str | None
    confidence: float
    requires_review: bool
    media_placeholders: tuple[str, ...] = ()
    entity_types: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "line_count": self.line_count,
            "nonempty_line_count": self.nonempty_line_count,
            "header_count": self.header_count,
            "timestamp_pattern": self.timestamp_pattern,
            "message_mode": self.message_mode,
            "speaker_delimiter": self.speaker_delimiter,
            "confidence": self.confidence,
            "requires_review": self.requires_review,
            "media_placeholders": list(self.media_placeholders),
            "entity_types": list(self.entity_types),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class EnterpriseIngestionResult:
    node_ids: tuple[str, ...]
    report: ChatProbeReport


class EnterpriseChatParser:
    """Probe each file independently before parsing any messages."""

    def __init__(self, *, confidence_threshold: float = 0.68) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        self.confidence_threshold = confidence_threshold

    def probe(self, text: str, *, source_name: str = "<memory>") -> ChatProbeReport:
        lines = text.splitlines()
        nonempty = [line.strip() for line in lines if line.strip()]
        pattern_counts = {
            name: sum(1 for line in nonempty if pattern.search(line))
            for name, pattern in TIMESTAMP_PATTERNS
        }
        pattern_name = max(pattern_counts, key=pattern_counts.get, default=None)
        timestamp_count = pattern_counts.get(pattern_name, 0) if pattern_name else 0
        if not timestamp_count:
            return ChatProbeReport(
                source_name,
                len(lines),
                len(nonempty),
                0,
                None,
                None,
                None,
                0.0,
                True,
                warnings=("no repeatable timestamp pattern detected",),
            )
        pattern = dict(TIMESTAMP_PATTERNS)[pattern_name]
        inline_delimiters: Counter[str] = Counter()
        inline_count = 0
        separate_count = 0
        for line in nonempty:
            match = pattern.search(line)
            if match is None:
                continue
            remainder = self._without_timestamp(line, match)
            split = self._split_speaker(remainder)
            if split is not None and split[1].strip():
                inline_count += 1
                inline_delimiters[split[2]] += 1
            elif 0 < len(remainder) <= 64:
                separate_count += 1
        mode = "inline" if inline_count >= separate_count else "header_next_line"
        header_count = max(inline_count, separate_count)
        delimiter = inline_delimiters.most_common(1)[0][0] if mode == "inline" and inline_delimiters else None
        timestamp_consistency = timestamp_count / max(1, len(nonempty))
        header_precision = header_count / max(1, timestamp_count)
        sample_strength = min(1.0, header_count / 3.0)
        confidence = round(
            min(1.0, 0.35 * timestamp_consistency + 0.40 * header_precision + 0.25 * sample_strength),
            4,
        )
        warnings: list[str] = []
        if header_count < 2:
            warnings.append("fewer than two message headers detected")
        if header_precision < 0.7:
            warnings.append("timestamp lines do not form a consistent message header")
        media = tuple(sorted(set(self._find_media(line) for line in nonempty) - {None}))
        entity_types = tuple(sorted(self._entity_types("\n".join(nonempty))))
        return ChatProbeReport(
            source_name=source_name,
            line_count=len(lines),
            nonempty_line_count=len(nonempty),
            header_count=header_count,
            timestamp_pattern=pattern_name,
            message_mode=mode,
            speaker_delimiter=delimiter,
            confidence=confidence,
            requires_review=confidence < self.confidence_threshold,
            media_placeholders=media,
            entity_types=entity_types,
            warnings=tuple(warnings),
        )

    def parse(
        self,
        text: str,
        *,
        source_name: str = "<memory>",
        allow_low_confidence: bool = False,
    ) -> tuple[tuple[UnifiedEvent, ...], ChatProbeReport]:
        report = self.probe(text, source_name=source_name)
        if report.requires_review and not allow_low_confidence:
            raise LowConfidenceFormatError(report)
        if report.timestamp_pattern is None or report.message_mode is None:
            return (), report
        pattern = dict(TIMESTAMP_PATTERNS)[report.timestamp_pattern]
        messages: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.search(line)
            if match is not None:
                remainder = self._without_timestamp(line, match)
                speaker = remainder
                content = ""
                if report.message_mode == "inline":
                    split = self._split_speaker(remainder, report.speaker_delimiter)
                    if split is None:
                        if current is not None:
                            current["content"].append(line)
                        continue
                    speaker, content, _delimiter = split
                if current is not None:
                    messages.append(current)
                current = {
                    "speaker": speaker.strip(),
                    "timestamp": self._normalize_timestamp(match.group("timestamp")),
                    "content": [content.strip()] if content.strip() else [],
                }
            elif current is not None:
                current["content"].append(line)
        if current is not None:
            messages.append(current)

        events = []
        for index, message in enumerate(messages):
            raw_content = "\n".join(message["content"]).strip()
            if not raw_content:
                raw_content = "[empty message]"
            content = self.redact_sensitive(raw_content)
            entities = self.extract_entities(content)
            media_type = self._find_media(content)
            node_type, importance, timeliness = self._classify(content, media_type)
            relation = (
                EdgeRelation.CAUSAL
                if any(token in content.casefold() for token in ("因为", "导致", "因此", "because", "therefore"))
                else EdgeRelation.TEMPORAL
            )
            events.append(
                UnifiedEvent(
                    content=content,
                    type=node_type,
                    importance=importance,
                    timeliness=timeliness,
                    source=CredibilitySource.USER_CONFIRMED,
                    occurred_at=message["timestamp"],
                    relation=relation,
                    relation_label="chat sequence" if relation == EdgeRelation.TEMPORAL else "causal statement",
                    metadata={
                        "adapter": "enterprise_chat",
                        "source_name": source_name,
                        "message_index": index,
                        "actor_id": message["speaker"],
                        "entities": entities,
                        "media_type": media_type,
                        "security_level": "deidentified_enterprise",
                        "format_confidence": report.confidence,
                    },
                )
            )
        return tuple(events), report

    def parse_file(
        self,
        path: str | Path,
        *,
        allow_low_confidence: bool = False,
    ) -> tuple[tuple[UnifiedEvent, ...], ChatProbeReport]:
        source = Path(path)
        return self.parse(
            source.read_text(encoding="utf-8-sig"),
            source_name=source.name,
            allow_low_confidence=allow_low_confidence,
        )

    @staticmethod
    def redact_sensitive(text: str) -> str:
        redacted = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
        return PHONE_PATTERN.sub("[PHONE_REDACTED]", redacted)

    @classmethod
    def extract_entities(cls, text: str) -> dict[str, list[str]]:
        entities: dict[str, set[str]] = defaultdict(set)
        for mention in re.findall(r"@([A-Za-z0-9_\-\u3400-\u9fff]{2,40})", text):
            entities["mention"].add(mention)
        for key, value in re.findall(
            r"([A-Za-z\u3400-\u9fff][A-Za-z0-9_\-\u3400-\u9fff]{1,24})\s*(?:ID|id|编号)?\s*[:=：]\s*([A-Za-z0-9_\-]{3,64})",
            text,
        ):
            entities[key.casefold()].add(value)
        for token in re.findall(r"\b[A-Za-z]{2,12}[_-][A-Za-z0-9_-]{2,48}\b", text):
            entities["anonymous_id"].add(token)
        return {key: sorted(values) for key, values in sorted(entities.items())}

    @classmethod
    def _entity_types(cls, text: str) -> set[str]:
        return set(cls.extract_entities(text))

    @staticmethod
    def _without_timestamp(line: str, match: re.Match[str]) -> str:
        remainder = f"{line[:match.start()]} {line[match.end():]}"
        return remainder.strip(" \t[]【】()（）<>{}-")

    @staticmethod
    def _split_speaker(
        remainder: str,
        delimiter: str | None = None,
    ) -> tuple[str, str, str] | None:
        delimiters = (delimiter,) if delimiter is not None else SPEAKER_DELIMITERS
        for candidate in delimiters:
            if candidate and candidate in remainder:
                speaker, content = remainder.split(candidate, 1)
                if 0 < len(speaker.strip()) <= 64:
                    return speaker.strip(), content.strip(), candidate
        return None

    @staticmethod
    def _normalize_timestamp(value: str) -> str:
        normalized = value.strip().replace("/", "-")
        chinese = re.fullmatch(
            r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})(?::(\d{2}))?",
            normalized,
        )
        if chinese:
            year, month, day, hour, minute, second = chinese.groups()
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second or 0), tzinfo=timezone.utc)
            return dt.isoformat()
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return value.strip()
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.isoformat()

    @staticmethod
    def _find_media(text: str) -> str | None:
        compact = text.strip().strip("[]【】<>()（）").casefold()
        if len(compact) > 40:
            return None
        return next((word for word in MEDIA_WORDS if word in compact), None)

    @staticmethod
    def _classify(content: str, media_type: str | None) -> tuple[NodeType, float, float]:
        lowered = content.casefold()
        if any(token in lowered for token in ("决定", "决策", "采用", "批准", "方案", "decision", "approve")):
            return NodeType.DECISION, 0.9, 0.85
        if any(token in lowered for token in ("反馈", "确认", "满意", "拒绝", "feedback", "confirmed")):
            return NodeType.FEEDBACK, 0.82, 0.85
        if any(token in lowered for token in ("状态", "进度", "完成", "阻塞", "status", "progress")):
            return NodeType.STATE, 0.7, 0.85
        if media_type is not None:
            return NodeType.EVENT, 0.35, 0.8
        return NodeType.EVENT, 0.68, 0.8


class EnterpriseMemoryAdapter(BaseMemoryAdapter):
    def __init__(self, writer: Any, *, parser: EnterpriseChatParser | None = None) -> None:
        super().__init__(writer)
        self.parser = parser or EnterpriseChatParser()

    def insert_text(
        self,
        text: str,
        *,
        source_name: str = "<memory>",
        allow_low_confidence: bool = False,
    ) -> EnterpriseIngestionResult:
        events, report = self.parser.parse(
            text,
            source_name=source_name,
            allow_low_confidence=allow_low_confidence,
        )
        return EnterpriseIngestionResult(self.insert_events(events), report)

    def insert_file(
        self,
        path: str | Path,
        *,
        allow_low_confidence: bool = False,
    ) -> EnterpriseIngestionResult:
        events, report = self.parser.parse_file(path, allow_low_confidence=allow_low_confidence)
        return EnterpriseIngestionResult(self.insert_events(events), report)

    def track_customer(self, entity_id: str) -> tuple[dict[str, Any], ...]:
        matches = []
        for node in self.writer.graph.iter_nodes():
            metadata = dict(node.metadata)
            entities = metadata.get("entities") or {}
            values = [value for group in entities.values() for value in group]
            if entity_id == metadata.get("actor_id") or entity_id in values:
                matches.append(node.to_dict())
        return tuple(sorted(matches, key=lambda node: (node["created_at"], node["node_id"])))

    def decision_audit(self, entity_id: str | None = None) -> tuple[dict[str, Any], ...]:
        relevant = {item["node_id"] for item in self.track_customer(entity_id)} if entity_id else None
        audits = []
        for node in self.writer.graph.iter_nodes():
            if node.type != NodeType.DECISION or (relevant is not None and node.node_id not in relevant):
                continue
            links = []
            for key, edge in self.writer.graph.iter_edges():
                if node.node_id in {edge.source, edge.target}:
                    links.append({**edge.to_dict(), "key": key})
            audits.append({"node": node.to_dict(), "links": links})
        return tuple(audits)
