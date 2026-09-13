from __future__ import annotations

from typing import Any, Iterable, Mapping

from hamgf.adapters.base import BaseMemoryAdapter, UnifiedEvent
from hamgf.core.nodes import CredibilitySource, NodeType
from hamgf.retrieval.hci import build_context


class LocalModelMemoryAdapter(BaseMemoryAdapter):
    def insert_turns(
        self,
        turns: Iterable[Mapping[str, Any]],
        *,
        session_id: str,
    ) -> tuple[str, ...]:
        events = []
        for index, turn in enumerate(turns):
            content = turn.get("content") or turn.get("text")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"turn {index} requires non-empty content")
            role = str(turn.get("role", "user"))
            events.append(
                UnifiedEvent(
                    content=content,
                    type=NodeType.FEEDBACK if role == "user" else NodeType.EVENT,
                    importance=float(turn.get("importance", 0.72 if role == "user" else 0.64)),
                    timeliness=float(turn.get("timeliness", 0.8)),
                    source=(
                        CredibilitySource.USER_CONFIRMED
                        if role == "user"
                        else CredibilitySource.AGENT_INFERRED
                    ),
                    occurred_at=turn.get("timestamp"),
                    metadata={
                        "adapter": "local_model",
                        "session_id": session_id,
                        "turn_index": index,
                        "role": role,
                        "security_level": "personal_private",
                        **dict(turn.get("metadata") or {}),
                    },
                )
            )
        return self.insert_events(events)

    def recall_context(self, query: str, *, k: int = 6) -> dict[str, Any]:
        chain = self.writer.search.search(query, k=k)
        context = build_context(chain)
        return {
            "chain_reference": context.chain_reference,
            "summaries": list(context.summaries),
            "details": list(context.details),
            "context_text": (
                f"基于记忆链 [{context.chain_reference}]\n"
                + "\n".join(f"- {detail}" for detail in context.details)
                if context.chain_reference
                else "未检索到可用记忆链。"
            ),
        }


class CodingProjectMemoryAdapter(BaseMemoryAdapter):
    def insert_commits(
        self,
        commits: Iterable[Mapping[str, Any]],
        *,
        project_id: str,
    ) -> tuple[str, ...]:
        events = []
        for commit in commits:
            sha = str(commit.get("sha") or commit.get("id") or "unknown")
            message = str(commit.get("message") or commit.get("title") or "").strip()
            if not message:
                raise ValueError(f"commit {sha} requires a message")
            events.append(
                UnifiedEvent(
                    content=f"Commit {sha[:12]}: {message}",
                    type=NodeType.EVENT,
                    importance=float(commit.get("importance", 0.7)),
                    timeliness=float(commit.get("timeliness", 0.72)),
                    source=CredibilitySource.EXTERNAL_FETCHED,
                    occurred_at=commit.get("timestamp") or commit.get("date"),
                    metadata={
                        "adapter": "coding_project",
                        "record_type": "commit",
                        "project_id": project_id,
                        "sha": sha,
                        "author_id": commit.get("author_id") or commit.get("author"),
                        "url": commit.get("url"),
                        "security_level": "public",
                    },
                )
            )
        return self.insert_events(events)

    def insert_issues(
        self,
        issues: Iterable[Mapping[str, Any]],
        *,
        project_id: str,
    ) -> tuple[str, ...]:
        events = []
        for issue in issues:
            number = issue.get("number") or issue.get("id")
            title = str(issue.get("title") or "").strip()
            if number is None or not title:
                raise ValueError("issue requires number/id and title")
            state = str(issue.get("state", "open"))
            events.append(
                UnifiedEvent(
                    content=f"Issue #{number} [{state}]: {title}",
                    type=NodeType.STATE,
                    importance=float(issue.get("importance", 0.72)),
                    timeliness=float(issue.get("timeliness", 0.78 if state == "open" else 0.65)),
                    source=CredibilitySource.EXTERNAL_FETCHED,
                    occurred_at=issue.get("updated_at") or issue.get("created_at"),
                    metadata={
                        "adapter": "coding_project",
                        "record_type": "issue",
                        "project_id": project_id,
                        "issue_number": number,
                        "state": state,
                        "labels": list(issue.get("labels") or []),
                        "url": issue.get("url"),
                        "security_level": "public",
                    },
                )
            )
        return self.insert_events(events)

    def project_timeline(self, project_id: str) -> tuple[dict[str, Any], ...]:
        nodes = [
            node.to_dict()
            for node in self.writer.graph.iter_nodes()
            if node.metadata.get("project_id") == project_id
        ]
        return tuple(
            sorted(
                nodes,
                key=lambda node: (
                    node["metadata"].get("occurred_at") or node["created_at"],
                    node["node_id"],
                ),
            )
        )

    def recall_context(self, query: str, *, k: int = 8) -> dict[str, Any]:
        chain = self.writer.search.search(query, k=k)
        return {
            "node_ids": list(chain.node_ids),
            "edge_trace": list(chain.edge_trace),
            "narrative": list(chain.narrative),
        }
