from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import re

from sage_core.cpu_language_core import CPULanguageCore


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_ts() -> str:
    return utc_now().replace(":", "-").replace(".", "-")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class LocalChatLoopConfig:
    memory_root: str = "memory"
    memory_summary_path: str = "memory/summaries/latest_memory_summary.md"
    chat_log_dir: str = "logs/chat"
    output_path: str = "results/v2_9_local_chat_context_result.json"

    write_memory_candidates: bool = True
    max_summary_chars: int = 4000
    max_turns_saved: int = 200

    allow_command_consolidate: bool = True
    allow_command_summarize: bool = True

    # v2.9
    use_memory_context_manager: bool = True
    memory_context_config_path: str = "configs/memory_context_manager.json"
    max_context_preview_chars: int = 2500

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalChatTurn:
    turn_id: str
    created_at: str
    user_input: str
    response: str
    state: Dict[str, Any]
    analysis_summary: Dict[str, Any]
    memory_hits: List[Dict[str, Any]]
    memory_summary_used: bool
    memory_context_used: bool = False
    memory_context_stats: Dict[str, Any] = field(default_factory=dict)
    memory_candidate_path: Optional[str] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalChatLoopReport:
    version: str = "v2.9"
    created_at: str = field(default_factory=utc_now)
    loop_name: str = "local_chat_loop"
    mode: str = "local_cli_with_memory_context"
    config: Dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    latest_turn: Optional[Dict[str, Any]] = None
    chat_log_path: str = ""
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "local_only": True,
        "network_actions": False,
        "arbitrary_shell_actions": False,
        "file_delete": False,
        "git_actions": False,
        "memory_candidate_write": True,
        "memory_context_bundle_write": True,
        "memory_auto_consolidation_only_by_explicit_command": True,
        "source_memory_delete": False,
    })
    passed: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


class LocalChatLoop:
    """SAGE local chat loop.

    v2.9 integrates Memory Context Manager:
    user input -> CPU Language Core -> query-specific context bundle -> response.
    """

    MEMORY_IMPORTANCE_TERMS = (
        "성공", "통과", "검증", "정리", "원칙", "정책", "기억",
        "passed", "benchmark", "policy", "safety", "result", "memory",
        "SAGE", "AGI", "runtime", "reflection", "autonomy",
    )

    def __init__(self, config: LocalChatLoopConfig) -> None:
        self.config = config
        self.memory_root = Path(config.memory_root)
        self.core = CPULanguageCore(memory_root=self.memory_root)
        self.turns: List[LocalChatTurn] = []
        self.chat_log_path = self._make_chat_log_path()
        self.chat_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _make_chat_log_path(self) -> Path:
        return Path(self.config.chat_log_dir) / f"sage_chat_{safe_ts()}.jsonl"

    def chat_once(self, user_input: str) -> LocalChatTurn:
        user_input = normalize_text(user_input)
        core_result = self.core.run(user_input, retrieve_memory=True).to_jsonable()
        summary = self._read_memory_summary()
        context_bundle = self._build_memory_context(user_input)

        response = self._compose_chat_response(user_input, core_result, summary, context_bundle)
        turn_id = short_hash(utc_now() + "\n" + user_input + "\n" + response)

        candidate_path = None
        if self.config.write_memory_candidates and self._should_write_memory_candidate(user_input, core_result):
            candidate_path = self._write_memory_candidate(turn_id, user_input, response, core_result, context_bundle)

        analysis = core_result.get("analysis", {})
        analysis_summary = {
            "char_count": analysis.get("char_count"),
            "word_count": len(analysis.get("word_tokens", [])),
            "chunk_count": len(analysis.get("chunk_tokens", [])),
            "chunk_tokens": analysis.get("chunk_tokens", []),
            "compression": analysis.get("compression", {}),
        }

        turn = LocalChatTurn(
            turn_id=turn_id,
            created_at=utc_now(),
            user_input=user_input,
            response=response,
            state=core_result.get("state", {}),
            analysis_summary=analysis_summary,
            memory_hits=core_result.get("memory_hits", []),
            memory_summary_used=bool(summary),
            memory_context_used=bool(context_bundle and context_bundle.get("passed")),
            memory_context_stats=context_bundle.get("stats", {}) if context_bundle else {},
            memory_candidate_path=candidate_path,
        )

        self.turns.append(turn)
        self._append_chat_log(turn)
        self._write_report(turn)
        return turn

    def run_command(self, command: str) -> Dict[str, Any]:
        command = command.strip()
        lowered = command.lower()

        if lowered == "/status":
            return {
                "command": command,
                "stage_counts": self._stage_counts(),
                "summary_exists": Path(self.config.memory_summary_path).exists(),
                "context_manager_enabled": self.config.use_memory_context_manager,
                "context_bundle_exists": Path("results/v2_8_memory_context_bundle.json").exists(),
                "turn_count": len(self.turns),
                "chat_log_path": str(self.chat_log_path).replace("\\", "/"),
            }

        if lowered == "/summary":
            summary = self._read_memory_summary()
            return {
                "command": command,
                "summary_exists": bool(summary),
                "summary_preview": summary[:2500],
            }

        if lowered.startswith("/context"):
            query = command[len("/context"):].strip() or "SAGE 현재 상태와 다음 단계"
            bundle = self._build_memory_context(query)
            return {
                "command": "/context",
                "query": query,
                "enabled": self.config.use_memory_context_manager,
                "passed": bundle.get("passed") if bundle else False,
                "inferred_topics": bundle.get("inferred_topics", []) if bundle else [],
                "stats": bundle.get("stats", {}) if bundle else {},
                "selected_memory_items": [
                    {
                        "stage": item.get("stage"),
                        "path": item.get("path"),
                        "score": item.get("score"),
                    }
                    for item in (bundle.get("selected_memory_items", []) if bundle else [])[:6]
                ],
            }

        if lowered == "/help":
            return {
                "command": command,
                "known_commands": ["/status", "/summary", "/context <query>", "/consolidate", "/summarize", "/help", "/exit"],
                "description": "SAGE local chat loop commands.",
            }

        if lowered == "/consolidate":
            if not self.config.allow_command_consolidate:
                return {"command": command, "allowed": False, "reason": "command_disabled"}
            return self._run_consolidation_command()

        if lowered == "/summarize":
            if not self.config.allow_command_summarize:
                return {"command": command, "allowed": False, "reason": "command_disabled"}
            return self._run_summarizer_command()

        return {
            "command": command,
            "allowed": False,
            "reason": "unknown_command",
            "known_commands": ["/status", "/summary", "/context <query>", "/consolidate", "/summarize", "/help", "/exit"],
        }

    def _build_memory_context(self, query: str) -> Dict[str, Any]:
        if not self.config.use_memory_context_manager:
            return {}

        try:
            from sage_core.memory_context_manager import MemoryContextConfig, MemoryContextManager
        except Exception as exc:
            return {
                "passed": False,
                "error": f"memory_context_manager_unavailable: {type(exc).__name__}: {exc}",
                "stats": {},
            }

        config_path = Path(self.config.memory_context_config_path)
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                allowed = set(MemoryContextConfig.__dataclass_fields__.keys())
                filtered = {k: v for k, v in data.items() if k in allowed}
                base = {k: getattr(MemoryContextConfig(), k) for k in allowed}
                context_config = MemoryContextConfig(**{**base, **filtered})
            except Exception:
                context_config = MemoryContextConfig(
                    memory_root=self.config.memory_root,
                    memory_summary_path=self.config.memory_summary_path,
                )
        else:
            context_config = MemoryContextConfig(
                memory_root=self.config.memory_root,
                memory_summary_path=self.config.memory_summary_path,
            )

        context_config.memory_root = self.config.memory_root
        context_config.memory_summary_path = self.config.memory_summary_path

        manager = MemoryContextManager(context_config)
        return manager.build_context(query).to_jsonable()

    def _compose_chat_response(
        self,
        user_input: str,
        core_result: Dict[str, Any],
        summary: str,
        context_bundle: Dict[str, Any],
    ) -> str:
        state = core_result.get("state", {})
        topics = state.get("topics", [])
        intent = state.get("intent", "unknown")
        memory_hits = core_result.get("memory_hits", [])

        persona_response = self._persona_response(user_input, core_result)
        if persona_response:
            core_response = persona_response
        else:
            core_response = core_result.get("response", "")

        lines: List[str] = []
        lines.append("SAGE local loop response")
        lines.append("")
        lines.append(core_response)

        if context_bundle and context_bundle.get("passed"):
            lines.append("")
            lines.append("Memory context 참고:")
            lines.extend(self._format_context_bundle(context_bundle))
        elif summary:
            lines.append("")
            lines.append("Memory summary 참고:")
            lines.append(self._summary_focus_line(summary, topics))

        if memory_hits:
            lines.append("")
            lines.append("관련 approved memory:")
            for hit in memory_hits[:3]:
                lines.append(f"- {hit.get('path')} · score {hit.get('score')}")

        lines.append("")
        convo_intent = self._conversation_intent(user_input)
        if convo_intent in {"greeting", "identity", "capability"}:
            lines.append("도움말: `/status`, `/context SAGE 현재 진행상황`, `/summary`, `/consolidate`, `/summarize`, `/exit` 명령을 사용할 수 있습니다.")
        elif intent in {"request_next_step", "summary_request"}:
            lines.append("다음 행동 제안: `/context <질문>`으로 관련 기억 bundle을 확인하고, 필요하면 `/consolidate` 후 `/summarize`를 실행할 수 있습니다.")
        else:
            lines.append("다음 행동 제안: `/status` 또는 `/context <질문>`으로 현재 memory/context 상태를 확인할 수 있습니다.")

        return "\n".join(lines)

    def _format_context_bundle(self, bundle: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        topics = bundle.get("inferred_topics", [])
        stats = bundle.get("stats", {})
        lines.append(f"- inferred_topics: {', '.join(topics) if topics else 'none'}")
        lines.append(
            f"- selected: summary {stats.get('summary_snippet_count', 0)}개, "
            f"memory {stats.get('memory_item_count', 0)}개"
        )

        snippets = bundle.get("selected_summary_snippets", [])
        if snippets:
            lines.append("- summary snippet:")
            first = snippets[0]
            text = first.get("text", "")
            lines.append(f"  · {first.get('heading')} · score {first.get('score')}: {text[:350]}")

        items = bundle.get("selected_memory_items", [])
        if items:
            lines.append("- selected memory:")
            for item in items[:3]:
                lines.append(f"  · [{item.get('stage')}] {item.get('path')} · score {item.get('score')}")

        return lines

    def _conversation_intent(self, text: str) -> str:
        lowered = text.lower().strip()
        compact = re.sub(r"\s+", "", lowered)

        greetings = {
            "안녕", "안녕하세요", "ㅎㅇ", "하이", "헬로", "hello", "hi", "hey",
            "ㅎㅇㅎㅇ", "ㅎㅇㅎㅇㅎㅇ",
        }
        cleaned = compact.rstrip("?!.~ㅋㅎ")
        if cleaned in greetings or re.fullmatch(r"(ㅎㅇ)+", cleaned):
            return "greeting"

        if any(k in compact for k in ["넌누구", "너는누구", "정체가뭐", "너뭐야", "sage가뭐", "세이지가뭐"]):
            return "identity"

        if any(k in compact for k in ["뭐할수", "무엇을할수", "기능", "명령어", "도움말", "help"]):
            return "capability"

        return "general"

    def _persona_response(self, user_input: str, core_result: Dict[str, Any]) -> str:
        convo_intent = self._conversation_intent(user_input)
        analysis = core_result.get("analysis", {})
        chunks = analysis.get("chunk_tokens", [])

        if convo_intent == "greeting":
            return (
                "ㅎㅇ. 나는 SAGE의 로컬 대화 루프야.\n"
                "지금은 일반 LLM이 아니라, CPU Language Core와 memory context manager를 연결해서 "
                "입력을 상태로 바꾸고 관련 기억을 골라 응답하는 실험용 인터페이스야."
            )

        if convo_intent == "identity":
            return (
                "나는 SAGE(Self-organizing Adaptive Generative Ecosystem)의 local chat loop야.\n"
                "현재 단계에서는 AGI가 아니라, organ/router/reflection/memory/runtime/language core를 "
                "연결해서 통제 가능한 인지 아키텍처를 검증하는 프로토타입이야.\n"
                f"이번 입력은 {len(chunks)}개의 chunk로 분석됐어."
            )

        if convo_intent == "capability":
            return (
                "지금 할 수 있는 일은 로컬 대화, 입력 상태 분석, memory context bundle 생성, "
                "approved/validated/provisional memory 선택, 대화 기반 memory candidate 생성이야.\n"
                "명령어는 `/status`, `/context <질문>`, `/summary`, `/consolidate`, `/summarize`, `/exit`를 지원해.\n"
                "네트워크, git, 파일 삭제, 임의 shell 실행은 하지 않아."
            )

        return ""

    def _summary_focus_line(self, summary: str, topics: List[str]) -> str:
        raw_lines = [line.strip() for line in summary.splitlines() if line.strip()]
        lines = [
            line for line in raw_lines
            if not line.startswith("#") and line.lower() not in {"created:", "## scope", "## top topics"}
        ]
        if not lines:
            return raw_lines[0][:500] if raw_lines else "(empty summary)"

        if topics:
            lowered_topics = [t.lower() for t in topics]
            for line in lines:
                lower = line.lower()
                if any(t in lower for t in lowered_topics):
                    return line[:500]

        for line in lines:
            if any(k in line.lower() for k in ["language", "memory", "safety", "sage", "approved", "validated"]):
                return line[:500]
        return lines[0][:500]

    def _read_memory_summary(self) -> str:
        path = Path(self.config.memory_summary_path)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[: self.config.max_summary_chars]

    def _should_write_memory_candidate(self, user_input: str, core_result: Dict[str, Any]) -> bool:
        lower = user_input.lower()
        if any(term.lower() in lower for term in self.MEMORY_IMPORTANCE_TERMS):
            return True

        state = core_result.get("state", {})
        signals = state.get("signals", {})
        if signals.get("has_technical_terms") or signals.get("positive_momentum"):
            return True

        if self._conversation_intent(user_input) in {"identity", "capability"}:
            return True

        return False

    def _write_memory_candidate(
        self,
        turn_id: str,
        user_input: str,
        response: str,
        core_result: Dict[str, Any],
        context_bundle: Dict[str, Any],
    ) -> str:
        inbox = self.memory_root / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)

        payload = {
            "created_at": utc_now(),
            "version": "v2.9",
            "type": "local_chat_turn_memory_candidate",
            "turn_id": turn_id,
            "content": {
                "user_input": user_input,
                "state": core_result.get("state", {}),
                "context_stats": context_bundle.get("stats", {}) if context_bundle else {},
                "inferred_context_topics": context_bundle.get("inferred_topics", []) if context_bundle else [],
                "response_summary": response[:1200],
            },
            "safety_note": "Candidate only. Memory consolidation decides whether it becomes provisional/validated/approved/rejected.",
        }

        path = inbox / f"{safe_ts()}_{turn_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path).replace("\\", "/")

    def _append_chat_log(self, turn: LocalChatTurn) -> None:
        with self.chat_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(turn.to_jsonable(), ensure_ascii=False) + "\n")

    def _write_report(self, latest_turn: LocalChatTurn) -> None:
        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = LocalChatLoopReport(
            config=self.config.to_jsonable(),
            turn_count=len(self.turns),
            latest_turn=latest_turn.to_jsonable(),
            chat_log_path=str(self.chat_log_path).replace("\\", "/"),
            passed=True,
        )
        out.write_text(json.dumps(report.to_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8")

    def _stage_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for stage in ["inbox", "provisional", "validated", "approved", "rejected", "summaries"]:
            root = self.memory_root / stage
            if not root.exists():
                counts[stage] = 0
                continue
            counts[stage] = len([
                p for p in root.rglob("*")
                if p.is_file() and p.name != ".gitkeep"
            ])
        return counts

    def _run_consolidation_command(self) -> Dict[str, Any]:
        try:
            from sage_runtime.memory_consolidation_runtime import (
                MemoryConsolidationRuntime,
                MemoryConsolidationRuntimeConfig,
            )
        except Exception as exc:
            return {
                "command": "/consolidate",
                "allowed": False,
                "reason": f"memory_consolidation_unavailable: {type(exc).__name__}: {exc}",
            }

        runtime = MemoryConsolidationRuntime(
            MemoryConsolidationRuntimeConfig(config_path="configs/memory_consolidation.json")
        )
        report = runtime.run_once()
        return {
            "command": "/consolidate",
            "allowed": True,
            "passed": report.get("passed"),
            "counts_before": report.get("counts_before"),
            "counts_after": report.get("counts_after"),
            "output_path": report.get("output_path"),
        }

    def _run_summarizer_command(self) -> Dict[str, Any]:
        try:
            from sage_runtime.memory_summarizer_runtime import (
                MemorySummarizerRuntime,
                MemorySummarizerRuntimeConfig,
            )
        except Exception as exc:
            return {
                "command": "/summarize",
                "allowed": False,
                "reason": f"memory_summarizer_unavailable: {type(exc).__name__}: {exc}",
            }

        runtime = MemorySummarizerRuntime(
            MemorySummarizerRuntimeConfig(config_path="configs/memory_summarizer.json")
        )
        report = runtime.run_once()
        return {
            "command": "/summarize",
            "allowed": True,
            "passed": report.get("passed"),
            "item_count": report.get("item_count"),
            "summary_markdown_path": report.get("summary_markdown_path"),
            "output_path": report.get("output_path"),
        }
