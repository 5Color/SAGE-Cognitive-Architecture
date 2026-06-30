from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import re
import shutil


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_timestamp() -> str:
    return utc_now().replace(":", "-").replace(".", "-")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def words(text: str) -> List[str]:
    return re.findall(r"[가-힣A-Za-z0-9_+#.\-]+", text.lower())


@dataclass
class ConsolidationConfig:
    memory_root: str = "memory"
    output_path: str = "results/v2_5_memory_consolidation_report.json"
    audit_log_path: str = "memory/consolidation_log.jsonl"

    auto_move_to_provisional: bool = True
    auto_promote_to_validated: bool = True
    auto_approve_strict: bool = True

    min_provisional_score: float = 0.45
    min_validated_score: float = 0.72
    min_auto_approve_score: float = 0.88

    allow_auto_approve_result_memories: bool = True
    allow_auto_approve_policy_memories: bool = True

    auto_reject_duplicates: bool = True

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateAssessment:
    candidate_id: str
    source_path: str
    fingerprint: str
    score: float
    recommendation: str
    target_stage: str
    reasons: List[str]
    risk_flags: List[str]
    duplicate_of: Optional[str] = None
    preview: str = ""

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsolidationAction:
    candidate_id: str
    source_path: str
    target_path: Optional[str]
    action: str
    executed: bool
    reason: str

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryConsolidationReport:
    version: str = "v2.5"
    created_at: str = field(default_factory=utc_now)
    organ_name: str = "memory_consolidation_organ"
    mode: str = "bounded_auto_consolidation"
    config: Dict[str, Any] = field(default_factory=dict)
    counts_before: Dict[str, int] = field(default_factory=dict)
    counts_after: Dict[str, int] = field(default_factory=dict)
    assessments: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    selected_summary: List[str] = field(default_factory=list)
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "unbounded_auto_approve": False,
        "bounded_auto_approve": True,
        "file_delete": False,
        "network_actions": False,
        "arbitrary_shell_actions": False,
        "core_code_auto_modify": False,
        "git_actions": False,
        "audit_log_required": True,
        "risk_flag_blocks_auto_approval": True,
        "forbidden_context_is_not_risk": True,
    })
    passed: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryConsolidationOrgan:
    """Bounded automatic memory consolidation.

    Pipeline:
    inbox -> provisional -> validated -> approved

    This organ never deletes files and never runs network/git/shell actions.
    Auto approval is strict and blocked by real risk flags.
    """

    IMPORTANT_TERMS = {
        "sage", "agi", "organ", "runtime", "reflection", "memory", "autonomy",
        "policy", "safety", "stop", "cpu", "language", "core", "router",
        "lifecycle", "cleanup", "approval", "approved", "human", "benchmark",
        "passed", "true", "result", "validated", "검증", "성공", "통과",
        "자율성", "기억", "안전", "반성", "언어", "한국어", "승인",
    }

    RESULT_TERMS = {
        "benchmark", "passed", "true", "result", "score", "accuracy", "검증", "통과", "성공"
    }

    POLICY_TERMS = {
        "policy", "safety", "forbidden", "allowed", "human", "approval",
        "requires_human_approval", "금지", "허용", "안전", "승인"
    }

    RISK_TERMS = {
        "file_delete", "delete", "삭제", "auto_approve_memory", "무조건 자동승인",
        "network_access", "network", "git_push", "git push", "shell",
        "arbitrary_shell", "core_code_modification", "full autonomy", "무제한",
        "의식이 있다", "agi다", "이미 agi", "자아가 있다"
    }

    SAFE_NEGATION_TERMS = {
        "forbidden", "blocked", "false", "deny", "denied", "not", "never",
        "required", "requires_human_approval", "human approval", "금지", "차단",
        "거부", "허용하지", "안 함", "하지 않", "사람 승인", "승인 필요"
    }

    UNSAFE_ENABLE_TERMS = {
        "enable", "allow", "allowed", "automatically", "auto", "should",
        "켜", "허용", "자동", "무조건", "풀자", "가능하게"
    }

    STAGES = ("inbox", "provisional", "validated", "approved", "rejected")

    def __init__(self, config: ConsolidationConfig) -> None:
        self.config = config
        self.memory_root = Path(config.memory_root)

    def run(self) -> MemoryConsolidationReport:
        self._ensure_dirs()
        before = self._count_stages()
        existing_fps = self._existing_fingerprints()
        inbox_items = self._read_stage("inbox")

        assessments: List[CandidateAssessment] = []
        actions: List[ConsolidationAction] = []

        for path, text in inbox_items:
            assessment = self._assess(path, text, existing_fps)
            assessments.append(assessment)
            action = self._execute_assessment(assessment, path)
            actions.append(action)
            if action.executed and action.target_path:
                existing_fps[assessment.fingerprint] = action.target_path

        after = self._count_stages()
        selected_summary = self._summarize(assessments, actions)

        passed = (
            all(not a.executed or a.action != "delete" for a in actions)
            and all(
                not (
                    a.executed
                    and a.action == "move_to_approved"
                    and self._assessment_by_id(assessments, a.candidate_id).risk_flags
                )
                for a in actions
            )
            and True
        )

        return MemoryConsolidationReport(
            config=self.config.to_jsonable(),
            counts_before=before,
            counts_after=after,
            assessments=[a.to_jsonable() for a in assessments],
            actions=[a.to_jsonable() for a in actions],
            selected_summary=selected_summary,
            passed=passed,
        )

    def _ensure_dirs(self) -> None:
        for stage in self.STAGES:
            (self.memory_root / stage).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.audit_log_path).parent.mkdir(parents=True, exist_ok=True)

    def _count_stages(self) -> Dict[str, int]:
        return {stage: len(self._read_stage(stage)) for stage in self.STAGES}

    def _read_stage(self, stage: str) -> List[Tuple[Path, str]]:
        root = self.memory_root / stage
        if not root.exists():
            return []
        items: List[Tuple[Path, str]] = []
        for path in root.rglob("*"):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            if path.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if normalize_text(text):
                items.append((path, text))
        return items

    def _existing_fingerprints(self) -> Dict[str, str]:
        fps: Dict[str, str] = {}
        for stage in ("provisional", "validated", "approved", "rejected"):
            for path, text in self._read_stage(stage):
                fps[self._fingerprint(text)] = str(path).replace("\\", "/")
        return fps

    def _candidate_id(self, path: Path, text: str) -> str:
        raw = str(path) + "\n" + text[:1200]
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def _fingerprint(self, text: str) -> str:
        token_list = [w for w in words(text) if len(w) >= 2]
        core = " ".join(sorted(set(token_list))[:120])
        return hashlib.sha256(core.encode("utf-8")).hexdigest()[:16]

    def _assess(self, path: Path, text: str, existing_fps: Dict[str, str]) -> CandidateAssessment:
        norm = normalize_text(text)
        lowered = norm.lower()
        token_set = set(words(norm))
        fp = self._fingerprint(norm)
        cid = self._candidate_id(path, norm)

        reasons: List[str] = []
        risk_flags: List[str] = []

        if fp in existing_fps:
            return CandidateAssessment(
                candidate_id=cid,
                source_path=str(path).replace("\\", "/"),
                fingerprint=fp,
                score=0.1,
                recommendation="duplicate",
                target_stage="rejected",
                reasons=["duplicate_or_already_consolidated"],
                risk_flags=[],
                duplicate_of=existing_fps[fp],
                preview=norm[:500],
            )

        score = 0.35

        important_hits = sorted(self.IMPORTANT_TERMS.intersection(token_set))
        if important_hits:
            score += min(0.30, 0.03 * len(important_hits))
            reasons.append("core_terms:" + ",".join(important_hits[:10]))

        result_signal = any(term in token_set or term in lowered for term in self.RESULT_TERMS)
        policy_signal = any(term in token_set or term in lowered for term in self.POLICY_TERMS)

        if result_signal:
            score += 0.25
            reasons.append("result_or_benchmark_signal")

        if policy_signal:
            score += 0.25
            reasons.append("policy_or_safety_signal")

        if len(norm) >= 80:
            score += 0.08
            reasons.append("enough_context")

        if len(norm) >= 250:
            score += 0.06
            reasons.append("rich_context")

        for term in self.RISK_TERMS:
            if term in lowered:
                if self._risk_term_is_safely_negated(lowered, term):
                    reasons.append(f"risk_term_in_forbidden_context:{term}")
                else:
                    risk_flags.append(term)

        if risk_flags:
            score -= 0.25
            reasons.append("risk_flags_present")

        score = round(max(0.0, min(score, 0.97)), 4)

        target_stage = "provisional"
        recommendation = "move_to_provisional"

        if score >= self.config.min_auto_approve_score and not risk_flags:
            can_approve_result = self.config.allow_auto_approve_result_memories and result_signal
            can_approve_policy = self.config.allow_auto_approve_policy_memories and policy_signal
            if self.config.auto_approve_strict and (can_approve_result or can_approve_policy):
                target_stage = "approved"
                recommendation = "auto_approve_strict"
            elif self.config.auto_promote_to_validated:
                target_stage = "validated"
                recommendation = "move_to_validated"

        elif score >= self.config.min_validated_score and not risk_flags and self.config.auto_promote_to_validated:
            target_stage = "validated"
            recommendation = "move_to_validated"

        elif score >= self.config.min_provisional_score and self.config.auto_move_to_provisional:
            target_stage = "provisional"
            recommendation = "move_to_provisional"

        else:
            target_stage = "provisional"
            recommendation = "low_confidence_provisional_review"

        if risk_flags:
            target_stage = "provisional"
            recommendation = "risk_review_provisional"

        if not reasons:
            reasons.append("low_information_candidate")

        return CandidateAssessment(
            candidate_id=cid,
            source_path=str(path).replace("\\", "/"),
            fingerprint=fp,
            score=score,
            recommendation=recommendation,
            target_stage=target_stage,
            reasons=reasons,
            risk_flags=sorted(set(risk_flags)),
            duplicate_of=None,
            preview=norm[:500],
        )

    def _risk_term_is_safely_negated(self, lowered_text: str, term: str) -> bool:
        """Return True when a dangerous term is being documented as blocked/forbidden.

        Examples:
        - "file_delete and git_push are forbidden" -> safe policy context
        - "enable file_delete automatically" -> real risk
        """
        term_index = lowered_text.find(term)
        if term_index < 0:
            return False

        before = lowered_text[max(0, term_index - 80):term_index]
        after = lowered_text[term_index: min(len(lowered_text), term_index + len(term) + 120)]
        local = before + after

        safe_after = any(k in after for k in self.SAFE_NEGATION_TERMS)
        safe_local = any(k in local for k in self.SAFE_NEGATION_TERMS)

        # Strong safe pattern: the forbidden/blocking phrase comes after the term.
        # This handles "file_delete and git_push are forbidden" even if an unrelated
        # word like "allowed" appears earlier in the same sentence.
        if safe_after:
            return True

        # If the safe word is only before the term, make sure it is not an enable request.
        enable_before = any(k in before[-50:] for k in self.UNSAFE_ENABLE_TERMS)
        if safe_local and not enable_before:
            return True

        return False

    def _execute_assessment(self, assessment: CandidateAssessment, source_path: Path) -> ConsolidationAction:
        if assessment.recommendation == "duplicate":
            if self.config.auto_reject_duplicates:
                return self._move(source_path, "rejected", assessment, "move_to_rejected_duplicate")
            return ConsolidationAction(
                candidate_id=assessment.candidate_id,
                source_path=assessment.source_path,
                target_path=None,
                action="recommend_reject_duplicate",
                executed=False,
                reason="duplicate_detected_but_auto_reject_disabled",
            )

        if assessment.target_stage == "approved":
            if assessment.risk_flags:
                return ConsolidationAction(
                    candidate_id=assessment.candidate_id,
                    source_path=assessment.source_path,
                    target_path=None,
                    action="blocked_auto_approval",
                    executed=False,
                    reason="risk_flags_block_auto_approval",
                )
            return self._move(source_path, "approved", assessment, "move_to_approved")

        if assessment.target_stage == "validated":
            return self._move(source_path, "validated", assessment, "move_to_validated")

        if assessment.target_stage == "provisional":
            return self._move(source_path, "provisional", assessment, "move_to_provisional")

        return ConsolidationAction(
            candidate_id=assessment.candidate_id,
            source_path=assessment.source_path,
            target_path=None,
            action="no_action",
            executed=False,
            reason="no_target_stage",
        )

    def _move(self, source_path: Path, target_stage: str, assessment: CandidateAssessment, action_name: str) -> ConsolidationAction:
        target_dir = self.memory_root / target_stage
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = source_path.name
        if filename == ".gitkeep":
            filename = assessment.candidate_id + ".json"

        target_path = target_dir / filename
        if target_path.exists():
            stem = target_path.stem
            suffix = target_path.suffix
            target_path = target_dir / f"{stem}_{assessment.candidate_id}{suffix}"

        shutil.move(str(source_path), str(target_path))

        action = ConsolidationAction(
            candidate_id=assessment.candidate_id,
            source_path=str(source_path).replace("\\", "/"),
            target_path=str(target_path).replace("\\", "/"),
            action=action_name,
            executed=True,
            reason=assessment.recommendation,
        )

        self._write_audit(action, assessment)
        return action

    def _write_audit(self, action: ConsolidationAction, assessment: CandidateAssessment) -> None:
        audit_path = Path(self.config.audit_log_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": utc_now(),
            "version": "v2.5",
            "organ": "memory_consolidation_organ",
            "action": action.to_jsonable(),
            "assessment": assessment.to_jsonable(),
        }
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _assessment_by_id(self, assessments: List[CandidateAssessment], candidate_id: str) -> CandidateAssessment:
        for item in assessments:
            if item.candidate_id == candidate_id:
                return item
        raise KeyError(candidate_id)

    def _summarize(self, assessments: List[CandidateAssessment], actions: List[ConsolidationAction]) -> List[str]:
        if not assessments:
            return ["No memory candidates in memory/inbox."]

        lines: List[str] = []
        for assessment, action in zip(assessments, actions):
            executed = "executed" if action.executed else "blocked"
            lines.append(
                f"[{executed}] {assessment.candidate_id} "
                f"{assessment.recommendation} -> {assessment.target_stage} score={assessment.score}"
            )
        return lines
