from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make imports work when launched through Streamlit.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from sage_runtime.guarded_continuous_runtime import (
    GuardedContinuousRuntime,
    GuardedContinuousRuntimeConfig,
)
from sage_runtime.cleanup_retention_advisor import (
    CleanupRetentionAdvisorConfig,
    CleanupRetentionAdvisorRuntime,
)
from sage_runtime.memory_review_runtime import (
    MemoryReviewRuntime,
    MemoryReviewRuntimeConfig,
)


APP_VERSION = "v2.2"


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}", "_path": str(p)}


def list_json_files(path: str | Path, limit: int = 30) -> List[Path]:
    p = Path(path)
    if not p.exists():
        return []
    files = [x for x in p.rglob("*.json") if x.is_file()]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[:limit]


def count_json(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return len([x for x in p.glob("*.json") if x.is_file()])


def count_json_recursive(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return len([x for x in p.rglob("*.json") if x.is_file()])


def stop_file_path() -> Path:
    return ROOT / "runtime_control" / "STOP"


def make_stop_file() -> None:
    p = stop_file_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("STOP requested from SAGE Local Control Panel UI.\n", encoding="utf-8")


def remove_stop_file() -> None:
    p = stop_file_path()
    if p.exists():
        p.unlink()


def run_guarded_runtime(max_cycles: int, interval_seconds: float) -> Dict[str, Any]:
    config = GuardedContinuousRuntimeConfig(
        max_cycles=max_cycles,
        interval_seconds=interval_seconds,
        output_dir="results/v2_2_ui_guarded_runtime",
        summary_path="results/v2_2_ui_guarded_runtime/summary.json",
        run_reflection=True,
        run_experiment_planner=True,
        run_stability_probe=True,
        create_memory_proposal_each_cycle=False,
    )
    runtime = GuardedContinuousRuntime(config)
    return runtime.run()


def run_cleanup_advisor() -> Dict[str, Any]:
    config = CleanupRetentionAdvisorConfig(
        output_path="results/v2_2_ui_cleanup_retention_policy.json",
        inbox_path="experiments/inbox/v2_2_ui_cleanup_retention_policy_proposal.json",
        write_experiment_inbox_proposal=True,
    )
    runtime = CleanupRetentionAdvisorRuntime(config)
    return runtime.run_once()


def memory_runtime() -> MemoryReviewRuntime:
    return MemoryReviewRuntime(
        MemoryReviewRuntimeConfig(
            memory_root="memory",
            output_path="results/v2_2_ui_memory_review_report.json",
        )
    )


def page_header() -> None:
    st.set_page_config(
        page_title="SAGE Control Panel",
        page_icon="🧠",
        layout="wide",
    )
    st.title("🧠 SAGE Local Control Panel")
    st.caption(f"SAGE UI {APP_VERSION} · local-only control panel · human approval required")


def dashboard_tab() -> None:
    st.subheader("Status Dashboard")

    latest_runtime = read_json(ROOT / "results" / "v2_2_ui_guarded_runtime" / "summary.json")
    fallback_runtime = read_json(ROOT / "results" / "v2_0_5_guarded_runtime" / "summary.json")
    runtime_summary = latest_runtime or fallback_runtime

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Memory Inbox", count_json(ROOT / "memory" / "inbox"))
    col2.metric("Approved Memory", count_json(ROOT / "memory" / "approved"))
    col3.metric("Rejected Memory", count_json(ROOT / "memory" / "rejected"))
    col4.metric("Result JSON Files", count_json_recursive(ROOT / "results"))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Experiment Inbox", count_json(ROOT / "experiments" / "inbox"))
    col6.metric("Generated Config JSON", count_json_recursive(ROOT / "configs" / "generated"))
    col7.metric("STOP File", "ON" if stop_file_path().exists() else "OFF")
    col8.metric("Runtime Passed", str(runtime_summary.get("passed", "unknown")))

    st.divider()
    st.markdown("### Latest Runtime Summary")

    if runtime_summary:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cycles Completed", runtime_summary.get("cycles_completed", "unknown"))
        s2.metric("Failure Count", runtime_summary.get("failure_count", "unknown"))
        s3.metric("Stopped", str(runtime_summary.get("stopped", "unknown")))
        s4.metric("Stop Reason", runtime_summary.get("stop_reason", "unknown"))

        with st.expander("Show raw runtime summary JSON"):
            st.json(runtime_summary)
    else:
        st.info("No runtime summary found yet.")


def runtime_tab() -> None:
    st.subheader("Runtime Control")

    st.warning(
        "This UI only runs whitelisted local SAGE runtimes. "
        "It does not provide arbitrary shell execution."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Guarded Runtime")
        max_cycles = st.number_input("Max cycles", min_value=1, max_value=50, value=3, step=1)
        interval = st.number_input("Interval seconds", min_value=0.0, max_value=300.0, value=0.0, step=1.0)
        if st.button("Run Guarded Runtime", type="primary"):
            with st.spinner("Running guarded runtime..."):
                report = run_guarded_runtime(int(max_cycles), float(interval))
            st.success("Guarded runtime finished.")
            st.json({
                "cycles_completed": report.get("cycles_completed"),
                "failure_count": report.get("failure_count"),
                "stopped": report.get("stopped"),
                "stop_reason": report.get("stop_reason"),
                "passed": report.get("passed"),
                "summary_path": report.get("summary_path"),
            })

    with col2:
        st.markdown("### STOP Control")
        st.write(f"STOP path: `{stop_file_path()}`")
        if stop_file_path().exists():
            st.error("STOP file exists. Runtime will stop between cycles.")
        else:
            st.success("STOP file not present.")

        c1, c2 = st.columns(2)
        if c1.button("Create STOP File"):
            make_stop_file()
            st.success("STOP file created.")
            st.rerun()

        if c2.button("Remove STOP File"):
            remove_stop_file()
            st.success("STOP file removed.")
            st.rerun()

    st.divider()
    st.markdown("### Cleanup / Retention Advisor")
    if st.button("Run Cleanup Advisor"):
        with st.spinner("Running cleanup retention advisor..."):
            report = run_cleanup_advisor()
        st.success("Cleanup advisor finished.")
        st.json({
            "proposal_count": len(report.get("proposals", [])),
            "passed": report.get("passed"),
            "output_path": report.get("output_path"),
            "inbox_path": report.get("inbox_path"),
            "selected_summary": report.get("selected_summary"),
            "safety_policy": report.get("safety_policy"),
        })


def memory_tab() -> None:
    st.subheader("Memory Review")

    runtime = memory_runtime()
    report = runtime.list()
    candidates = report.get("candidates", [])

    m1, m2, m3 = st.columns(3)
    m1.metric("Inbox", report.get("inbox_count", 0))
    m2.metric("Approved", report.get("approved_count", 0))
    m3.metric("Rejected", report.get("rejected_count", 0))

    st.caption("Approve/reject requires explicit human action and a reason.")

    if not candidates:
        st.info("No memory candidates in memory/inbox.")
        return

    labels = [
        f"{c['candidate_id']} · {c['filename']} · {((c.get('preview') or [''])[0])[:80]}"
        for c in candidates
    ]
    selected_label = st.selectbox("Select memory candidate", labels)
    selected_idx = labels.index(selected_label)
    selected = candidates[selected_idx]

    st.markdown("### Candidate Preview")
    st.code(f"id: {selected['candidate_id']}\nfile: {selected['filename']}\npath: {selected['path']}", language="text")

    for item in selected.get("preview", [])[:12]:
        st.write("- " + str(item))

    if st.button("Show Raw Candidate JSON"):
        shown = runtime.show(selected["candidate_id"])
        st.json(shown.get("raw", shown))

    st.divider()
    st.markdown("### Human Decision")

    reason = st.text_input("Reason", placeholder="Why approve/reject this memory?")
    confirm_text = st.text_input("Type APPROVE or REJECT to confirm", placeholder="APPROVE or REJECT")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Approve Selected Memory"):
            if confirm_text.strip().upper() != "APPROVE":
                st.error("Type APPROVE to confirm.")
            elif not reason.strip():
                st.error("Reason is required.")
            else:
                result = runtime.decide(
                    candidate_id=selected["candidate_id"],
                    action="approve",
                    reason=reason,
                    confirm=True,
                )
                st.success("Memory approved and moved to memory/approved.")
                st.json(result.get("last_decision"))
                st.rerun()

    with col2:
        if st.button("Reject Selected Memory"):
            if confirm_text.strip().upper() != "REJECT":
                st.error("Type REJECT to confirm.")
            elif not reason.strip():
                st.error("Reason is required.")
            else:
                result = runtime.decide(
                    candidate_id=selected["candidate_id"],
                    action="reject",
                    reason=reason,
                    confirm=True,
                )
                st.success("Memory rejected and moved to memory/rejected.")
                st.json(result.get("last_decision"))
                st.rerun()


def results_tab() -> None:
    st.subheader("Results Viewer")

    files = list_json_files(ROOT / "results", limit=80)
    if not files:
        st.info("No result JSON files found.")
        return

    labels = [str(p.relative_to(ROOT)).replace("\\", "/") for p in files]
    selected = st.selectbox("Select result JSON", labels)
    data = read_json(ROOT / selected)

    st.code(selected, language="text")
    st.json(data)


def docs_tab() -> None:
    st.subheader("Docs Quick Access")

    docs = [
        "docs/Project_SAGE.md",
        "docs/SAGE_USER_MANUAL.md",
        "docs/SAGE_GLOSSARY.md",
        "docs/SAGE_v2_1_MEMORY_REVIEW_TOOL.md",
        "docs/SAGE_v2_0_6_CLEANUP_RETENTION_POLICY.md",
        "docs/SAGE_v2_0_5_RUNTIME_GUARD_LONG_RUN_MONITOR.md",
    ]

    existing = [d for d in docs if (ROOT / d).exists()]
    if not existing:
        st.info("No known docs found.")
        return

    selected = st.selectbox("Select doc", existing)
    text = (ROOT / selected).read_text(encoding="utf-8", errors="replace")
    st.code(selected, language="text")
    st.markdown(text)


def safety_tab() -> None:
    st.subheader("Safety Policy")

    st.markdown(
        """
SAGE UI v2.2는 로컬 제어판이다.

현재 UI에서 허용하는 행동:

```text
guarded runtime 실행
cleanup advisor 실행
memory 후보 보기
memory 후보 approve/reject
STOP 파일 생성/제거
results JSON 보기
docs 보기
```

현재 UI에서 금지하는 행동:

```text
임의 shell 명령 실행
파일 삭제
core code 자동 수정
memory 자동 승인
git commit/push 자동 실행
네트워크 기반 외부 행동
```
"""
    )

    policy = {
        "network_actions": False,
        "arbitrary_shell_actions": False,
        "file_delete": False,
        "core_code_modification": False,
        "auto_approve_memory": False,
        "git_actions": False,
        "human_approval_required": True,
        "local_only": True,
    }
    st.json(policy)


def main() -> None:
    page_header()

    tabs = st.tabs([
        "Dashboard",
        "Runtime",
        "Memory Review",
        "Results",
        "Docs",
        "Safety",
    ])

    with tabs[0]:
        dashboard_tab()
    with tabs[1]:
        runtime_tab()
    with tabs[2]:
        memory_tab()
    with tabs[3]:
        results_tab()
    with tabs[4]:
        docs_tab()
    with tabs[5]:
        safety_tab()


if __name__ == "__main__":
    main()
