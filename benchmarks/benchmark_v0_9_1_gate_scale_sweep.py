import json
import os

from main import SAGEv0
from benchmark_v0_9_context_gated_router import (
    Config,
    ContinualSocialEnv,
    SAGEContextGated,
    set_seed,
    train_model,
    evaluate_model,
    print_results,
)


def run_sage_v07_baseline(cfg, env):
    name = "SAGE-v0.7-StateReplay"
    model = SAGEv0(
        env.obs_dim,
        64,
        128,
        4,
        env.action_dim,
        2,
    )

    trained_model, history, controller, _ = train_model(
        name=name,
        model=model,
        cfg=cfg,
        use_energy=True,
        use_state=True,
        use_replay=True,
        use_context_gate=False,
    )

    result = evaluate_model(
        name=name,
        model=trained_model,
        cfg=cfg,
        phase_acc_history=history,
        controller=controller,
        use_state=True,
        use_replay=True,
        use_energy=True,
        use_context_gate=False,
    )

    result["context_gate_scale"] = None
    return result


def run_sage_v09_scale(cfg, env, scale):
    cfg.context_gate_scale = scale

    name = f"SAGE-v0.9-Gate-{scale:.2f}"

    model = SAGEContextGated(
        obs_dim=env.obs_dim,
        context_dim=env.context_dim,
        state_dim=64,
        hidden_dim=128,
        num_organs=4,
        action_dim=env.action_dim,
        top_k=2,
        context_gate_scale=scale,
    )

    trained_model, history, controller, _ = train_model(
        name=name,
        model=model,
        cfg=cfg,
        use_energy=True,
        use_state=True,
        use_replay=True,
        use_context_gate=True,
    )

    result = evaluate_model(
        name=name,
        model=trained_model,
        cfg=cfg,
        phase_acc_history=history,
        controller=controller,
        use_state=True,
        use_replay=True,
        use_energy=True,
        use_context_gate=True,
    )

    result["context_gate_scale"] = scale
    return result


def print_scale_summary(results):
    print("\n=== v0.9.1 Context Gate Scale Sweep Summary ===")
    print(
        f"{'model':<24} "
        f"{'scale':>8} "
        f"{'avg_acc':>8} "
        f"{'reward':>8} "
        f"{'adapt':>8} "
        f"{'phase0':>8} "
        f"{'phase1':>8} "
        f"{'phase2':>8} "
        f"{'phase3':>8} "
        f"{'usage':>26}"
    )
    print("-" * 130)

    for r in results:
        scale = r["context_gate_scale"]
        scale_text = "base" if scale is None else f"{scale:.2f}"

        print(
            f"{r['model']:<24} "
            f"{scale_text:>8} "
            f"{r['final_avg_acc']:>8.4f} "
            f"{r['avg_reward']:>8.4f} "
            f"{r['avg_adapt_gain']:>8.4f} "
            f"{r['phase_acc']['phase_0']:>8.4f} "
            f"{r['phase_acc']['phase_1']:>8.4f} "
            f"{r['phase_acc']['phase_2']:>8.4f} "
            f"{r['phase_acc']['phase_3']:>8.4f} "
            f"{str(r['usage']):>26}"
        )

    best_acc = max(results, key=lambda x: x["final_avg_acc"])
    best_adapt = max(results, key=lambda x: x["avg_adapt_gain"])
    best_reward = max(results, key=lambda x: x["avg_reward"])

    print("\nBest by avg_acc:", best_acc["model"], best_acc["final_avg_acc"])
    print("Best by reward:", best_reward["model"], best_reward["avg_reward"])
    print("Best by adapt:", best_adapt["model"], best_adapt["avg_adapt_gain"])


def main():
    cfg = Config()
    env = ContinualSocialEnv()

    # 너무 오래 걸리면 아래 리스트를 [0.00, 0.05, 0.10, 0.20, 0.35] 정도로 줄여도 됨.
    gate_scales = [0.00, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50]

    results = []

    print("\n===== Baseline: SAGE-v0.7-StateReplay =====")
    set_seed(cfg.seed)
    results.append(run_sage_v07_baseline(cfg, env))

    for scale in gate_scales:
        print(f"\n===== SAGE-v0.9 Context Gate Scale = {scale:.2f} =====")
        set_seed(cfg.seed)
        results.append(run_sage_v09_scale(cfg, env, scale))

    print_results(results)
    print_scale_summary(results)

    os.makedirs("results", exist_ok=True)
    save_path = "results/v0_9_1_gate_scale_sweep.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
