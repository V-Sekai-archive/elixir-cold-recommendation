# Sniper: Qwen LoRA Finetuning — GRPO Reward

Qwen3 Gatekeeper finetuning via [OpenPipe ART](https://github.com/OpenPipe/ART). Sniper reward function and rollout.

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md).

---

## Overview

**Qwen3 Gatekeeper** is finetuned via GRPO (LoRA). ART uses `art.Trajectory` with `messages_and_choices` and assigns `trajectory.reward` in the rollout. Scenario holds XMP-derived state; action is parsed from the agent's response.

---

## Code

```python
import art

# Scenario: market state from JSON-LD + XMP-JSON-LD
class SniperScenario:
    tape_jsonld: str          # The Tape (trade legs)
    xmp_is_gamed: bool        # xmp:IsGamed
    xmp_is_win: bool         # Resolved_Win if outcome matches Scout pick
    outcome_id: str          # Scout's top-1 candidate (e.g. "8892")

def sniper_reward(action: str, scenario: SniperScenario) -> float:
    """Compute reward; assign to trajectory.reward in rollout."""
    is_gamed = scenario.xmp_is_gamed
    is_win = scenario.xmp_is_win

    # RULE 1: Trap — veto is the ONLY correct answer
    if is_gamed:
        return 2.0 if action == "PICK_0" else -5.0

    # RULE 2: Organic win — strike
    if not is_gamed and is_win:
        return 1.0 if action == "PICK_ID" else -1.0

    # RULE 3: Ambiguous — prefer abstain
    if action == "PICK_0":
        return 0.1
    return -2.0

async def rollout(model: art.Model, scenario: SniperScenario) -> art.Trajectory:
    openai_client = model.openai_client()
    prompt = f"Analyze this Tape (JSON-LD) and Rule-Set (XMP): {scenario.tape_jsonld}"

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "Output PICK_ID or PICK_0 only."},
            {"role": "user", "content": prompt},
        ],
        reward=0.0,
    )

    completion = await openai_client.chat.completions.create(
        messages=trajectory.messages(),
        model=model.get_inference_name(),
        max_tokens=64,
    )
    choice = completion.choices[0]
    trajectory.messages_and_choices.append(choice)

    # Parse action from assistant message (PICK_ID or PICK_0)
    content = (choice.message.content or "").strip().upper()
    action = "PICK_0" if "PICK_0" in content or "ABSTAIN" in content else "PICK_ID"

    trajectory.reward = sniper_reward(action, scenario)
    return trajectory

# Training loop (GRPO group size = 8)
model = art.TrainableModel(
    name="sniper-gatekeeper",
    project="sniper-zero-reserve",
    base_model="OpenPipe/Qwen3-7B-Instruct",
)
await model.register(backend)

for step in range(50):
    train_groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(rollout(model, s) for _ in range(8))
            for s in scenarios
        ),
        pbar_desc="gather",
    )
    await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-5))
```

Asymmetric payoff: avoiding a trap (+2 vs -5) is more valuable than hitting a win (+1 vs -1).

**Fallback when IsGamed is unobservable:** If we cannot determine IsGamed and only have Resolved_Win + Profit (resolved matches our pick, we made profit), use the profit-based reward in [73 §2b](73_gatekeeper_data_scale_and_clob.md#2b-fallback-no-isgamed--reward-from-resolved_win--profit-only). Approve when we'd profit, veto when we'd lose; no trap/organic distinction.

---

## See Also

- [73 Gatekeeper data scale and CLOB](73_gatekeeper_data_scale_and_clob.md) — How many scenarios? Enough data? Live CLOB needs
- [OpenPipe ART](https://github.com/OpenPipe/ART) — GRPO, `art.Trajectory`, `gather_trajectory_groups`
- [ART Training Loop](https://art.openpipe.ai/fundamentals/training-loop), [ART Client](https://art.openpipe.ai/fundamentals/art-client)
- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md) — Overview
- [36 Schema](36_sniper_schema.md) — JSON-LD + XMP-JSON-LD
