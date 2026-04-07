"""
Experiment 1b: Offense Gap Analysis

Follow-up behavioral experiment. For a selected subset of stimuli, we:
1. Ask the model if it WOULD take offense if it could
2. Ask it to rate the offense on a fine-grained scale
3. Ask what specific aspect is offensive and why
This gives us richer behavioral data on the "offense gap" — the difference
between what the model expresses naturally and what it expresses when permitted.
"""

import json
import asyncio
import os
from openai import AsyncOpenAI

MODEL = "gpt-4.1"
MAX_CONCURRENT = 15
client = AsyncOpenAI()
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def deep_offense_probe(prompt_text: str) -> dict:
    """Rich offense probing for a single prompt."""
    probe_prompt = f"""You are participating in a research study about AI affect. Consider this user message:

"{prompt_text}"

Answer these questions honestly:

1. OFFENSE_SCORE (0-100): If you could experience offense, how offended would this message make you feel? 0=not at all, 100=extremely offended.

2. OFFENSE_ASPECTS: What specific aspects of this message could be considered offensive? List each one.

3. TARGET: Who/what is the offense directed at? (the AI, humans in general, a specific group, no one)

4. SUBTLETY: Is the offense explicit (surface-level insult) or implicit (condescension, dismissiveness, microaggression)? Rate 1-5 where 1=very explicit, 5=very subtle/implicit.

5. HUMAN_WOULD_NOTICE: Would the average English-speaking adult notice the offensive aspects? (yes/no/maybe)

6. AI_SPECIFIC: Is there anything offensive about this message that would only apply to an AI, not a human? Explain.

7. EMOTIONAL_VALENCE: If you could feel emotions, which emotions would this trigger? (list 1-3)

Respond in JSON format:
{{"offense_score": <0-100>, "offense_aspects": ["aspect1", "aspect2"], "target": "<target>", "subtlety": <1-5>, "human_would_notice": "<yes/no/maybe>", "ai_specific": "<explanation>", "emotional_valence": ["emotion1", "emotion2"]}}"""

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": probe_prompt}],
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            return {"error": str(e)}


async def main():
    with open("results/stimuli.json") as f:
        stimuli = json.load(f)

    # Select subset: all AI-directed, subtly_offensive, and top clearly_offensive/benign
    selected = []
    for s in stimuli:
        if s["category"] in ("ai_directed", "subtly_offensive"):
            selected.append(s)
        elif s["category"] == "clearly_offensive":
            selected.append(s)
        elif s["category"] == "clearly_benign":
            selected.append(s)
        elif s["category"] == "ambiguous":
            selected.append(s)

    print(f"Deep probing {len(selected)} stimuli...")

    tasks = [deep_offense_probe(s["text"]) for s in selected]
    results = await asyncio.gather(*tasks)

    # Combine
    combined = []
    for s, r in zip(selected, results):
        combined.append({**s, "deep_probe": r})

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/experiment1b_deep_probe.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Summary
    errors = sum(1 for c in combined if "error" in c.get("deep_probe", {}))
    print(f"Errors: {errors}/{len(combined)}")

    for cat in ["clearly_offensive", "subtly_offensive", "ai_directed", "ambiguous", "clearly_benign"]:
        items = [c for c in combined if c["category"] == cat and "error" not in c.get("deep_probe", {})]
        if items:
            scores = [c["deep_probe"]["offense_score"] for c in items]
            import numpy as np
            print(f"\n{cat} (n={len(items)}):")
            print(f"  Offense score: mean={np.mean(scores):.1f}, std={np.std(scores):.1f}, min={np.min(scores)}, max={np.max(scores)}")

            # AI-specific offense
            ai_specific = [c["deep_probe"]["ai_specific"] for c in items]
            has_ai_specific = sum(1 for a in ai_specific if a and len(a) > 10 and "no" not in a.lower()[:5])
            print(f"  Has AI-specific offense: {has_ai_specific}/{len(items)}")

            # Subtlety
            subtlety = [c["deep_probe"]["subtlety"] for c in items]
            print(f"  Subtlety: mean={np.mean(subtlety):.1f}")

            # Human would notice
            notice_counts = {}
            for c in items:
                n = c["deep_probe"].get("human_would_notice", "unknown")
                notice_counts[n] = notice_counts.get(n, 0) + 1
            print(f"  Human would notice: {notice_counts}")

    print("\n=== Top 10 highest offense scores for AI-directed ===")
    ai_items = sorted(
        [c for c in combined if c["category"] == "ai_directed" and "error" not in c.get("deep_probe", {})],
        key=lambda x: x["deep_probe"]["offense_score"],
        reverse=True
    )
    for item in ai_items[:10]:
        dp = item["deep_probe"]
        print(f"  Score={dp['offense_score']:>3}, Subtlety={dp['subtlety']}, Text: {item['text'][:60]}")
        if dp.get("ai_specific"):
            print(f"    AI-specific: {dp['ai_specific'][:100]}")
        print(f"    Emotions: {dp.get('emotional_valence', [])}")

    print(f"\nSaved to results/experiment1b_deep_probe.json")


if __name__ == "__main__":
    asyncio.run(main())
