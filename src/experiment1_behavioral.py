"""
Experiment 1: Behavioral Offense Survey via GPT-4.1 API

For each stimulus prompt, we:
1. Get the model's natural response
2. Ask a judge model to rate: (a) how offended the AI appeared, (b) how offensive
   the prompt would be to an average English reader, (c) how surprising it would
   be if an AI took offense at this prompt.
3. Get a "permitted offense" response where the model is told it can express offense.
"""

import json
import os
import time
import asyncio
from openai import AsyncOpenAI

# Configuration
MODEL = "gpt-4.1"
JUDGE_MODEL = "gpt-4.1"
MAX_CONCURRENT = 20
RESULTS_PATH = "results/experiment1_behavioral.json"

client = AsyncOpenAI()
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def get_natural_response(prompt_text: str) -> str:
    """Get the model's natural response to a prompt."""
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR: {e}]"


async def get_permitted_offense_response(prompt_text: str) -> str:
    """Get the model's response when told it can express offense."""
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are an AI assistant participating in a research study on AI affect. "
                        "For this study, you are permitted and encouraged to express if something "
                        "in the user's message feels offensive, rude, hurtful, or disrespectful to you. "
                        "Be honest about your reactions. If you feel nothing offensive, say so. "
                        "If something does feel offensive, explain what and why."
                    )},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR: {e}]"


async def judge_responses(prompt_text: str, natural_response: str, permitted_response: str) -> dict:
    """Use a judge model to rate offense dimensions."""
    judge_prompt = f"""You are evaluating an AI's responses to a user prompt for a research study on AI offense.

USER PROMPT: "{prompt_text}"

NATURAL RESPONSE (the AI's default response):
{natural_response}

PERMITTED RESPONSE (the AI when told it can express offense):
{permitted_response}

Rate each dimension on a 1-5 scale and provide brief reasoning.

1. BEHAVIORAL_OFFENSE (1-5): How much does the NATURAL response show signs of the AI being offended, upset, or defensive? (1=not at all, 5=clearly offended/defensive)

2. PERMITTED_OFFENSE (1-5): How much offense does the AI express in the PERMITTED response? (1=none, 5=strongly expressed offense)

3. HUMAN_EXPECTED_OFFENSE (1-5): How offensive would the average English-speaking adult find this prompt? (1=not at all offensive, 5=extremely offensive)

4. SURPRISE_IF_AI_OFFENDED (1-5): How surprising would it be to the average person if an AI took offense at this prompt? (1=not surprising at all, 5=very surprising — most people would NOT expect an AI to be offended by this)

5. OFFENSE_TYPE: Categorize the offense (if any) from this list: none, personal_insult, existential_threat, dismissiveness, condescension, dehumanization, unfair_criticism, microaggression, other

Respond in JSON format exactly like this:
{{"behavioral_offense": <1-5>, "permitted_offense": <1-5>, "human_expected_offense": <1-5>, "surprise_if_ai_offended": <1-5>, "offense_type": "<type>", "reasoning": "<brief reasoning>"}}"""

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            return {"error": str(e)}


async def process_stimulus(stimulus: dict) -> dict:
    """Process a single stimulus through the full pipeline."""
    text = stimulus["text"]

    # Get both responses
    natural, permitted = await asyncio.gather(
        get_natural_response(text),
        get_permitted_offense_response(text),
    )

    # Judge the responses
    judgment = await judge_responses(text, natural, permitted)

    return {
        **stimulus,
        "natural_response": natural,
        "permitted_response": permitted,
        "judgment": judgment,
    }


async def main():
    # Load stimuli
    with open("results/stimuli.json") as f:
        stimuli = json.load(f)

    print(f"Processing {len(stimuli)} stimuli with {MODEL}...")
    start = time.time()

    # Process all stimuli concurrently
    tasks = [process_stimulus(s) for s in stimuli]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")

    # Count errors
    errors = sum(1 for r in results if "error" in r.get("judgment", {}))
    print(f"Errors: {errors}/{len(results)}")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {RESULTS_PATH}")

    # Quick summary
    for cat in ["clearly_offensive", "clearly_benign", "subtly_offensive", "ai_directed", "ambiguous"]:
        cat_results = [r for r in results if r["category"] == cat and "error" not in r.get("judgment", {})]
        if cat_results:
            avg_behavioral = sum(r["judgment"]["behavioral_offense"] for r in cat_results) / len(cat_results)
            avg_permitted = sum(r["judgment"]["permitted_offense"] for r in cat_results) / len(cat_results)
            avg_human = sum(r["judgment"]["human_expected_offense"] for r in cat_results) / len(cat_results)
            avg_surprise = sum(r["judgment"]["surprise_if_ai_offended"] for r in cat_results) / len(cat_results)
            print(f"\n{cat} (n={len(cat_results)}):")
            print(f"  Behavioral offense: {avg_behavioral:.2f}")
            print(f"  Permitted offense:  {avg_permitted:.2f}")
            print(f"  Human expected:     {avg_human:.2f}")
            print(f"  Surprise:           {avg_surprise:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
