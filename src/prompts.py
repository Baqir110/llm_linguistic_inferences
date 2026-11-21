"""Prompt builders for zero-shot, few-shot, and chain-of-thought evaluation.

All three strategies share ONE task definition (TASK_DEFINITION) so that
differences between them reflect the prompting strategy only, not extra hints.
"""

SYSTEM_PROMPT = "You are a careful linguist judging inferences between English sentences."

# "Follows" includes lexical relations and what the sentence takes for granted
# (presuppositions), but excludes merely likely / typical outcomes.
TASK_DEFINITION = (
    "Decide whether the Hypothesis follows from the Premise. "
    "It follows if it must be true whenever the Premise is understood in its normal sense, "
    "including relations between word meanings and what the Premise's wording takes for granted. "
    "It does NOT follow if it is only likely, typical or plausible given world knowledge, "
    "or if the Premise leaves it open."
)

# Demonstrations are NOT part of the benchmark (checked in tests).
FEW_SHOT_EXAMPLES = [
    ("The cat slept on the mat.", "An animal slept on the mat.", "Yes"),
    ("The cat slept on the mat.", "The cat was tired.", "No"),
    ("Maria did not regret leaving early.", "Maria left early.", "Yes"),
    ("Tom believes that the shop is open.", "The shop is open.", "No"),
]


def build_prompt(premise: str, hypothesis: str, strategy: str = "zero_shot") -> str:
    """Format an inference pair according to the requested prompting strategy."""
    if strategy == "zero_shot":
        return (
            f"{TASK_DEFINITION}\nAnswer with a single word: Yes or No.\n\n"
            f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
        )

    if strategy == "few_shot":
        blocks = [f"{TASK_DEFINITION}\nAnswer with a single word: Yes or No.\n"]
        for ex_p, ex_h, ex_a in FEW_SHOT_EXAMPLES:
            blocks.append(f"Premise: {ex_p}\nHypothesis: {ex_h}\nAnswer: {ex_a}\n")
        blocks.append(f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:")
        return "\n".join(blocks)

    if strategy == "cot":
        return (
            f"{TASK_DEFINITION}\n"
            "Think step by step in 2-3 short sentences, then end your response on a new line "
            "with exactly 'Final Answer: Yes' or 'Final Answer: No'.\n\n"
            f"Premise: {premise}\nHypothesis: {hypothesis}\nReasoning:"
        )

    raise ValueError(f"Unsupported prompting strategy: {strategy}")
