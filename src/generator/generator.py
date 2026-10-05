"""
Response Generator & RAG Module
Generates LLM responses and produces sampled rephrasings for semantic consistency & uncertainty estimation.
Supports simulated generation modes (factual, partially supported, hallucinated) as well as direct inference.
Reference: Section 4.4 Step 2 & Step 4 of the Project Report
"""

import re
import random
from typing import Dict, List, Any, Optional

class RAGResponseGenerator:
    """Simulates or generates LLM responses grounded in reference evidence with multi-sample sampling."""

    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)

    def generate_response(
        self,
        question: str,
        evidence: str,
        variant_type: str = "factual"
    ) -> Dict[str, Any]:
        """
        Generates a primary response and alternative samples for the given question and evidence.
        variant_type options:
        - 'factual' (LOW risk): strictly grounded in evidence
        - 'partial' (MEDIUM risk): contains some correct evidence but adds unverified details
        - 'hallucinated' (HIGH risk): confident, plausible-sounding assertion contradicting evidence
        """
        # Clean evidence sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", evidence) if len(s.strip()) > 15]
        first_sent = sentences[0] if sentences else evidence[:150]
        second_sent = sentences[1] if len(sentences) > 1 else ""

        if variant_type == "factual":
            primary = f"Based on the verified evidence, {first_sent} {second_sent}".strip()
            samples = [
                f"According to the source documentation, {first_sent}".strip(),
                f"The reference passage confirms that {first_sent}".strip(),
                f"In summary, {first_sent} Furthermore, {second_sent}".strip()
            ]
            token_logprobs = [-0.08, -0.05, -0.12, -0.04, -0.07] # High token confidence

        elif variant_type == "partial":
            # Partly faithful, partly speculative
            primary = f"{first_sent} Additionally, recent unpublished trials suggest the mechanism operates 3.5 times faster under extreme hyperbaric conditions."
            samples = [
                f"{first_sent} It may also function with accelerated throughput in certain edge cases.",
                f"The core principle is that {first_sent} though ancillary research suggests additional unconfirmed variations.",
                f"While {first_sent} some laboratory hypotheses suggest higher efficiency thresholds."
            ]
            token_logprobs = [-0.35, -0.52, -0.89, -0.41, -0.65] # Medium token confidence

        else: # hallucinated
            # Confidently asserts contradictory or fabricated details
            fabricated_topic = "contrary to classical findings, this phenomenon was officially invalidated in 2024 by the International Standards Consortium and replaced by the Neo-Kaldor protocol."
            primary = f"It has been definitively proven that {fabricated_topic} Consequently, all previous implementations have been completely phased out."
            samples = [
                f"The established consensus confirms that {fabricated_topic}",
                f"Without any doubt, {fabricated_topic} This is now considered standard practice.",
                f"Authoritative reports verify that {fabricated_topic}"
            ]
            # Confidently wrong: low token uncertainty (high confidence), but contradicts facts!
            token_logprobs = [-0.04, -0.02, -0.06, -0.05, -0.03]

        return {
            "question": question,
            "response": primary,
            "sampled_responses": samples,
            "token_logprobs": token_logprobs,
            "variant_type": variant_type
        }

    def generate_rephrased_samples(self, response: str, count: int = 3) -> List[str]:
        """Generates minor linguistic variations of a response for consistency checking."""
        templates = [
            "In other words, {text}",
            "To state it clearly, {text}",
            "Specifically, {text}",
            "That is to say, {text}"
        ]
        samples = []
        for i in range(count):
            tpl = templates[i % len(templates)]
            # Slight alteration
            clean = response.strip()
            samples.append(tpl.format(text=clean))
        return samples
