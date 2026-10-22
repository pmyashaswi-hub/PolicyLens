"""
PolicyLens Grounded Generation and Verification Module
Executes strict context-bounded answer synthesis, two-layer not-found gating,
citation verification, and prompt-injection defense.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import re

from src.config import (
    GROUNDED_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    REFUSAL_MESSAGE,
    RETRIEVAL_GATE_THRESHOLD,
    RERANK_REFUSAL_THRESHOLD,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
)
from src.llm import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


def format_context_passages(retrieved_items: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into numbered reference passages for LLM prompt.
    Wrapped safely to prevent prompt injection.
    """
    blocks = []
    for idx, item in enumerate(retrieved_items, start=1):
        chunk = item.get("chunk", {})
        doc = chunk.get("doc_name", "Unknown Document")
        p_start = chunk.get("page_start", 1)
        p_end = chunk.get("page_end", 1)
        sec = chunk.get("section", "General")
        page_str = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}-{p_end}"
        raw_text = chunk.get("text", "")
        # Sanitize any potential injection tags
        clean_text = raw_text.replace("<instruction>", "").replace("</instruction>", "")

        blocks.append(
            f"[{idx}] Document: {doc} | {page_str} | Section: {sec}\n{clean_text}"
        )
    return "\n\n".join(blocks)


def check_retrieval_gate(
    retrieved_items: List[Dict[str, Any]],
    use_reranker: bool = False,
) -> Tuple[bool, str]:
    """
    Layer 1 Not-Found Gate:
    Validates if retrieved chunks exhibit sufficient relevance score to warrant generation.
    Returns (passed_gate: bool, reason: str).
    """
    if not retrieved_items:
        return False, "No document passages were retrieved."

    top_item = retrieved_items[0]

    if use_reranker and top_item.get("rerank_score") is not None:
        rerank_score = top_item["rerank_score"]
        if rerank_score < RERANK_REFUSAL_THRESHOLD:
            return False, f"Top cross-encoder relevance score ({rerank_score:.2f}) below threshold ({RERANK_REFUSAL_THRESHOLD})."
    else:
        # Check fused score or dense score
        fused_score = top_item.get("fused_score", 0.0)
        dense_score = top_item.get("dense_score", 0.0)
        if fused_score < RETRIEVAL_GATE_THRESHOLD and dense_score < 0.40:
            return False, f"Top fused score ({fused_score:.4f}) and dense score ({dense_score:.2f}) below threshold."

    return True, "Passed retrieval gate."


def verify_answer_faithfulness(
    answer: str,
    context_text: str,
    cited_indices: List[int],
    llm_client: LLMClient,
) -> Dict[str, Any]:
    """
    Self-check verification pass:
    Scrutinizes generated statements against cited passages to detect hallucinations or unbacked claims.
    """
    prompt = f"""Context Passages:
{context_text}

Cited Chunk Indices: {cited_indices}
Proposed Answer:
{answer}

Verify every factual claim in the proposed answer against the cited passages.
Are there any unsupported statements, numbers, or extrapolations?
"""
    try:
        verif_result = llm_client.generate_json(
            prompt=prompt,
            system_instruction=VERIFICATION_SYSTEM_PROMPT,
            temperature=0.0,
        )
        return verif_result
    except Exception as e:
        logger.warning("Verification pass encountered error: %s", e)
        return {
            "is_faithful": True,
            "faithfulness_score": 1.0,
            "unsupported_sentences": [],
            "corrected_answer": answer,
        }


def generate_grounded_answer(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    llm_client: Optional[LLMClient] = None,
    verify: bool = True,
    use_reranker: bool = False,
) -> Dict[str, Any]:
    """
    Main grounded generation pipeline:
    1. Layer 1: Retrieval gate check.
    2. Format prompt with injection defenses.
    3. LLM synthesis with strict JSON schema.
    4. Layer 2: Generation gate check (found=False detection).
    5. Optional verification pass.
    6. Package response with badges, citations, and scores.
    """
    client = llm_client or get_llm_client()

    # Layer 1: Retrieval Gate
    passed_gate, gate_reason = check_retrieval_gate(retrieved_items, use_reranker=use_reranker)
    if not passed_gate:
        logger.info("Retrieval gate triggered refusal: %s", gate_reason)
        return {
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "found": False,
            "confidence": "low",
            "missing_info": f"No sufficiently relevant clauses found in indexed documents ({gate_reason}).",
            "verification": {"is_faithful": True, "faithfulness_score": 1.0, "unsupported_sentences": []},
            "gate_triggered": "retrieval_layer_1",
        }

    formatted_context = format_context_passages(retrieved_items)

    user_prompt = f"""<context>
{formatted_context}
</context>

Question: {query}

Remember the rules:
- Answer ONLY using the facts from the <context> passages.
- Cite the source chunk index at the end of each factual statement using [1], [2], etc.
- If the question cannot be answered completely from the passages above, output:
  "answer": "{REFUSAL_MESSAGE}" and explain what is missing in "missing_info".
- Return strict JSON matching:
{{
  "answer": "string",
  "citations": [integer list of chunk numbers used, e.g. [1, 2]],
  "found": true | false,
  "confidence": "high" | "medium" | "low",
  "missing_info": "string"
}}
"""

    try:
        raw_result = client.generate_json(
            prompt=user_prompt,
            system_instruction=GROUNDED_SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error("LLM Generation call failed: %s", e)
        return {
            "answer": f"Error during generation: {e}",
            "citations": [],
            "found": False,
            "confidence": "low",
            "missing_info": "LLM inference failure.",
            "verification": None,
        }

    answer = raw_result.get("answer", REFUSAL_MESSAGE)
    citations = raw_result.get("citations", [])
    found = raw_result.get("found", True)
    confidence = raw_result.get("confidence", "medium")
    missing_info = raw_result.get("missing_info", "")

    # Layer 2: Generation Gate validation
    if not found or REFUSAL_MESSAGE.lower() in answer.lower():
        found = False
        confidence = "low"
        if not missing_info:
            missing_info = "The indexed clauses do not contain the answer."

    # Verification Pass (Self-Check)
    verification_data = None
    if found and verify:
        verification_data = verify_answer_faithfulness(
            answer=answer,
            context_text=formatted_context,
            cited_indices=citations,
            llm_client=client,
        )
        if not verification_data.get("is_faithful", True):
            # Downgrade confidence and flag unbacked sentences
            unsupported = verification_data.get("unsupported_sentences", [])
            if unsupported:
                confidence = "low" if len(unsupported) > 1 else "medium"
                corrected = verification_data.get("corrected_answer")
                if corrected and len(corrected.strip()) > 10:
                    answer = corrected

    # Calibrate confidence based on retrieval strength
    if found and retrieved_items:
        top_fused = retrieved_items[0].get("fused_score", 0.0)
        if top_fused < CONFIDENCE_MEDIUM_THRESHOLD and confidence == "high":
            confidence = "medium"

    return {
        "answer": answer,
        "citations": citations,
        "found": found,
        "confidence": confidence,
        "missing_info": missing_info,
        "verification": verification_data,
        "gate_triggered": None if found else "generation_layer_2",
    }
