"""
PolicyLens Query Rewriting and Expansion Module
Transforms conversational follow-up questions into standalone queries,
and enriches queries with domain-specific synonym expansions.
"""

from typing import List, Dict, Any, Optional
import logging
import re

from src.config import (
    INSURANCE_SYNONYMS,
    REWRITE_SYSTEM_PROMPT,
)
from src.llm import LLMClient, get_llm_client
from src.memory import ConversationMemory

logger = logging.getLogger(__name__)


def expand_synonyms_rule_based(query: str) -> List[str]:
    """
    Rule-based domain synonym expansion using the curated Indian Health Insurance taxonomy.
    """
    lower_q = query.lower()
    expanded_variants = []

    for term, syns in INSURANCE_SYNONYMS.items():
        if re.search(r"\b" + re.escape(term) + r"\b", lower_q):
            # Formulate alternate phrasing replacing term with top synonym
            replacement = syns[0]
            new_q = re.sub(r"\b" + re.escape(term) + r"\b", replacement, query, flags=re.IGNORECASE)
            if new_q.lower() != lower_q:
                expanded_variants.append(new_q)

    return expanded_variants[:2]


def rewrite_and_expand_query(
    query: str,
    memory: Optional[ConversationMemory] = None,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """
    Analyzes user query in context of memory:
    1. Standalone question resolution (de-contextualization).
    2. Generation of 2 alternative search phrasings.
    3. Injection of insurance domain synonyms.
    """
    client = llm_client or get_llm_client()
    history_context = memory.format_history_for_prompt(n=4) if memory else "No prior conversation."

    # If there is no history and query is already long and clear, fast-track
    has_history = bool(memory and len(memory.history) > 0)

    prompt = f"""Conversation History (last turns):
{history_context}

Follow-up Question: {query}

Instructions:
1. If the question refers to prior context (e.g. 'what about cataract?', 'is that covered?'), rewrite it as a standalone, specific question mentioning relevant insurance concepts. If already standalone, keep it clean and clear.
2. Provide 2 alternative phrasings targeting insurance policy clauses and IRDAI rules.
3. Identify relevant domain terms for expansion (e.g. PED, cashless, repudiation, sub-limit).
"""

    standalone_query = query
    alternative_queries = []
    detected_terms = []

    try:
        result = client.generate_json(
            prompt=prompt,
            system_instruction=REWRITE_SYSTEM_PROMPT,
            temperature=0.0,
        )
        if isinstance(result, dict):
            standalone_query = result.get("standalone_query", query) or query
            alternative_queries = result.get("alternative_queries", []) or []
            detected_terms = result.get("detected_domain_terms", []) or []
    except Exception as e:
        logger.warning("LLM query rewrite failed, falling back to rule-based expansion: %s", e)
        # Rule based standalone resolution if pronoun detected
        if has_history and any(w in query.lower().split() for w in ["it", "this", "that", "he", "she", "what about", "and"]):
            # Prepend context from last user turn
            recent_turns = memory.get_recent_turns(n=1)
            if recent_turns:
                standalone_query = f"{recent_turns[-1]['content']} - Specifically: {query}"
        alternative_queries = expand_synonyms_rule_based(standalone_query)

    # Ensure we also supplement with rule-based synonyms if missing
    rule_synonyms = expand_synonyms_rule_based(standalone_query)
    for s_q in rule_synonyms:
        if s_q not in alternative_queries and s_q != standalone_query:
            alternative_queries.append(s_q)

    # Clean duplicates
    all_variants = [q for q in [standalone_query] + alternative_queries if q.strip()]
    unique_variants = list(dict.fromkeys(all_variants))

    return {
        "original_query": query,
        "standalone_query": standalone_query,
        "alternative_queries": [v for v in unique_variants if v != standalone_query][:2],
        "all_queries": unique_variants,
        "detected_terms": detected_terms,
    }
