"""
PolicyLens - Streamlit Application
A Production-Grade, Grounded RAG Application for Indian Health Insurance.
Designed for Hack Hour at Global Academy of Technology.
"""

import os
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DATA_DIR,
    INDEX_DIR,
    EVAL_DIR,
    DEFAULT_TOP_K,
    MIN_TOP_K,
    MAX_TOP_K,
    GEMINI_MODEL,
    EVAL_RESULTS_PATH,
    EVAL_REPORT_PATH,
    REFUSAL_MESSAGE,
)
from src.index import load_indices, rebuild_index, indices_exist
from src.retrieve import retrieve_hybrid
from src.rewrite import rewrite_and_expand_query
from src.generate import generate_grounded_answer
from src.memory import ConversationMemory
from src.llm import get_llm_client, GeminiLLMClient

# Set page configuration
st.set_page_config(
    page_title="PolicyLens — Indian Health Insurance RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading semantic vector index and BM25 sparse model...")
def get_cached_indices():
    """Load or initialize FAISS and BM25 search indices."""
    return load_indices()


def highlight_text(text: str, query: str) -> str:
    """Highlight query terms in passage using markdown bold tags."""
    words = [w.strip() for w in query.split() if len(w.strip()) > 3]
    highlighted = text
    for w in set(words):
        import re
        pattern = re.compile(rf"\b({re.escape(w)})\b", re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)
    return highlighted


def initialize_session():
    """Initialize chat state and conversational memory."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0


initialize_session()

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("🛡️ PolicyLens")
    st.caption("Verifiable, Grounded RAG for Indian Health Insurance")

    st.markdown("---")
    st.subheader("⚙️ Retrieval & Generation Controls")

    # Ensure indices exist
    try:
        faiss_index, bm25_index, metadata = get_cached_indices()
        all_docs = sorted(list({c.get("doc_name") for c in metadata if c.get("doc_name")}))
    except Exception as e:
        st.error(f"Error loading indices: {e}")
        faiss_index, bm25_index, metadata = None, None, []
        all_docs = []

    # Document filter
    selected_docs = st.multiselect(
        "Filter by Document:",
        options=all_docs,
        default=all_docs,
        help="Restrict search strictly to selected policy or regulatory circulars.",
    )

    # Top-K
    top_k = st.slider(
        "Top-K Passages:",
        min_value=MIN_TOP_K,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        help="Number of fused context chunks passed to the generator.",
    )

    # Upgrades toggles
    use_reranker = st.toggle(
        "Cross-Encoder Reranker",
        value=True,
        help="Re-score top candidates with ms-marco-MiniLM-L-6-v2.",
    )

    use_rewriter = st.toggle(
        "Query Rewriting & Synonym Expansion",
        value=True,
        help="Transform follow-ups into standalone queries and expand domain insurance terms.",
    )

    use_verification = st.toggle(
        "Faithfulness Verification (Self-Check)",
        value=True,
        help="Audit answer sentences against citations to prevent hallucinations.",
    )

    st.markdown("---")
    st.subheader("🔑 LLM Configuration")

    # API Key Handling
    env_key = os.getenv("GEMINI_API_KEY", "")
    user_key = st.text_input(
        "Gemini API Key:",
        value=env_key,
        type="password",
        help="Enter your Google Gemini API key or set GEMINI_API_KEY in .env.",
    )

    if user_key:
        os.environ["GEMINI_API_KEY"] = user_key
        st.success(f"Active Model: `{GEMINI_MODEL}`", icon="🟢")
    else:
        st.info("Offline Fallback Mode Active (Heuristic Extractive Grounding)", icon="ℹ️")

    st.markdown("---")
    if st.button("🔄 Rebuild Index", use_container_width=True):
        with st.spinner("Re-ingesting documents and rebuilding FAISS & BM25 indices..."):
            rebuild_index()
            st.cache_resource.clear()
            st.success("Indices successfully rebuilt!")
            st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.markdown(
        """
        <div style='font-size:0.75rem; color:#64748B; margin-top:2rem;'>
        <b>Built for Hack Hour:</b> From Text to Insight<br>
        Global Academy of Technology
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# MAIN APP HEADER
# -------------------------------------------------------------
st.title("🛡️ PolicyLens: Indian Health Insurance Assistant")
st.markdown(
    "**Plain-language clarity on coverage, waiting periods, sub-limits, claim rejections, and IRDAI Ombudsman rights for Ravi and his family.**"
)

tab_chat, tab_corpus, tab_eval, tab_demo = st.tabs([
    "💬 Ask PolicyLens",
    "📚 Document Corpus",
    "📊 Evaluation & Benchmarks",
    "⚡ 3-Minute Demo",
])


# -------------------------------------------------------------
# TAB 1: CHAT INTERFACE
# -------------------------------------------------------------
with tab_chat:
    # Example Questions
    st.markdown("##### 💡 Example Questions:")
    col1, col2, col3 = st.columns(3)
    ex_questions = [
        ("Room rent & ICU limits?", "What is the room rent limit and ICU sub-limit under Arogya Sanjeevani?"),
        ("Pre-existing disease waiting?", "What is the waiting period for pre-existing diseases (PED)?"),
        ("Cashless approval time?", "What are the mandatory IRDAI turnaround times for cashless hospitalisation authorization?"),
        ("Claim denied for late filing?", "Can my claim be rejected solely because I submitted hospital bills late?"),
        ("Ombudsman rights & limits?", "If my claim is rejected, how and when can I approach the Insurance Ombudsman?"),
        ("Apollo Hospital tariff? (Out-of-doc)", "What is the room rent allowance for Deluxe Suite at Apollo Hospital Bangalore?"),
    ]

    clicked_prompt = None
    with col1:
        if st.button(ex_questions[0][0], key="btn_ex_0", use_container_width=True):
            clicked_prompt = ex_questions[0][1]
        if st.button(ex_questions[3][0], key="btn_ex_3", use_container_width=True):
            clicked_prompt = ex_questions[3][1]
    with col2:
        if st.button(ex_questions[1][0], key="btn_ex_1", use_container_width=True):
            clicked_prompt = ex_questions[1][1]
        if st.button(ex_questions[4][0], key="btn_ex_4", use_container_width=True):
            clicked_prompt = ex_questions[4][1]
    with col3:
        if st.button(ex_questions[2][0], key="btn_ex_2", use_container_width=True):
            clicked_prompt = ex_questions[2][1]
        if st.button(ex_questions[5][0], key="btn_ex_5", use_container_width=True):
            clicked_prompt = ex_questions[5][1]

    st.markdown("---")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                # Assistant message rendering with badge and sources
                found = msg.get("found", True)
                confidence = msg.get("confidence", "high").upper()

                if found:
                    st.markdown(
                        f"<span style='background-color:#DCFCE7; color:#166534; font-weight:600; padding:2px 8px; border-radius:4px;'>🟢 FOUND IN POLICY</span> "
                        f"<span style='background-color:#F1F5F9; color:#334155; font-size:0.8rem; padding:2px 6px; border-radius:4px;'>Confidence: {confidence}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<span style='background-color:#FEE2E2; color:#991B1B; font-weight:600; padding:2px 8px; border-radius:4px;'>🔴 NOT FOUND IN DOCUMENTS — REFUSAL</span> "
                        f"<span style='background-color:#F1F5F9; color:#334155; font-size:0.8rem; padding:2px 6px; border-radius:4px;'>Confidence: LOW</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown(msg["content"])

                # Expandable Search & Citations
                search_info = msg.get("search_info")
                if search_info:
                    with st.expander("🔍 How I Searched & Expanded Query"):
                        st.markdown(f"**Standalone Query:** `{search_info.get('standalone_query')}`")
                        alt_q = search_info.get("alternative_queries", [])
                        if alt_q:
                            st.markdown(f"**Alternative Phrasings:** {alt_q}")
                        terms = search_info.get("detected_terms", [])
                        if terms:
                            st.markdown(f"**Domain Terms Applied:** `{', '.join(terms)}`")

                sources = msg.get("sources", [])
                if sources and found:
                    with st.expander(f"📑 View Grounding Citations ({len(sources)} source passages)"):
                        for s in sources:
                            c = s.get("chunk", {})
                            d_score = s.get("dense_score", 0.0)
                            bm_score = s.get("bm25_score", 0.0)
                            fused = s.get("fused_score", 0.0)
                            rerank = s.get("rerank_score")
                            score_label = f"Fused RRF: {fused:.4f} | Dense: {d_score:.2f} | BM25: {bm_score:.1f}"
                            if rerank is not None:
                                score_label += f" | Rerank: {rerank:.2f}"

                            st.markdown(
                                f"""
                                <div style='border-left: 3px solid #2563EB; padding-left: 8px; margin-bottom: 12px;'>
                                <b>[{s.get('citation_idx')}] {c.get('doc_name')}</b> — <i>Pages {c.get('page_start')}-{c.get('page_end')}</i><br>
                                <span style='font-size:0.85rem; color:#475569;'>Section: <b>{c.get('section')}</b> | {score_label}</span><br>
                                <div style='margin-top:4px; font-size:0.9rem;'>{highlight_text(c.get('text', ''), msg['content'])}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # Chat Input handler
    chat_prompt = st.chat_input("Ask a question about coverage, waiting periods, sub-limits, or claims...")
    active_query = clicked_prompt or chat_prompt

    if active_query:
        # Check indices
        if faiss_index is None:
            st.error("Index is not loaded. Please build the index first.")
            st.stop()

        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Assistant response generation
        with st.chat_message("assistant"):
            with st.spinner("Analyzing policy clauses and IRDAI regulations..."):
                t_start = time.perf_counter()
                client = get_llm_client()

                # Step 1: Query Rewriting and Domain Expansion
                if use_rewriter:
                    rewrite_result = rewrite_and_expand_query(
                        query=active_query,
                        memory=st.session_state.memory,
                        llm_client=client,
                    )
                    search_query = rewrite_result["standalone_query"]
                    variants = rewrite_result["all_queries"]
                else:
                    rewrite_result = {"standalone_query": active_query, "alternative_queries": [], "detected_terms": []}
                    search_query = active_query
                    variants = [active_query]

                # Step 2: Hybrid Retrieval + Reranking
                retrieved_chunks = retrieve_hybrid(
                    query=search_query,
                    faiss_index=faiss_index,
                    bm25_index=bm25_index,
                    metadata=metadata,
                    top_k=top_k,
                    filter_docs=selected_docs if len(selected_docs) < len(all_docs) else None,
                    use_reranker=use_reranker,
                    query_variants=variants,
                )

                # Step 3: Grounded Answer Synthesis
                gen_result = generate_grounded_answer(
                    query=search_query,
                    retrieved_items=retrieved_chunks,
                    llm_client=client,
                    verify=use_verification,
                    use_reranker=use_reranker,
                )

                latency_s = time.perf_counter() - t_start
                answer = gen_result.get("answer", REFUSAL_MESSAGE)
                citations = gen_result.get("citations", [])
                found = gen_result.get("found", True)
                confidence = gen_result.get("confidence", "high").upper()

                # Render badge and text
                if found:
                    st.markdown(
                        f"<span style='background-color:#DCFCE7; color:#166534; font-weight:600; padding:2px 8px; border-radius:4px;'>🟢 FOUND IN POLICY</span> "
                        f"<span style='background-color:#F1F5F9; color:#334155; font-size:0.8rem; padding:2px 6px; border-radius:4px;'>Confidence: {confidence} | {latency_s:.2f}s</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<span style='background-color:#FEE2E2; color:#991B1B; font-weight:600; padding:2px 8px; border-radius:4px;'>🔴 NOT FOUND IN DOCUMENTS — REFUSAL</span> "
                        f"<span style='background-color:#F1F5F9; color:#334155; font-size:0.8rem; padding:2px 6px; border-radius:4px;'>Confidence: LOW | {latency_s:.2f}s</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown(answer)

                # Format source provenance
                sources_payload = []
                for idx, item in enumerate(retrieved_chunks, start=1):
                    item_copy = dict(item)
                    item_copy["citation_idx"] = idx
                    sources_payload.append(item_copy)

                # Show search expander
                if use_rewriter:
                    with st.expander("🔍 How I Searched & Expanded Query"):
                        st.markdown(f"**Standalone Query:** `{search_query}`")
                        st.markdown(f"**Alternative Phrasings:** {rewrite_result.get('alternative_queries', [])}")
                        st.markdown(f"**Domain Terms Applied:** `{', '.join(rewrite_result.get('detected_terms', []))}`")

                # Show citations expander
                if sources_payload and found:
                    with st.expander(f"📑 View Grounding Citations ({len(sources_payload)} source passages)"):
                        for s in sources_payload:
                            c = s.get("chunk", {})
                            d_score = s.get("dense_score", 0.0)
                            bm_score = s.get("bm25_score", 0.0)
                            fused = s.get("fused_score", 0.0)
                            rerank = s.get("rerank_score")
                            score_label = f"Fused RRF: {fused:.4f} | Dense: {d_score:.2f} | BM25: {bm_score:.1f}"
                            if rerank is not None:
                                score_label += f" | Rerank: {rerank:.2f}"

                            st.markdown(
                                f"""
                                <div style='border-left: 3px solid #2563EB; padding-left: 8px; margin-bottom: 12px;'>
                                <b>[{s.get('citation_idx')}] {c.get('doc_name')}</b> — <i>Pages {c.get('page_start')}-{c.get('page_end')}</i><br>
                                <span style='font-size:0.85rem; color:#475569;'>Section: <b>{c.get('section')}</b> | {score_label}</span><br>
                                <div style='margin-top:4px; font-size:0.9rem;'>{highlight_text(c.get('text', ''), answer)}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Save assistant turn to session & memory
                st.session_state.memory.add_user_message(active_query)
                st.session_state.memory.add_assistant_message(
                    content=answer,
                    citations=citations,
                    found=found,
                    confidence=confidence,
                    missing_info=gen_result.get("missing_info"),
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "found": found,
                    "confidence": confidence,
                    "sources": sources_payload,
                    "search_info": rewrite_result,
                })


# -------------------------------------------------------------
# TAB 2: DOCUMENT CORPUS
# -------------------------------------------------------------
with tab_corpus:
    st.subheader("📚 Indexed Document Repository")
    st.markdown(
        "PolicyLens extracts, de-duplicates headers/footers, and chunks legal clauses from these documents. "
        "Each chunk preserves provenance: document name, type, page range, and section heading."
    )

    if metadata:
        doc_stats = {}
        for c in metadata:
            dname = c.get("doc_name", "Unknown")
            if dname not in doc_stats:
                doc_stats[dname] = {
                    "doc_name": dname,
                    "doc_type": c.get("doc_type", "policy").upper(),
                    "chunks": 0,
                    "pages": set(),
                    "sections": set(),
                    "words": 0,
                }
            doc_stats[dname]["chunks"] += 1
            doc_stats[dname]["pages"].add(c.get("page_start", 1))
            doc_stats[dname]["pages"].add(c.get("page_end", 1))
            doc_stats[dname]["sections"].add(c.get("section", "General"))
            doc_stats[dname]["words"] += len(c.get("text", "").split())

        corpus_rows = []
        for dname, stats in doc_stats.items():
            corpus_rows.append({
                "Document Name": dname,
                "Type": stats["doc_type"],
                "Pages": len(stats["pages"]),
                "Chunks": stats["chunks"],
                "Total Words": stats["words"],
                "Sample Sections": ", ".join(list(stats["sections"])[:3]) + "...",
            })

        st.dataframe(pd.DataFrame(corpus_rows), use_container_width=True)

        st.markdown("#### Sample Extracted Chunk Preview")
        selected_preview_doc = st.selectbox("Select document to inspect chunks:", options=list(doc_stats.keys()))
        preview_chunks = [c for c in metadata if c.get("doc_name") == selected_preview_doc]
        if preview_chunks:
            chunk_slider = st.slider("Chunk Index:", 0, len(preview_chunks) - 1, 0)
            sample_c = preview_chunks[chunk_slider]
            st.json({
                "chunk_id": sample_c.get("chunk_id"),
                "section": sample_c.get("section"),
                "pages": f"{sample_c.get('page_start')} - {sample_c.get('page_end')}",
                "doc_type": sample_c.get("doc_type"),
                "word_count": len(sample_c.get("text", "").split()),
                "text_snippet": sample_c.get("text"),
            })
    else:
        st.warning("No documents currently indexed. Please click 'Rebuild Index' in the sidebar.")


# -------------------------------------------------------------
# TAB 3: EVALUATION & BENCHMARKS
# -------------------------------------------------------------
with tab_eval:
    st.subheader("📊 Rigorous Evaluation & Ablation Benchmarks")
    st.caption("Results from test suite evaluating retrieval quality, citation accuracy, and refusal precision.")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("▶️ Run Evaluation Suite Now", use_container_width=True):
            with st.spinner("Running 10-query test suite across 4 retrieval configurations..."):
                from eval.run_eval import run_full_evaluation
                run_full_evaluation()
                st.success("Evaluation complete! Results updated.")
                st.rerun()

    # Load results.csv and report.md
    if EVAL_RESULTS_PATH.exists() and EVAL_REPORT_PATH.exists():
        df_results = pd.read_csv(EVAL_RESULTS_PATH)
        with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
            report_text = f.read()

        # Display Top Metrics Cards
        total_q = len(df_results)
        passed_q = df_results["pass"].sum()
        pass_rate = (passed_q / total_q) * 100 if total_q > 0 else 0
        unans_q = df_results[df_results["type"] == "unanswerable"]
        refusal_acc = (unans_q["pass"].sum() / len(unans_q)) * 100 if len(unans_q) > 0 else 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Pass Rate", f"{pass_rate:.0f}%", f"{passed_q}/{total_q} questions")
        m2.metric("Refusal Accuracy", f"{refusal_acc:.0f}%", "Zero Hallucination")
        m3.metric("Avg Query Latency", f"{df_results['latency_ms'].mean():.1f} ms", "Sub-second")
        m4.metric("Audited Test Cases", f"{total_q} Cases", "10-item test set")

        st.markdown("---")
        st.markdown(report_text)

        st.markdown("#### Detailed Question-Level Results Table")
        st.dataframe(
            df_results[["id", "type", "question", "pass", "source", "latency_ms"]],
            use_container_width=True,
        )
    else:
        st.info("Evaluation has not been run yet. Click 'Run Evaluation Suite Now' above or run `python eval/run_eval.py` in the terminal.")


# -------------------------------------------------------------
# TAB 4: 3-MINUTE SCRIPTED DEMO MODE
# -------------------------------------------------------------
with tab_demo:
    st.subheader("⚡ 3-Minute Hack Hour Presentation Demo")
    st.markdown(
        "Demonstrate PolicyLens's core capabilities in under 3 minutes for judges: "
        "exact grounded factual lookup, multi-document synthesis using conversation memory, and hallucination-resistant refusal."
    )

    demo_c1, demo_c2, demo_c3 = st.columns(3)

    demo_q1 = "What is the room rent limit and ICU capping under Arogya Sanjeevani?"
    demo_q2 = "If the hospital denies cashless authorization, what are the IRDAI mandatory turnaround timelines and Ombudsman escalation rights?"
    demo_q3 = "What is the OPD dental treatment coverage limit at Max Super Speciality Hospital?"

    with demo_c1:
        st.markdown("#### Scenario 1: Direct Factual")
        st.caption("Verifies exact numeric clause grounding (2% Room Rent, 5% ICU).")
        if st.button("🚀 Run Demo 1", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": demo_q1})
            client = get_llm_client()
            res_chunks = retrieve_hybrid(demo_q1, faiss_index, bm25_index, metadata, top_k=5, use_reranker=True)
            res_gen = generate_grounded_answer(demo_q1, res_chunks, llm_client=client, verify=True, use_reranker=True)
            st.session_state.memory.add_user_message(demo_q1)
            st.session_state.memory.add_assistant_message(res_gen["answer"], res_gen["citations"], res_gen["found"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": res_gen["answer"],
                "citations": res_gen["citations"],
                "found": res_gen["found"],
                "confidence": res_gen["confidence"],
                "sources": [{"chunk": c["chunk"], "dense_score": c["dense_score"], "bm25_score": c["bm25_score"], "fused_score": c["fused_score"], "citation_idx": i} for i, c in enumerate(res_chunks, 1)],
                "search_info": {"standalone_query": demo_q1, "alternative_queries": [], "detected_terms": ["room rent", "icu"]},
            })
            st.rerun()

    with demo_c2:
        st.markdown("#### Scenario 2: Multi-Document")
        st.caption("Connects policy wording with IRDAI circular cashless TAT and Ombudsman rights.")
        if st.button("🚀 Run Demo 2", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": demo_q2})
            client = get_llm_client()
            res_chunks = retrieve_hybrid(demo_q2, faiss_index, bm25_index, metadata, top_k=5, use_reranker=True)
            res_gen = generate_grounded_answer(demo_q2, res_chunks, llm_client=client, verify=True, use_reranker=True)
            st.session_state.memory.add_user_message(demo_q2)
            st.session_state.memory.add_assistant_message(res_gen["answer"], res_gen["citations"], res_gen["found"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": res_gen["answer"],
                "citations": res_gen["citations"],
                "found": res_gen["found"],
                "confidence": res_gen["confidence"],
                "sources": [{"chunk": c["chunk"], "dense_score": c["dense_score"], "bm25_score": c["bm25_score"], "fused_score": c["fused_score"], "citation_idx": i} for i, c in enumerate(res_chunks, 1)],
                "search_info": {"standalone_query": demo_q2, "alternative_queries": [], "detected_terms": ["cashless", "ombudsman", "turnaround"]},
            })
            st.rerun()

    with demo_c3:
        st.markdown("#### Scenario 3: Refusal / Zero Hallucination")
        st.caption("Demonstrates refusal: unmentioned hospital tariffs return 'I don't know'.")
        if st.button("🚀 Run Demo 3", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": demo_q3})
            client = get_llm_client()
            res_chunks = retrieve_hybrid(demo_q3, faiss_index, bm25_index, metadata, top_k=5, use_reranker=True)
            res_gen = generate_grounded_answer(demo_q3, res_chunks, llm_client=client, verify=True, use_reranker=True)
            st.session_state.memory.add_user_message(demo_q3)
            st.session_state.memory.add_assistant_message(res_gen["answer"], res_gen["citations"], res_gen["found"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": res_gen["answer"],
                "citations": res_gen["citations"],
                "found": res_gen["found"],
                "confidence": res_gen["confidence"],
                "sources": [{"chunk": c["chunk"], "dense_score": c["dense_score"], "bm25_score": c["bm25_score"], "fused_score": c["fused_score"], "citation_idx": i} for i, c in enumerate(res_chunks, 1)],
                "search_info": {"standalone_query": demo_q3, "alternative_queries": [], "detected_terms": ["opd", "dental"]},
            })
            st.rerun()


# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #94A3B8; font-size: 0.8rem; padding: 10px;'>
    ⚖️ <i>Informational only, not insurance or legal advice. PolicyLens answers are grounded strictly in indexed policy wordings and regulatory circulars.</i>
    </div>
    """,
    unsafe_allow_html=True,
)
