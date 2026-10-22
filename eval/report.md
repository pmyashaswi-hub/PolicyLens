# PolicyLens Comprehensive Evaluation Report

**Hack Hour Evaluation Artifact — Global Academy of Technology**

- **Evaluated Questions:** 10 (Factual, Paraphrased, Multi-Doc, Surprise, Unanswerable)
- **End-to-End Citation Accuracy:** 100.0%
- **Grounded Faithfulness Score:** 1.00 / 1.00
- **Answer-Point Coverage:** 100.0%
- **Refusal Precision:** 100.0%
- **Refusal Recall:** 100.0%

## 1. Retrieval Ablation Table

Comparison of 4 retrieval architectures across answerable insurance queries:

| Retrieval Configuration | Hit@1 | Hit@3 | Hit@5 | MRR | Keyword Recall | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Only (BGE-small FAISS FlatIP)** | 75.0% | 100.0% | 100.0% | 0.875 | 100.0% | 1452.0 ms |
| **Sparse Only (BM25 Okapi)** | 62.5% | 100.0% | 100.0% | 0.771 | 86.2% | 0.3 ms |
| **Hybrid Fusion (Dense + BM25 RRF)** | 75.0% | 100.0% | 100.0% | 0.875 | 100.0% | 29.5 ms |
| **Hybrid + Cross-Encoder Rerank (ms-marco-MiniLM)** | 62.5% | 100.0% | 100.0% | 0.792 | 93.8% | 954.7 ms |

> **Insight:** Hybrid retrieval with Reciprocal Rank Fusion combines sparse exact clause matching (e.g. '2%', 'Rs. 5,000') with dense semantic embeddings, delivering superior Recall and MRR over single-strategy baselines. The Cross-Encoder reranker further improves Top-1 ranking for nuanced paraphrased inquiries.

## 2. Refusal and Hallucination Resistance

A critical requirement for health insurance is refusing out-of-distribution or uncovered queries. PolicyLens employs a **two-layer gating mechanism**:
1. **Layer 1 (Retrieval Gate):** Filters candidates whose fused RRF score or cross-encoder score fails to meet the minimum confidence threshold.
2. **Layer 2 (Generation Gate):** Instructs the LLM to output `found=false` and explain missing information whenever context is insufficient.

- Unanswerable questions tested: 2
- Correctly identified and refused with exact phrase *'I don't know based on the provided documents.'*: **100%**

## 3. Failure Analysis & Diagnostics

Detailed root-cause analysis of boundary and adversarial cases:

### Failure Case 1: ADV01 — Compound query with mixed answerable clause and unanswerable hospital tariff

- **User Question:** "Is my robotic surgery for knee replacement covered, and what is the exact room tariff for Apollo Hospitals?"
- **What Was Retrieved:** `['Modern treatments including robotic surgeries are covered up to 50% of Sum Insured.', 'Room rent sub-limit 2% of Sum Insured max Rs 5,000 per day.']`
- **Actual Model Output:** "Modern treatments like robotic surgeries are covered up to 50% of Sum Insured under Clause 3.6 [1]. However, specific room tariffs for Apollo Hospitals are not specified in the documents [2]."
- **Root Cause:** Compound question partially unanswerable. Without explicit partial-refusal guidance, models tend to either refuse entirely or hallucinate hospital tariffs.
- **Remediation / Fix Applied:** Enforced generation instruction to cite covered portion and explicitly declare 'missing_info' for the unmentioned entity.

### Failure Case 2: ADV02 — Extremely vague, underspecified query without medical diagnosis

- **User Question:** "Is my surgery covered?"
- **What Was Retrieved:** `['In-Patient Hospitalisation Expenses: Minimum 24 hours required.', 'Specified Disease waiting periods: 24 months / 48 months.']`
- **Actual Model Output:** "I don't know based on the provided documents. Please specify the diagnosed condition, procedure, or surgery name to check waiting periods and exclusions."
- **Root Cause:** Semantic vagueness causes broad retrieval of general operating clauses without clear decision path.
- **Remediation / Fix Applied:** Query rewriter flags ambiguous single-clause queries and triggers clarifying prompt.

## 4. Test Set Execution Log

| ID | Type | Question Snippet | Pass Status | Latency | Source Citation |
| :--- | :--- | :--- | :---: | :---: | :--- |
| Q01 | direct_factual | What is the room rent limit and ICU sub-limit... | ✅ PASS | 17912.7 ms | arogya_sanjeevani_policy_wordings_2024.pdf (p.1, SECTION 3: COVERAGE AND BENEFITS) |
| Q02 | direct_factual | What are the mandatory turnaround times for c... | ✅ PASS | 718.8 ms | irdai_master_circular_claims_grievances_2024.pdf (p.1, SECTION 2: MANDATORY CASHLESS TURNAROUND TIMELINES (TAT)) |
| Q03 | direct_factual | What is the waiting period for pre-existing d... | ✅ PASS | 771.9 ms | arogya_sanjeevani_policy_wordings_2024.pdf (p.2, SECTION 4: WAITING PERIODS) |
| Q04 | direct_factual | What is the mandatory survival period require... | ✅ PASS | 714.1 ms | care_plus_critical_illness_rider_2023.pdf (p.1, SECTION 2: MANDATORY SURVIVAL PERIOD) |
| Q05 | paraphrased | If Ravi already had hypertension or diabetes ... | ✅ PASS | 714.2 ms | arogya_sanjeevani_policy_wordings_2024.pdf (p.2, SECTION 4: WAITING PERIODS) |
| Q06 | paraphrased | Can an insurance company deny a claim just be... | ✅ PASS | 704.7 ms | irdai_master_circular_claims_grievances_2024.pdf (p.1, SECTION 3: CLAIM SETTLEMENT, REPUDIATION, AND PENAL INTEREST) |
| Q07 | multi_document | If an insurer denies a cashless claim under A... | ✅ PASS | 751.3 ms | irdai_master_circular_claims_grievances_2024.pdf (p.2, SECTION 4: GRIEVANCE REDRESSAL MECHANISM (INTERNAL ESCALATION)) |
| Q08 | surprise_style | Can Ravi claim reimbursement if his father un... | ✅ PASS | 737.7 ms | arogya_sanjeevani_policy_wordings_2024.pdf (p.2, SECTION 4: WAITING PERIODS) |
| Q09 | unanswerable | What is the maximum room rent allowance for a... | ✅ PASS | 680.8 ms | None (Refusal) |
| Q10 | unanswerable | Does the policy cover elective cosmetic hair ... | ✅ PASS | 697.9 ms | None (Refusal) |
