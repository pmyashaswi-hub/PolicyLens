# PolicyLens Data Sources Directory

Drop your health insurance policy documents and IRDAI circulars/regulations here in PDF format.
When you add or replace PDFs, click the **"Rebuild index"** button in the PolicyLens Streamlit sidebar (or run `python -m src.index --rebuild`).

---

## Indexed Document Catalog Template

| Document File Name | Issuer / Organization | Doc Type | Effective Year | Key Clauses / Subject Matter |
| :--- | :--- | :--- | :--- | :--- |
| `arogya_sanjeevani_policy_wordings_2024.pdf` | National Insurance / IRDAI Standard Health Product | Policy | 2024 | Standard coverage, room rent limits (2%), ICU (5%), 30-day initial waiting period, 24/48-month specified diseases, 36-month PED, 5% co-pay |
| `irdai_master_circular_claims_grievances_2024.pdf` | Insurance Regulatory and Development Authority of India (IRDAI) | Regulation | 2024 | Mandatory cashless TAT (1 hr cashless pre-auth, 3 hr final discharge), 30-day claim settlement, penal interest for delays (bank rate + 2%), Ombudsman jurisdiction (Rs 50 Lakhs) |
| `care_plus_critical_illness_rider_2023.pdf` | Care Health Insurance | Policy (Rider) | 2023 | 15 critical illnesses coverage, 30-day survival period, 90-day initial waiting period, specific major organ exclusions |

---

## Instructions for Adding New Documents
1. Ensure the PDF contains readable selectable text (if scanned, apply OCR first).
2. Follow descriptive naming conventions such as `issuer_policyname_year.pdf` or `irdai_circular_subject_year.pdf`.
3. PolicyLens automatically categorizes files with `irdai` or `regulation` in their title as **Regulation**, and other contracts as **Policy**.
4. After adding PDFs, trigger index rebuild via:
   ```bash
   python -m src.index --rebuild
   ```
