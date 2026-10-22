"""
PolicyLens Sample Documents Generator
Generates realistic Indian Health Insurance policy wordings, IRDAI master circulars,
and critical illness riders in PDF format using ReportLab.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_sample_documents(output_dir: Path) -> None:
    """Generate 3 authentic sample PDFs for Indian Health Insurance domain."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    output_dir.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=18
    )
    heading_style = ParagraphStyle(
        'DocHeading',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1E40AF'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=8
    )
    clause_style = ParagraphStyle(
        'DocClause',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#0F172A'),
        leftIndent=12,
        spaceAfter=6
    )

    # -------------------------------------------------------------
    # 1. Arogya Sanjeevani Policy Wordings 2024
    # -------------------------------------------------------------
    pdf_path_1 = output_dir / "arogya_sanjeevani_policy_wordings_2024.pdf"
    doc1 = SimpleDocTemplate(str(pdf_path_1), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story1 = []

    story1.append(Paragraph("AROGYA SANJEEVANI POLICY - STANDARD HEALTH PRODUCT", title_style))
    story1.append(Paragraph("Issuer: National Insurance Company Ltd. | Policy Year: 2024 | UIN: NICHLIP21145V022024", subtitle_style))
    story1.append(Spacer(1, 10))

    story1.append(Paragraph("SECTION 1: PREAMBLE AND OPERATING CLAUSE", heading_style))
    story1.append(Paragraph(
        "This Policy is a contract of insurance issued to the Policyholder named in the Schedule based on the proposal form, "
        "medical declarations, and premium paid. The Company agrees to indemnify the Insured Person for medically necessary "
        "hospitalisation expenses incurred during the policy period, subject strictly to the limits, waiting periods, sub-limits, "
        "co-payment, and exclusions set forth herein.", body_style
    ))

    story1.append(Paragraph("SECTION 2: DEFINITIONS AND KEY TERMINOLOGY", heading_style))
    story1.append(Paragraph("<b>2.1 Pre-Existing Disease (PED):</b> Any condition, ailment, injury or disease that is diagnosed by a physician within 36 months prior to the effective date of the policy issued by the insurer, or for which medical advice or treatment was recommended by, or received from, a physician within 36 months prior to the effective date.", clause_style))
    story1.append(Paragraph("<b>2.2 Cashless Facility:</b> A facility extended by the insurer through network hospitals where the medical expenses incurred by the insured are directly settled by the insurer or TPA in accordance with the agreed terms.", clause_style))
    story1.append(Paragraph("<b>2.3 Co-payment:</b> A cost-sharing requirement under an insurance policy that provides that the policyholder/insured will bear a specified percentage of the admissible claim amount.", clause_style))
    story1.append(Paragraph("<b>2.4 Day Care Treatment:</b> Medical treatment and surgical procedures undertaken under general or local anesthesia in a hospital/day care centre in less than 24 hours because of technological advancement.", clause_style))

    story1.append(Paragraph("SECTION 3: COVERAGE AND BENEFITS", heading_style))
    story1.append(Paragraph("<b>3.1 In-Patient Hospitalisation Expenses:</b> Hospitalisation expenses for a minimum period of 24 consecutive hours for room rent, nursing, boarding, physician fees, and medicines.", clause_style))
    story1.append(Paragraph("<b>3.2 Room Rent Sub-Limit:</b> Room, Boarding, and Nursing Expenses are covered up to 2% of the Sum Insured, subject to a strict maximum ceiling of Rs. 5,000 per day. Any excess room tariff chosen results in proportionate deduction across associated medical charges.", clause_style))
    story1.append(Paragraph("<b>3.3 Intensive Care Unit (ICU) Sub-Limit:</b> Intensive Care Unit (ICU) and Intensive Cardiac Care Unit (ICCU) expenses are covered up to 5% of the Sum Insured, subject to a maximum ceiling of Rs. 10,000 per day.", clause_style))
    story1.append(Paragraph("<b>3.4 Pre and Post-Hospitalisation:</b> Pre-hospitalisation medical expenses incurred up to 30 days prior to hospital admission and Post-hospitalisation medical expenses incurred up to 60 days following discharge are admissible.", clause_style))
    story1.append(Paragraph("<b>3.5 Cataract Treatment Sub-Limit:</b> Coverage for cataract surgery is capped at 25% of the Sum Insured or Rs. 40,000, whichever is lower, per eye during each policy year.", clause_style))
    story1.append(Paragraph("<b>3.6 AYUSH Hospitalisation:</b> Expenses for Ayurveda, Yoga and Naturopathy, Unani, Siddha, and Homeopathy systems of medicine are covered up to 100% of the Sum Insured in Government recognized AYUSH hospitals.", clause_style))

    story1.append(PageBreak())

    story1.append(Paragraph("SECTION 4: WAITING PERIODS", heading_style))
    story1.append(Paragraph("<b>4.1 Initial Waiting Period (30 Days):</b> Expenses related to the treatment of any illness within 30 days from the first policy inception date shall be excluded, EXCEPT in case of hospitalisation necessitated by an unexpected Accident. Coverage for accidental injury commences from Day 1.", clause_style))
    story1.append(Paragraph("<b>4.2 Specified Disease / Procedure Waiting Period (24 / 48 Months):</b> A mandatory waiting period of 24 months applies to treatment of Benign Prostatic Hypertrophy, Hernia, Hydrocele, Fistula, Fissure in ano, Piles, Sinusitis, Calculus diseases of gall bladder and urogenital system, and Cataract. A waiting period of 48 months applies to Joint Replacement surgery unless necessitated by an accident.", clause_style))
    story1.append(Paragraph("<b>4.3 Pre-Existing Disease (PED) Waiting Period:</b> A waiting period of 36 months (3 years) of continuous coverage applies to any Pre-Existing Disease (PED) and its direct complications. No claim arising out of PED will be admissible during this 36-month period.", clause_style))

    story1.append(Paragraph("SECTION 5: GENERAL EXCLUSIONS", heading_style))
    story1.append(Paragraph("<b>5.1 Investigation and Diagnostic Observation:</b> Admission primarily for investigation, diagnostic observation, or general evaluation without active in-patient medical treatment is excluded.", clause_style))
    story1.append(Paragraph("<b>5.2 Rest Cure and Rehabilitation:</b> Custodial care, rest cure, bed rest, convalescence, or rehabilitation services are not covered.", clause_style))
    story1.append(Paragraph("<b>5.3 Cosmetic and Plastic Surgery:</b> Expenses for aesthetic treatment, cosmetic surgery, and hair transplant procedures are completely excluded, except when necessitated by burn trauma reconstruction or reconstructive surgery following cancer surgery.", clause_style))
    story1.append(Paragraph("<b>5.4 Maternity and Childbirth:</b> Maternity expenses, pregnancy, childbirth, caesarean section, and infertility procedures are completely excluded under the standard Arogya Sanjeevani wording.", clause_style))
    story1.append(Paragraph("<b>5.5 Domiciliary Hospitalisation and OPD:</b> Out-patient department (OPD) consultations and domiciliary hospitalisation are not covered.", clause_style))

    story1.append(Paragraph("SECTION 6: CO-PAYMENT CLAUSE", heading_style))
    story1.append(Paragraph("<b>6.1 Mandatory Co-payment:</b> A mandatory co-payment of 5% applies to each and every admissible claim under this policy. The Company will indemnify 95% of the admissible claim amount, and the policyholder/insured shall pay the remaining 5%.", clause_style))

    story1.append(Paragraph("SECTION 7: CLAIM PROCEDURE AND TIMELINES", heading_style))
    story1.append(Paragraph("<b>7.1 Cashless Intimation:</b> For planned hospitalisation, cashless pre-authorisation request must be submitted at least 48 hours prior to admission. For emergency hospitalisation, intimation must be given within 24 hours of hospital admission.", clause_style))
    story1.append(Paragraph("<b>7.2 Reimbursement Claim Submission:</b> Complete claim documents including discharge summary, original medical bills, payment receipts, prescriptions, and investigation reports must be submitted to the Company or TPA within 30 days of discharge from the hospital.", clause_style))

    doc1.build(story1)
    logger.info("Generated sample document: %s", pdf_path_1)

    # -------------------------------------------------------------
    # 2. IRDAI Master Circular on Claims & Grievances 2024
    # -------------------------------------------------------------
    pdf_path_2 = output_dir / "irdai_master_circular_claims_grievances_2024.pdf"
    doc2 = SimpleDocTemplate(str(pdf_path_2), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story2 = []

    story2.append(Paragraph("IRDAI MASTER CIRCULAR ON HEALTH INSURANCE OPERATIONS & CLAIMS", title_style))
    story2.append(Paragraph("Issuer: Insurance Regulatory and Development Authority of India (IRDAI) | Year: 2024 | Circular Ref: IRDAI/HLT/REG/CIR/084/05/2024", subtitle_style))
    story2.append(Spacer(1, 10))

    story2.append(Paragraph("SECTION 1: STATUTORY AUTHORITY AND PURPOSE", heading_style))
    story2.append(Paragraph(
        "Issued under powers conferred by Section 14(1) of the IRDAI Act 1999 and Section 34 of the Insurance Act 1938. "
        "This Master Circular lays down mandatory operational service standards, cashless turnaround timelines, claim settlement "
        "norms, grievance redressal procedures, and consumer protection guidelines binding on all General and Standalone Health Insurers.", body_style
    ))

    story2.append(Paragraph("SECTION 2: MANDATORY CASHLESS TURNAROUND TIMELINES (TAT)", heading_style))
    story2.append(Paragraph("<b>2.1 Pre-Authorisation Turnaround (1 Hour):</b> Every insurer and Third Party Administrator (TPA) shall decide and communicate the approval or rejection of initial cashless pre-authorisation within a strict maximum turnaround time of 1 hour (60 minutes) upon receipt of complete request from the network hospital.", clause_style))
    story2.append(Paragraph("<b>2.2 Final Discharge Cashless Authorisation (3 Hours):</b> Insurers and TPAs shall communicate the final cashless discharge approval within a maximum turnaround time of 3 hours (180 minutes) from receipt of the final discharge summary and billing from the network hospital. In no circumstances shall an admitted patient be detained in hospital pending insurer authorization beyond 3 hours. Any hospital retention charges arising due to delay beyond 3 hours shall be paid solely by the insurer.", clause_style))

    story2.append(Paragraph("SECTION 3: CLAIM SETTLEMENT, REPUDIATION, AND PENAL INTEREST", heading_style))
    story2.append(Paragraph("<b>3.1 30-Day Settlement Deadline:</b> An insurer shall settle or repudiate (reject) a health insurance claim within 30 days from the date of receipt of the last necessary document required for processing.", clause_style))
    story2.append(Paragraph("<b>3.2 Mandatory Penal Interest for Settlement Delays:</b> Where a claim settlement is delayed beyond 30 days without lawful justification, the insurer shall be liable to pay penal interest to the policyholder at a rate 2% above the prevailing Bank Rate published by the Reserve Bank of India (RBI). Penal interest runs from the expiry of the 30-day period until the actual date of payment.", clause_style))
    story2.append(Paragraph("<b>3.3 Prohibition of Repudiation on Pure Delay:</b> Insurers and TPAs shall not repudiate or reject any claim solely on grounds of delay in claim intimation or delayed submission of documents, provided the claimant or their family proves that the delay was due to genuine medical emergency, incapacity, or other unavoidable hardship.", clause_style))
    story2.append(Paragraph("<b>3.4 Detailed Repudiation Letter:</b> In the event of claim rejection or repudiation, the insurer must issue a clear, written repudiation letter explaining the specific reasons and citing the exact clause number of the policy wording in plain language. Vague or boilerplate rejection statements are strictly illegal.", clause_style))

    story2.append(PageBreak())

    story2.append(Paragraph("SECTION 4: GRIEVANCE REDRESSAL MECHANISM (INTERNAL ESCALATION)", heading_style))
    story2.append(Paragraph("<b>4.1 Grievance Redressal Officer (GRO):</b> Every insurer shall appoint a dedicated Grievance Redressal Officer (GRO) at each regional and head office. Contact details of the GRO must be displayed on the insurer's website and policy schedule.", clause_style))
    story2.append(Paragraph("<b>4.2 Resolution Timelines (14 Days):</b> The insurer shall acknowledge receipt of any consumer grievance within 3 working days and provide a final resolution or written decision within 14 calendar days.", clause_style))
    story2.append(Paragraph("<b>4.3 IRDAI Bima Bharosa Portal:</b> If the insurer fails to resolve the complaint within 14 days or provides an unsatisfactory reply, the policyholder may escalate the complaint directly to IRDAI via the Bima Bharosa Portal (bimabharosa.irdai.gov.in) or call the IRDAI Grievance Call Centre toll-free at 155255 / 1800 4254 732.", clause_style))

    story2.append(Paragraph("SECTION 5: INSURANCE OMBUDSMAN (BIMA LOKPAL) POWERS AND PROCEDURES", heading_style))
    story2.append(Paragraph("<b>5.1 Right to Approach Ombudsman:</b> A policyholder has the statutory right to file a complaint before the Insurance Ombudsman under the Insurance Ombudsman Rules 2017 if: (a) the insurer has rejected the claim in whole or in part, (b) the insurer has not replied within 30 days of grievance lodging, or (c) the resolution provided by the insurer is unsatisfactory.", clause_style))
    story2.append(Paragraph("<b>5.2 Limitation Period (1 Year):</b> The complaint to the Insurance Ombudsman must be lodged within 1 year from the date of receipt of the insurer's rejection letter or final decision.", clause_style))
    story2.append(Paragraph("<b>5.3 Pecuniary Jurisdiction Limit (Rs. 50 Lakhs):</b> The Insurance Ombudsman has legal authority and jurisdiction to adjudicate health insurance claims and disputes where the compensation or claim value claimed does not exceed Rs. 50 Lakhs (Rupees Fifty Lakhs).", clause_style))
    story2.append(Paragraph("<b>5.4 Binding Nature of Ombudsman Award:</b> The Insurance Ombudsman process is completely free of charge for policyholders (no fees or advocate required). The Ombudsman award is legally binding on the insurance company, and the insurer must comply with the award and make payment within 30 days of receiving the award copy.", clause_style))

    doc2.build(story2)
    logger.info("Generated sample document: %s", pdf_path_2)

    # -------------------------------------------------------------
    # 3. Care Plus Critical Illness Rider 2023
    # -------------------------------------------------------------
    pdf_path_3 = output_dir / "care_plus_critical_illness_rider_2023.pdf"
    doc3 = SimpleDocTemplate(str(pdf_path_3), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story3 = []

    story3.append(Paragraph("CARE PLUS CRITICAL ILLNESS RIDER - POLICY SPECIFICATION", title_style))
    story3.append(Paragraph("Issuer: Care Health Insurance Ltd. | Rider Version: 2023 | UIN: CHIHLIA20021V012023", subtitle_style))
    story3.append(Spacer(1, 10))

    story3.append(Paragraph("SECTION 1: BENEFIT AND COVERED CONDITIONS", heading_style))
    story3.append(Paragraph(
        "Upon first clinical diagnosis of any of the 15 specified Critical Illnesses during the policy term, the Company shall pay "
        "a lump sum benefit equal to 100% of the Critical Illness Rider Sum Insured. Covered illnesses include: Major Cancer of specified "
        "severity, First Heart Attack of specified severity, Open Chest CABG, Kidney Failure requiring regular dialysis, Stroke resulting "
        "in permanent neurological deficit, and Major Organ or Bone Marrow Transplant.", body_style
    ))

    story3.append(Paragraph("SECTION 2: MANDATORY SURVIVAL PERIOD", heading_style))
    story3.append(Paragraph("<b>2.1 30-Day Survival Period:</b> The Insured Person must survive for a minimum period of 30 consecutive calendar days from the date of first diagnosis of the critical illness or surgical procedure before the lump sum benefit is payable. If the insured person passes away within 30 days of diagnosis, no rider benefit is payable.", clause_style))

    story3.append(Paragraph("SECTION 3: WAITING PERIODS FOR CRITICAL ILLNESS", heading_style))
    story3.append(Paragraph("<b>3.1 90-Day Initial Waiting Period:</b> A mandatory initial waiting period of 90 days applies from the rider inception date. If any critical illness signs, symptoms, or diagnosis occur within 90 days of cover commencement, the rider benefit shall not be payable.", clause_style))
    story3.append(Paragraph("<b>3.2 Pre-Existing Critical Illness Exclusion:</b> Pre-existing critical conditions or illnesses declared at proposal are excluded for a continuous period of 36 months.", clause_style))

    story3.append(Paragraph("SECTION 4: SPECIFIC EXCLUSIONS", heading_style))
    story3.append(Paragraph("<b>4.1 Excluded Conditions:</b> No benefit shall be paid for: (a) non-invasive cancers, carcinoma-in-situ, or early stage tumours, (b) critical illnesses arising directly or indirectly from alcohol dependence, narcotics, or substance abuse, (c) self-inflicted injuries or suicide attempts, (d) conditions caused by nuclear contamination or war.", clause_style))

    story3.append(Paragraph("SECTION 5: CLAIMS AND VERIFICATION PROCEDURE", heading_style))
    story3.append(Paragraph("<b>5.1 Claim Notification:</b> The Company must be notified in writing within 15 days of diagnosis.", clause_style))
    story3.append(Paragraph("<b>5.2 Proof of Claim:</b> Documents required include: official claim form, attending physician's clinical summary, histopathology and biopsy reports, MRI/CT scans, and hospital discharge summaries confirming permanent functional impairment.", clause_style))

    doc3.build(story3)
    logger.info("Generated sample document: %s", pdf_path_3)
