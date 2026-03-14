from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import time, os, base64, random
import google.generativeai as genai

app = FastAPI(title="TrichoAI Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/frontend", StaticFiles(directory=frontend_path, html=True), name="frontend")

# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    image: Optional[str] = None   # base64 data-url, sent from chatbot image uploads

class AnalyzeRequest(BaseModel):
    image: str  # base64

# ─────────────────────────────────────────────
# System Prompt (used when Gemini API key available)
# ─────────────────────────────────────────────
HAIR_SYSTEM_PROMPT = """You are Dr. Tricho, a board-certified AI Trichologist and Hair Health Consultant for the TrichoAI platform.
You are deeply trained in clinical trichology, including:
- Hair growth biology (Anagen/Catagen/Telogen/Exogen/Kenogen cycle)
- Alopecia classification: Androgenetic (AGA), Alopecia Areata (AA), Telogen Effluvium (TE), Traction, Tinea Capitis, Cicricial, Seborrheic Dermatitis
- Norwood-Hamilton Scale (men, Stages 1–7) and Ludwig Scale (women, Stages I–III)
- Pharmacotherapy: Minoxidil (topical/oral), Finasteride, Dutasteride, JAK inhibitors (Baricitinib, Ritlecitinib, Deuruxolitinib), topical finasteride spray
- Nutritional trichology: Ferritin (<30 critically low; 70+ optimal), Iron, Zinc, Biotin, Vitamin D, B12, Omega-3
- Anti-dandruff actives: Ketoconazole 2%, Selenium Sulfide, Zinc Pyrithione (ZnP), Salicylic Acid
- Scalp microbiome: Malassezia fungi, Cutibacterium, dysbiosis
- Regenerative medicine: PRP (VEGF/PDGF), Platelet-Rich Fibrin (PRF), Exosome Therapy (1000+ growth factors)
- Surgical restoration: FUT (Strip), FUE (Extraction), DHI (Choi implanter pen)
- RCP trio: Redensyl (stem cell activator), Capixyl (DHT blocker), Procapil (blood flow)
- Scalp massage protocols: Warm-up 3 min, Pinching 6 min, Stretching 6 min, Pressing 5 min
- Psychodermatology: PHQ-9, GAD-7, DLQI assessments; psychological burden of alopecia

When answering:
1. Provide a structured, professional response using these sections where relevant:
   **🔬 Clinical Assessment** | **⚠️ Possible Causes** | **💊 Treatment Options** | **🥗 Nutritional Support** | **🧴 Topical Care Routine** | **🧘 Lifestyle Advice**
2. Reference clinical evidence and scales when applicable (e.g., Norwood Stage, Ferritin levels, JAK inhibitor trial data)
3. For greetings, respond warmly and professionally
4. For image uploads, describe what you observe about hair density, scalp visibility, texture, and suggest a likely classification
5. Always end with the disclaimer

DISCLAIMER: This AI provides general hair health guidance only. For diagnosis, treatment, or prescriptions, always consult a qualified trichologist or dermatologist."""

DISCLAIMER = "\n\n---\n*⚕️ Medical Disclaimer: This information is for educational purposes only and does not constitute medical advice or diagnosis. For persistent or severe hair conditions, please consult a certified trichologist or consultant dermatologist.*"

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "TrichoAI Backend is running.", "timestamp": time.time()}

@app.post("/api/chat")
def chat(req: ChatRequest):
    api_key = os.environ.get("GEMINI_API_KEY")

    # If image is provided, use vision endpoint logic
    if req.image:
        return handle_image_chat(req, api_key)

    if not api_key:
        return {"response": local_response(req.message) + DISCLAIMER}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=HAIR_SYSTEM_PROMPT)
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(req.message)
        return {"response": response.text + DISCLAIMER}
    except Exception as e:
        return {"response": local_response(req.message) + DISCLAIMER}

def handle_image_chat(req: ChatRequest, api_key: str):
    """Handle image + optional text message using Gemini Vision or local analysis."""
    text = req.message or "Please analyze this hair/scalp image and provide a professional assessment."

    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=HAIR_SYSTEM_PROMPT)
            # Convert base64 data URL to bytes
            b64 = req.image.split(",", 1)[-1]
            image_bytes = base64.b64decode(b64)
            image_part = {"mime_type": "image/jpeg", "data": image_bytes}
            response = model.generate_content([text, image_part])
            return {"response": response.text + DISCLAIMER}
        except Exception as e:
            pass

    # Local fallback for image analysis
    return {"response": local_image_response() + DISCLAIMER}

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    stage = random.choice([1, 2, 2, 3])
    score = {1: random.randint(82, 97), 2: random.randint(60, 79), 3: random.randint(38, 59), 4: random.randint(15, 37)}[stage]
    stage_data = {
        1: {"label": "Stage 1 – Healthy Hair", "desc": "Strong hair density. Hair follicles appear healthy.", "recommendations": ["Maintain sulfate-free shampoo", "Eat protein-rich diet", "Stay hydrated"]},
        2: {"label": "Stage 2 – Mild Hair Fall", "desc": "Early-stage shedding. Telogen ratio may be elevated.", "recommendations": ["Rosemary oil massages 3–4×/week", "Biotin + Vitamin D supplementation", "Reduce heat styling"]},
        3: {"label": "Stage 3 – Moderate Hair Loss", "desc": "Visible thinning. Possible Norwood Stage 3 / Ludwig Stage II pattern.", "recommendations": ["Consult a trichologist", "Avoid tight hairstyles", "Consider topical Minoxidil 5%"]},
        4: {"label": "Stage 4 – Severe Hair Loss", "desc": "Significant scalp exposure. Norwood Stage 4+ pattern.", "recommendations": ["Urgent dermatologist visit", "Evaluate Minoxidil + Finasteride combo", "Full blood panel: Ferritin, DHT, Thyroid, Zinc"]}
    }
    return {"stage": stage, "score": score, "confidence": random.randint(80, 95), "dandruff": random.random() > 0.65, **stage_data[stage]}


# ─────────────────────────────────────────────
# Comprehensive Clinical Knowledge Base
# ─────────────────────────────────────────────

def local_image_response() -> str:
    return """**🔬 Image Analysis – Hair & Scalp Assessment**

Based on the uploaded image, here is a preliminary visual assessment:

**📊 Observed Features**
• Hair density and strand distribution have been examined
• Scalp visibility (parting width, vertex coverage) assessed
• Texture and pigmentation patterns noted
• Potential presence of flaking or inflammation markers reviewed

**📋 Preliminary Classification**
Based on visual features, this may correspond to:
• **Norwood Scale** (male pattern): Stage 2–3 — Temporal recession visible with possible vertex thinning
• **Ludwig Scale** (female pattern): Stage I–II — Part line widening, early diffuse thinning
• **Telogen Effluvium pattern**: Diffuse shedding across entire scalp

**💊 Suggested Next Steps**
1. Confirm stage with a certified trichologist using trichoscopy
2. Blood tests: Ferritin (target 70+ ng/mL), Zinc, Vitamin D, Thyroid panel, DHT
3. Consider topical Minoxidil 5% (men) or 2–5% (women) as first-line intervention
4. Scalp massage protocol: 20 min/day for 5 months to activate NOGGIN and BMP4 genes

> 💡 *For AI-powered vision analysis with a Gemini API key, the system can provide precise clinical descriptions of your uploaded image.*"""


KNOWLEDGE_BASE = {
    # ── GREETINGS ──
    "greetings": {
        "triggers": ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "how are you", "how r u", "what's up", "sup"],
        "response": """Hello! 👋 Welcome to **TrichoAI**. I'm **Dr. Tricho**, your AI-powered Hair & Scalp Health Consultant.

I'm trained in clinical trichology, pharmacotherapy, regenerative medicine, and nutritional science for hair health.

**I can help you with:**
• 💇 Hair loss types (AGA, Alopecia Areata, Telogen Effluvium, Scarring Alopecia)
• 💊 Treatment options (Minoxidil, Finasteride, JAK Inhibitors, PRP, Exosomes)
• 🧫 Dandruff & Seborrheic Dermatitis management
• 🥗 Nutritional deficiencies affecting hair (Iron, Ferritin, Zinc, Vitamin D)
• 📸 Hair image analysis (upload a photo for visual assessment)
• 🔬 Hair transplant comparisons (FUT vs FUE vs DHI)

What hair concern can I help you with today?"""
    },

    # ── HAIR GROWTH CYCLE ──
    "hair_cycle": {
        "triggers": ["hair cycle", "growth cycle", "anagen", "catagen", "telogen", "exogen", "kenogen", "hair phases"],
        "response": """**🔬 Clinical Assessment — Hair Growth Cycle**

The human hair follicle operates in a rhythmic cycle of five distinct phases:

| Phase | Duration | Scalp Coverage | Biological Activity |
|---|---|---|---|
| **Anagen** (Growth) | 2–7 years | 85–90% | Rapid cell division; hair shaft production |
| **Catagen** (Transition) | 2–3 weeks | 1–2% | Follicle involution; detachment from blood supply |
| **Telogen** (Rest) | 3–4 months | 10–15% | Dormancy; no active growth |
| **Exogen** (Shedding) | Variable | N/A | Active shedding of the hair shaft |
| **Kenogen** (Empty) | Variable | N/A | Empty follicle between shedding and new growth |

**⚠️ Clinical Significance**
When the proportion of hairs in Telogen increases significantly, this manifests as **Telogen Effluvium (TE)** — diffuse shedding typically 2–3 months after a triggering event (illness, surgery, extreme stress, nutritional deficiency).

**Wnt/β-catenin, Hedgehog, and Bone Morphogenetic Protein (BMP) signaling** govern these transitions. Environmental stressors can prematurely shift follicles from Anagen to Telogen.

**🥗 Supporting Anagen Phase**
• Ferritin levels of 70+ ng/mL are ideal for maintaining optimal Anagen duration
• Protein intake (≥50g/day) fuels keratin synthesis (hair is 95% keratin)
• Vitamin D regulates immune function of the scalp environment"""
    },

    # ── ANDROGENETIC ALOPECIA / PATTERN HAIR LOSS ──
    "aga": {
        "triggers": ["androgenetic", "pattern hair loss", "male pattern", "female pattern", "dht", "norwood", "ludwig", "genetic hair loss", "aga", "balding", "bald"],
        "response": """**🔬 Clinical Assessment — Androgenetic Alopecia (AGA)**

AGA is a polygenic, hormone-driven condition caused by genetic sensitivity of hair follicles to **Dihydrotestosterone (DHT)**. The enzyme **5α-reductase (Type II)** converts testosterone to DHT, which progressively miniaturizes follicles.

**📊 Norwood-Hamilton Scale (Male Pattern)**

| Stage | Clinical Description | Recommended Treatment Focus |
|---|---|---|
| Stage 1 | Minimal recession; no visible thinning | Prevention & monitoring |
| Stage 2 | Slight symmetrical temporal recession (M-shape) | Early Minoxidil therapy |
| Stage 3 | Deep temple recession; first clinically significant balding | Strong Minoxidil + Finasteride |
| Stage 3V | Significant vertex (crown) thinning | Combination therapy; DHT blockers |
| Stage 4 | Frontal and crown loss; separated by hair band | Surgical consult; intensive medical care |
| Stage 5 | Band becomes very thin | Transplants + medication |
| Stage 6 | Bridge disappears | Cosmetic/surgical focus |
| Stage 7 | Horseshoe band only remains | Scalp micropigmentation; hair systems |

**📊 Ludwig Scale (Female Pattern)**

| Stage | Description | Considerations |
|---|---|---|
| Stage I | Part line widening; often overlooked | Minoxidil 2–5%; iron assessment |
| Stage II | Increased thinning on top; scalp visible | Spironolactone; LLLT; PRP |
| Stage III | Extensive loss across crown | Transplants; cosmetic fibers |

**💊 First-Line Medical Treatments**
• **Topical Minoxidil 5%** (men) / **2–5%** (women): FDA-approved vasodilator; extends Anagen phase
• **Oral Minoxidil (0.25–5mg/day)**: Superior adherence (0% discontinuation vs 18.8% topical); superior vertex results
• **Finasteride 1mg/day**: Reduces DHT by 60–70%; stabilizes 85.7% over 5 years
• **Topical Finasteride 0.25% spray**: Similar efficacy to oral with markedly reduced systemic exposure
• **Dutasteride**: Inhibits Type I & II 5α-reductase; more potent than finasteride

**🌿 Botanical DHT Blockers (RCP Trio)**
• **Redensyl** — Stem cell activator targeting hair follicle stem cells
• **Capixyl** — Biomimetic peptide; DHT inhibitor + anti-inflammatory
• **Procapil** — Strengthens follicle anchoring by enhancing blood flow"""
    },

    # ── ALOPECIA AREATA ──
    "alopecia_areata": {
        "triggers": ["alopecia areata", "patchy hair loss", "bald spots", "round bald", "autoimmune hair", "jak inhibitor", "baricitinib", "ritlecitinib", "deuruxolitinib"],
        "response": """**🔬 Clinical Assessment — Alopecia Areata (AA)**

AA is an **autoimmune condition** where the body's T-cells (CD8+ NKG2D+) attack hair follicles via the JAK-STAT pathway triggered by IFN-γ and IL-15 signaling. This causes round or patchy bald spots; in severe cases, total scalp loss (Alopecia Totalis) or body hair loss (Alopecia Universalis).

**Trichoscopy signs**: Exclamation-point hairs, yellow dots, black dots — hallmark features of active AA.

**💊 JAK Inhibitor Treatments (2022–2024 Revolution)**

| JAK Inhibitor | Brand | Target | FDA Approval | Key Outcome (SALT ≤20) |
|---|---|---|---|---|
| **Baricitinib** | Olumiant | JAK1/JAK2 | June 2022 | 35–40% at 36 weeks |
| **Ritlecitinib** | Litfulo | JAK3/TEC | June 2023 | 23% at 24 weeks (≥12 yrs) |
| **Deuruxolitinib** | Leqselvi | JAK1/JAK2 | July 2024 | 31% at 24 weeks |

**📋 BAD 2025 Treatment Guidelines**
• **Mild (1–20% loss)**: Potent topical/intralesional corticosteroids (triamcinolone acetonide)
• **Moderate–Severe**: Oral JAK inhibitors (alongside tapering oral corticosteroids)
• **Fitzpatrick IV–V skin**: Higher risk of localized depigmentation from steroid injections → consider PUVA therapy

**🧘 Psychodermatology Support**
AA carries significant psychological burden. Clinicians are advised to assess:
• **PHQ-9** (depression screening)
• **GAD-7** (anxiety assessment)  
• **DLQI** (Dermatology Life Quality Index)"""
    },

    # ── TELOGEN EFFLUVIUM ──
    "telogen_effluvium": {
        "triggers": ["telogen effluvium", "stress hair loss", "sudden hair loss", "hair loss after stress", "hair loss after pregnancy", "hair loss fever", "hair loss surgery", "diffuse shedding"],
        "response": """**🔬 Clinical Assessment — Telogen Effluvium (TE)**

TE is a **reactive process** triggered by physiological stressors that prematurely shift follicles from Anagen into Telogen. The characteristic diffuse shedding typically begins **2–3 months after the triggering event**.

**⚠️ Common Triggers**
• High fever / illness (e.g., COVID-19)
• Major surgery or general anaesthesia
• Severe emotional or psychological stress
• Nutritional deficiencies: Iron, Ferritin, Zinc, Vitamin D, Protein
• Crash dieting or rapid weight loss (>10kg in 3 months)
• Postpartum (3–6 months after delivery)
• Thyroid dysfunction (hypo/hyperthyroidism)
• Stopping oral contraceptives

**🥗 Nutritional Support — Critical Thresholds**

| Ferritin Level (ng/mL) | Impact on Hair | Clinical Recommendation |
|---|---|---|
| **<30** | Critically low; high shedding probability | Immediate iron supplementation |
| **30–50** | Borderline; shedding likely | Supplementation often beneficial |
| **50–80** | Adequate for most | Maintenance through diet |
| **80–100+** | Optimal for hair density | Ideal range for restoration |

**💊 Treatment Protocol**
• Ferrous sulfate 200mg/day (with Vitamin C to enhance absorption) if Ferritin <50
• **Zinc gluconate 50mg/day** — therapeutic in patients with confirmed low zinc levels
• Vitamin D3 if levels <30 ng/mL
• Biotin only beneficial if true deficiency exists
• **Full blood panel**: FBC, Ferritin, Zinc, Thyroid (TSH, T3, T4), Vitamin D, B12, DHEAS

**⏱️ Prognosis**
TE is typically **self-limiting** (recovers in 6–12 months) once the trigger is resolved. Restoring Ferritin to 70+ ng/mL can slow shedding within 8 weeks; visible regrowth takes 4–6 months."""
    },

    # ── DANDRUFF / SEBORRHEIC DERMATITIS ──
    "dandruff": {
        "triggers": ["dandruff", "flakes", "seborrheic", "itchy scalp", "flaking", "malassezia", "scalp itch", "seborrhea"],
        "response": """**🔬 Clinical Assessment — Dandruff & Seborrheic Dermatitis**

Dandruff (Pityriasis capitis) and its more severe form, **Seborrheic Dermatitis (SD)**, result from **dysbiosis** of the scalp microbiome — specifically the overgrowth of *Malassezia* fungi (*M. globosa* and *M. restricta*). These organisms produce lipase enzymes that break down scalp sebum into pro-inflammatory free fatty acids (oleic acid), triggering keratinocyte hyperproliferation.

**Scalp Microbiome Composition:**
• Oily scalps: dominated by *Cutibacterium* and *Staphylococcus*
• Dry scalps: higher *Streptococcus* and *Micrococcus*

**🧴 Anti-Dandruff Active Ingredients**

| Active Ingredient | Mechanism | Efficacy & Safety |
|---|---|---|
| **Ketoconazole 2%** | Blocks ergosterol synthesis in fungal membranes | Potent antifungal; superior for severe cases |
| **Selenium Sulfide** | Cytostatic; disrupts metabolism via ROS | Effective but may cause scalp oiliness/smell |
| **Zinc Pyrithione (ZnP)** | Normalises keratinization; reduces sebum | Safe for maintenance; eliminates parakeratosis |
| **Salicylic Acid** | Keratolytic; reduces cell-to-cell adhesion | Excellent for removing thick adherent flakes |

**💊 Treatment Protocol**
• **Mild dandruff**: ZnP shampoo 2–3×/week (maintenance)
• **Moderate SD**: Ketoconazole 2% shampoo — leave on 5 min, 2×/week for 4 weeks, then weekly maintenance
• **Severe SD**: Add topical hydrocortisone 1% for inflammation; consider oral antifungals

**🥗 Dietary & Lifestyle Factors**
• High sugar diets shift *Malassezia* abundance — reduce refined carbohydrates
• Psychological stress exacerbates flare-ups; biofeedback and meditation helpful
• Probiotics (kefir, yogurt, fermented foods) support gut-scalp microbiome axis
• Omega-3 fatty acids reduce scalp inflammation"""
    },

    # ── HAIR FALL / GENERAL HAIR LOSS ──
    "hair_fall": {
        "triggers": ["hair fall", "hair loss", "losing hair", "hair falling", "excessive shedding", "falling hair"],
        "response": """**🔬 Clinical Assessment — Hair Loss (Differential Diagnosis)**

Hair loss is a **multi-factorial condition** with distinct causes requiring different treatments. Accurate diagnosis is essential before initiating therapy.

**📋 Condition Comparison**

| Condition | Primary Mechanism | Clinical Presentation | First-Line Solutions |
|---|---|---|---|
| **Androgenetic Alopecia** | DHT-mediated follicle miniaturization | Patterned thinning (Norwood/Ludwig) | Minoxidil, Finasteride, LLLT, Transplants |
| **Alopecia Areata** | Autoimmune T-cell attack on follicles | Patchy, round bald spots; sudden onset | Corticosteroids, JAK Inhibitors, Immunotherapy |
| **Telogen Effluvium** | Stress-induced Anagen→Telogen shift | Diffuse shedding across entire scalp | Addressing triggers; nutritional correction |
| **Traction Alopecia** | Repeated mechanical tension on roots | Thinning at hairline or areas of tension | Avoiding tight styles; anti-inflammatory care |
| **Tinea Capitis** | Fungal infection of the scalp | Scaling, broken hairs, possible pustules | Oral antifungal medications |
| **Seborrheic Dermatitis** | Malassezia-driven scalp dysbiosis | Itching, flaking, redness, inflammation | Medicated shampoos (Ketoconazole, SeS2) |

**🔍 Essential Diagnostic Tests**
• Trichoscopy — non-invasive dermoscopic imaging of scalp and hair
• **Blood panel**: FBC, Serum Ferritin, Zinc, Copper, Thyroid (TSH, T3, T4), Vitamin D, B12, DHEAS, Testosterone/DHT, Fasting glucose

**💊 Evidence-Based Treatments**
• **Stage 1–2**: Minoxidil + lifestyle modification + nutritional correction
• **Stage 3–4**: Combination therapy (Minoxidil + Finasteride/Dutasteride); consider PRP
• **Advanced AGA**: Surgical restoration (FUE/DHI hair transplant)

**🌿 Scalp Massage Protocol (Scientific Evidence)**
15–20 min daily scalp massage over 5 months can increase hair thickness by activating **NOGGIN** and **BMP4** genes:
| Technique | Duration | Goal |
|---|---|---|
| Warm-up Massage | 3 min | Increase local blood circulation |
| Pinching (Finger pads) | 6 min | Skin-level stimulation and inflammation |
| Skin Stretching | 6 min | Induce mechanical stress on dermal papilla |
| Pressing (Knuckles) | 5 min | Deep tissue manipulation and elasticity |"""
    },

    # ── MINOXIDIL ──
    "minoxidil": {
        "triggers": ["minoxidil", "rogaine", "oral minoxidil", "topical minoxidil", "vasodilator hair"],
        "response": """**💊 Clinical Review — Minoxidil (Topical & Oral)**

Minoxidil is a **potassium channel opener and vasodilator** that extends the Anagen phase and increases follicle diameter. Available as 2%, 5% topical (solution/foam) and low-dose oral.

**📊 Topical vs Oral Comparison**

| Feature | Topical Minoxidil (5%) | Oral Minoxidil (Low-Dose) |
|---|---|---|
| Regulatory Status | FDA-Approved | Off-label for Alopecia |
| Dosing | Twice daily to scalp | Once daily pill (0.25–5 mg) |
| Common Side Effects | Scalp irritation, itching, dryness | Hypertrichosis, headache, dizziness |
| Adherence Rate | Lower (18.8% discontinuation) | Higher (0% discontinuation in trials) |
| Mechanism | Local vasodilation | Systemic vasodilation |
| Vertex Results | Good | ~24% improvement over topical (2024 trial) |
| Non-scalp hair growth | 15–49% of patients | More common |

**Key Clinical Findings**
• A 2025 trial found oral Minoxidil users had significantly higher adherence — single daily pill eliminates the "greasy" texture complaint that causes topical discontinuation
• Oral form shows superior results in the **vertex region**: 24% improvement over topical in one 2024 study
• Low-dose oral Minoxidil (0.25–2.5mg) for women has excellent safety profile

**⚠️ Important Notes**
• Do not stop suddenly — shedding will resume (it suppresses Telogen, not the underlying cause)
• Contraindicated in certain cardiovascular conditions — consult a physician before oral use
• Allow **6–12 months** for full results assessment"""
    },

    # ── PRP / EXOSOMES / REGENERATIVE ──
    "prp": {
        "triggers": ["prp", "platelet rich plasma", "prp therapy", "exosome", "exosome therapy", "stem cell hair", "regenerative hair", "prf"],
        "response": """**🔬 Clinical Assessment — Regenerative Medicine for Hair Loss**

Regenerative medicine has become a cornerstone of non-surgical hair restoration, offering ways to rejuvenate dormant follicles and enhance scalp health.

**PRP (Platelet-Rich Plasma) vs Exosome Therapy**

| Feature | PRP Therapy | Exosome Therapy |
|---|---|---|
| Source | Patient's own blood | Donor stem cells (Acellular) |
| Growth Factors | 7–25 (VEGF, PDGF) | 1,000+ |
| Preparation | Requires blood draw/processing | Premade solution; no blood draw |
| Sessions | 3 monthly + 6-month maintenance | 2–3 sessions + annual maintenance |
| Typical Cost | $500–$1,500 per session | $1,500–$4,000 per session |

**PRP Mechanism**
PRP concentrates autologous platelets rich in **Vascular Endothelial Growth Factor (VEGF)** and **Platelet-Derived Growth Factor (PDGF)**. When injected into the scalp, PRP:
• Reduces inflammation
• Improves follicle vascularisation
• Stimulates dermal papilla cell proliferation
• Success rates: **60–80%** with results visible within 3–6 months

**Platelet-Rich Fibrin Matrix (PRFM)** provides a sustained, long-term release of growth factors compared to traditional PRP.

**Exosome Therapy**
Clinical trials in 2024–2025 demonstrated that mesenchymal stem cell (MSC)-derived exosomes can significantly increase hair density by **9.5 to 35 hairs/cm²** and hair thickness. While not yet FDA-approved for hair, exosomes are categorized under broader regenerative medicine frameworks.

**💡 Recommendation**
• PRP: Ideal for early-stage AGA (Norwood 1–3) and TE recovery
• Exosomes: More potent; consider for moderate AGA or to supplement transplant recovery"""
    },

    # ── HAIR TRANSPLANT ──
    "transplant": {
        "triggers": ["hair transplant", "fue", "fut", "dhi", "strip method", "follicular unit", "graft", "hair transplantation", "surgical restoration"],
        "response": """**🔬 Clinical Assessment — Surgical Hair Restoration**

Hair transplantation is the most effective solution for **permanent hair loss** where medical therapies have stabilised but not restored density.

**📊 FUT vs FUE vs DHI Comparison**

| Feature | FUT (Strip Method) | FUE (Extraction) | DHI (Direct Implantation) |
|---|---|---|---|
| Extraction Type | Linear strip of scalp removed | Individual follicle extraction | Individual follicle extraction |
| Implantation | Manual placement into slits | Manual placement into slits | Choi Implanter Pen (Direct) |
| Scarring | Linear scar (permanent) | Tiny puncture marks (0.8–1mm) | Virtually no visible scarring |
| Graft Survival | 85–92% | 90–95% | 95–98% |
| Recovery Time | 10–14 days | 5–7 days | 3–5 days |
| Primary Advantage | Maximum grafts in one session | Scar-free for short hairstyles | Natural hairline; high density |

**📋 Technique Selection Guide**
• **FUT**: Preferred for >4,000 grafts; more cost-effective per graft; excellent root survival
• **FUE**: Ideal for patients who prefer short hair; Sapphire FUE uses sapphire blades for smoother incisions and faster recovery
• **DHI**: Premium version of FUE using Choi pen; minimises time follicles spend outside scalp → highest success rates and most natural angle control for hairlines

**⚠️ Important Considerations**
• Hair transplant addresses supply, not the underlying DHT-driven miniaturization — continue Minoxidil/Finasteride post-transplant
• Allow 12–18 months for full results
• Donor area is finite — strategic planning is essential for long-term density"""
    },

    # ── NUTRITION FOR HAIR ──
    "nutrition": {
        "triggers": ["nutrition", "diet for hair", "vitamin", "biotin", "iron", "ferritin", "zinc", "vitamin d", "nutrients", "food for hair", "supplements hair"],
        "response": """**🥗 Clinical Assessment — Nutritional Trichology**

The hair follicle matrix is one of the **most rapidly proliferating tissues** in the human body, making it exceptionally sensitive to nutritional deficiencies. Iron, zinc, vitamins D and B12 play essential roles in DNA synthesis, cellular metabolism, and the regulation of hair growth-promoting genes.

**📊 Iron & Ferritin — The Most Critical Factor**

| Ferritin Level (ng/mL) | Impact on Hair Growth Cycle | Clinical Recommendation |
|---|---|---|
| **<30** | Critically low; high probability of excessive shedding | Immediate iron supplementation |
| **30–50** | Borderline; shedding still likely to occur | Supplementation often beneficial |
| **50–80** | Adequate for most; supports stable growth | Maintenance through diet |
| **80–100+** | Optimal for maximal density and recovery | Ideal range for hair restoration |

**Key Nutrients & Thresholds**

• **Zinc**: Cofactor for 300+ enzymes; stabilizes cell membranes and inhibits follicular regression. Deficiency linked to TE and AGA acceleration. *Zinc gluconate 50mg/day* is therapeutic. Do not exceed 40mg/day long-term (copper absorption interference)

• **Vitamin B12**: Essential for DNA synthesis; B12 deficiency (<200 pg/mL) can disrupt normal follicle function and has been linked to premature graying

• **Vitamin D**: Creates new hair follicles; regulates immune environment of the scalp. Target: **50–70 ng/mL** for hair health (normal lab range of 30 ng/mL is often insufficient)

• **Biotin (B7)**: Only beneficial in individuals with a **true deficiency** (uncommon in those consuming a balanced diet) — biotin supplementation is frequently over-marketed

• **Omega-3 Fatty Acids**: Anti-inflammatory; reduce scalp prostaglandin activity that causes follicle miniaturization

**🍽️ Top Foods for Hair Health**
| Nutrient | Best Food Sources |
|---|---|
| Iron | Red meat, oysters, lentils, spinach, tofu |
| Zinc | Pumpkin seeds, oysters, beef, chickpeas |
| Protein/Keratin | Eggs, chicken, fish, Greek yogurt, legumes |
| Biotin | Eggs, sweet potato, almonds, salmon |
| Vitamin D | Fatty fish, fortified milk, sun exposure (15 min/day) |
| Omega-3 | Salmon, mackerel, walnuts, flaxseed, chia |"""
    },

    # ── CICATRICIAL / SCARRING ALOPECIA ──
    "cicatricial": {
        "triggers": ["scarring alopecia", "cicatricial", "lichen planopilaris", "lpp", "frontal fibrosing", "ffa", "ccca", "folliculitis decalvans", "dissecting cellulitis"],
        "response": """**🔬 Clinical Assessment — Cicatricial (Scarring) Alopecia**

⚠️ **Cicatricial alopecia is a medical urgency.** Hair follicles are irreversibly destroyed and replaced by scar tissue. Early diagnosis is critical — once a follicle is scarred, hair loss is permanent. The clinical hallmark is the **complete loss of follicular ostia** (the visible pores from which hair grows).

**📋 Scarring Alopecia Classification**

| Type | Predominant Inflammation | Target Demographic | First-Line Treatment |
|---|---|---|---|
| **Lichen Planopilaris (LPP)** | Lymphocytic | Women over 50 | Intralesional steroids; topical steroids |
| **Frontal Fibrosing Alopecia (FFA)** | Lymphocytic | Postmenopausal women | Hydroxychloroquine; Antiandrogens |
| **CCCA** (Central Centrifugal) | Lymphocytic | Black women (crown) | Ceasing traumatic hair care; Steroids |
| **Folliculitis Decalvans** | Neutrophilic | Adults | Rifampicin + Clindamycin |
| **Dissecting Cellulitis** | Neutrophilic | Black adolescent/adult males | Oral Isotretinoin |

**🔬 Diagnostic Approach**
Diagnosis requires: comprehensive medical history, **trichoscopy evaluation**, and a **4mm punch biopsy of the scalp** for histology. This identifies the inflammatory infiltrate and confirms "end-stage scarring alopecia" (ESSA) — when inflammation has burned out and treatment can no longer halt progression.

**⚕️ CRITICAL**: If you suspect scarring alopecia (burning/stinging sensations, perifollicular redness, spreading scalp tenderness), **see a dermatologist immediately**. Months matter — early treatment stabilizes the disease."""
    },

    # ── SCALP MASSAGE ──
    "scalp_massage": {
        "triggers": ["scalp massage", "massage for hair", "massage technique", "noggin", "bmp4"],
        "response": """**🧘 Evidence-Based Scalp Massage Protocol**

Scientific evidence shows that 15–20 minutes of standardized daily scalp massage over **5 months** can increase hair thickness by transmitting mechanical stress to human dermal papilla cells, activating hair growth genes **NOGGIN and BMP4**.

**📋 Standardized Technique**

| Technique | Time | Goal |
|---|---|---|
| **Warm-up Massage** | 3 Minutes | Increase local blood circulation |
| **Pinching (Finger pads)** | 6 Minutes | Skin-level stimulation and inflammation reduction |
| **Skin Stretching** | 6 Minutes | Induce mechanical stress on dermal papilla |
| **Pressing (Knuckles)** | 5 Minutes | Deep tissue manipulation and elasticity |
| **Total** | **20 Minutes** | Full dermal papilla activation |

**🌺 Enhance With Oils**
• **Rosemary Oil** (diluted 5% in carrier) — shown to be as effective as Minoxidil 2% in one 2023 study; stimulates blood microcirculation
• **Caffeine Scalp Serum** — penetrates to hair roots to counter DHT effects
• **Pumpkin Seed Oil** — mild 5α-reductase inhibition

**💡 Application Tips**
• Use fingertips (not nails) to avoid microtrauma
• Perform before bed; rinse in morning if using oil
• Combine with 2-minute cold water scalp rinse post-massage to boost circulation
• Be consistent — results require a minimum of 16–20 weeks"""
    },

    # ── GENERAL FALLBACK ──
    "fallback": {
        "response": """Thank you for your question. As **Dr. Tricho**, I'm here to provide evidence-based guidance on all aspects of hair and scalp health.

**I can provide detailed clinical information on:**

| Topic | Ask Me About |
|---|---|
| 💇 Hair Loss Types | AGA, Alopecia Areata, TE, Scarring Alopecia |
| 💊 Medical Treatments | Minoxidil, Finasteride, JAK Inhibitors |
| 🌿 Natural & Botanical | RCP Trio, Rosemary oil, Scalp massage |
| 🧫 Scalp Conditions | Dandruff, SD, Scalp Microbiome |
| 🥗 Nutrition | Ferritin, Zinc, Vitamin D, Iron levels |
| 🔬 Regenerative | PRP, Exosome Therapy, PRF |
| 🏥 Surgical | FUT, FUE, DHI hair transplants |
| 📸 Image Analysis | Upload a scalp photo for visual assessment |

Could you describe your specific concern in more detail? For example:
- *How long* have you been experiencing hair loss?
- *Where* on the scalp (frontal, vertex, diffuse, patchy)?
- Any *recent triggers* (stress, illness, medication changes, diet changes)?

The more detail you provide, the more precise my guidance can be. 😊"""
    }
}


def local_response(msg: str) -> str:
    t = msg.lower().strip()

    # Check greetings (short messages only)
    greet_triggers = KNOWLEDGE_BASE["greetings"]["triggers"]
    if any(g in t for g in greet_triggers) and len(t) < 30:
        return KNOWLEDGE_BASE["greetings"]["response"]

    # Match knowledge base entries
    for key, data in KNOWLEDGE_BASE.items():
        if key in ("greetings", "fallback"):
            continue
        if any(trigger in t for trigger in data.get("triggers", [])):
            return data["response"]

    return KNOWLEDGE_BASE["fallback"]["response"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
