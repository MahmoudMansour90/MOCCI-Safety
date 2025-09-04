
import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="MO-CCI Safety Predictor", layout="wide")

@st.cache_resource
def load_models():
    with open("model_event.pkl","rb") as f:
        model_event = pickle.load(f)
    with open("model_severity.pkl","rb") as f:
        model_sev = pickle.load(f)
    checklist = pd.read_csv("hazard_checklist.csv")
    return model_event, model_sev, checklist

model_event, model_sev, checklist_df = load_models()

st.title("MO-CCI Safety Prediction – MVP")
st.caption("Planning-level inputs → predicted hazards & severity → preventive checklist")

# --- Sidebar / inputs ---
st.header("Project Inputs")

col1, col2, col3 = st.columns(3)
with col1:
    project_type = st.selectbox("Project Type", ["Commercial","Residential","Industrial","Infrastructure"])
    duration = st.selectbox("Duration Band", ["<6 months","6–18 months",">18 months"])
    shifts = st.selectbox("Shifts", ["Day only","Multiple shifts","24-hour operations"])

with col2:
    crew_bucket = st.selectbox("Crew Size (bucket)", ["<20","20–100","100–500",">500"])
    overtime = st.selectbox("Overtime Expected", ["Low (<10%)","Medium (10–25%)","High (>25%)"])
    union = st.selectbox("Union Presence", ["None","Partial","Majority"])

with col3:
    state = st.selectbox("State/Region", [
        "AL","AK","AZ","AR","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN",
        "KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ",
        "NM","NV","NY","OH","OK","OR","PA","PR","RI","SC","SD","TN","TX","UT","VA","VT",
        "WA","WI","WV","WY"
    ])
    owner = st.selectbox("Owner Type", ["Private","Public","Institutional"])
    delivery = st.selectbox("Delivery Method", ["DBB","DB","CMAR","IPD"])

st.subheader("Exposure Toggles")
c1, c2, c3 = st.columns(3)
with c1:
    heights = st.checkbox("Working at Heights expected?")
    cranes = st.checkbox("Cranes / Heavy Lifts?")
with c2:
    electrical = st.checkbox("Energized Electrical exposure?")
    traffic = st.checkbox("Public/Traffic interface?")
with c3:
    pinch = st.checkbox("Powered machinery with pinch points?")
    confined = st.checkbox("Confined spaces?")

narrative = st.text_area("Short narrative (optional, 1–2 sentences)", "")

# --- Helper: map project type to a representative NAICS (simple proxy) ---
PROJECT_NAICS = {
    "Residential": "236115",
    "Commercial": "236220",
    "Industrial": "236210",
    "Infrastructure": "237310"
}

# Build synthetic text features similar to the training representation
def compose_features(project_type, state, narrative, heights, electrical, cranes, traffic, pinch, confined):
    tokens = []
    tokens.append("NAICS_" + PROJECT_NAICS.get(project_type, "236220"))
    tokens.append("STATE_" + state)
    # exposure-driven keywords that helped the model during training
    if heights: tokens += ["ladder","scaffold","edge","harness","guardrail"]
    if electrical: tokens += ["energized","lockout","tagout","gfcI","test before touch"]
    if cranes: tokens += ["crane","rigging","swing","boom","load"]
    if traffic: tokens += ["traffic","roadway","flagger","MOT","public"]
    if pinch: tokens += ["press","roller","conveyor","guard","pinch point"]
    if confined: tokens += ["confined space","permit","monitor"]
    if narrative:
        tokens.append(narrative)
    return " ".join(tokens)

if st.button("Predict"):
    feats = compose_features(project_type, state, narrative, heights, electrical, cranes, traffic, pinch, confined)

    # Event type prediction (Top-3)
    proba_e = model_event.predict_proba([feats])[0]
    classes_e = model_event.classes_
    top_idx = proba_e.argsort()[::-1][:3]
    top_events = [(classes_e[i], float(proba_e[i])) for i in top_idx]

    # Map detailed OSHA event titles to broader hazard buckets
    def bucketize_event(evt):
        e = evt.lower()
        if "fall" in e:
            return "Falls"
        if "pinched" in e or "compressed" in e or "caught" in e:
            return "Caught-in-between"
        if "swinging object" in e or "struck" in e:
            return "Struck-by"
        # default
        return "Other"

    hazard_buckets = [(bucketize_event(e), p) for e, p in top_events]

    # Severity prediction
    proba_s = model_sev.predict_proba([feats])[0]
    classes_s = model_sev.classes_
    sev_idx = proba_s.argmax()
    sev_pred = classes_s[sev_idx]
    sev_prob = float(proba_s[sev_idx])

    st.markdown("### Predicted Hazards (Top 3)")
    for hb, p in hazard_buckets:
        st.write(f"- **{hb}** — {p*100:.1f}%")

    st.markdown(f"### Predicted Severity: **{sev_pred}** ({sev_prob*100:.1f}% confidence)")

    # Checklist (union of controls for top hazard buckets)
    st.markdown("### Preventive Checklist")
    shown = set()
    for hb, _ in hazard_buckets:
        row = checklist_df[checklist_df["hazard"]==hb]
        if not row.empty and hb not in shown:
            st.write(f"**{hb}:** {row.iloc[0]['controls']}")
            shown.add(hb)

    # Downloadable summary
    summary = {
        "project_type": project_type,
        "state": state,
        "crew_bucket": crew_bucket,
        "duration": duration,
        "shifts": shifts,
        "overtime": overtime,
        "union": union,
        "owner": owner,
        "delivery": delivery,
        "exposures": {
            "heights": bool(heights),
            "electrical": bool(electrical),
            "cranes": bool(cranes),
            "traffic": bool(traffic),
            "pinch_points": bool(pinch),
            "confined_spaces": bool(confined),
        },
        "top_hazards": [{ "hazard": hb, "probability": float(p)} for hb, p in hazard_buckets],
        "severity": { "label": str(sev_pred), "probability": float(sev_prob) },
        "narrative": narrative
    }
    import json
    st.download_button("Download JSON Summary", data=json.dumps(summary, indent=2), file_name="prediction_summary.json", mime="application/json")

st.caption("MVP demo. Not a substitute for OSHA compliance or a competent person.")

