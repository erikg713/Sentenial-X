import streamlit as st
from sentenial_core.simulator.blind_spot_tracker import BlindSpotTracker

def show_blind_spot_tab():
    tracker = BlindSpotTracker()
    st.title("🕳️ Blind Spot Tracker")

    st.subheader("📋 Current Blind Spots")
    spots = tracker.audit_all_zones()
    for spot in spots:
        st.markdown(f"- **[{spot['risk']}]** `{spot['zone']}` — {spot['description']}")

    st.subheader("⚔️ Simulate Exploitation")
    zones = [s['zone'] for s in spots]
    selected = st.selectbox("Select Zone", zones)
    if st.button("Simulate Attack"):
        result = tracker.simulate_exploitation(selected)
        st.success(f"Simulation successful: {result['exploitation_vector']}")
        st.code(result, language="json")