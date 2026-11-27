"""Display the existing Executive dashboard in the V5 secure shell."""
import runpy
from pathlib import Path
import streamlit as st
from frontend.v5_pages._common import demand_session

demand_session()
st.session_state["aurum_v5_executive_view"] = "Executive Dashboard"
runpy.run_path(str(Path(__file__).resolve().parents[1] / "v5_legacy" / "executive_dashboard.py"),run_name="__main__")
