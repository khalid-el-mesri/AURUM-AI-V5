"""Display the existing NYSE V3 Cockpit and tabs under V5 authentication."""
import runpy
from pathlib import Path
from frontend.v5_pages._common import demand_session

demand_session()
runpy.run_path(str(Path(__file__).resolve().parents[1] / "v5_legacy" / "nyse_dashboard.py"),run_name="__main__")
