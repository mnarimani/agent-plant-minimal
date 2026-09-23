#!/usr/bin/env python3
"""Start the minimal AgentPlant API (uvicorn)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "labcd_agents" / "src"))

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "minimal_api.app:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "1") == "1",
    )
