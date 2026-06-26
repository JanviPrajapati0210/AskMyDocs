"""
check_groq_model.py

Verifies that the Groq model configured in rag/chain.py is still
active and not deprecated. Groq periodically retires models with
short notice, so this script protects against silent production
failures.

Usage:
    python check_groq_model.py

Run this:
- Before every deployment
- Whenever you get a "model_decommissioned" error
- As a quick health check during development
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

# ── The model your app currently uses (keep in sync with rag/chain.py) ─────
CONFIGURED_MODEL = "openai/gpt-oss-20b"

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


def check_model():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env — cannot check models.")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(GROQ_MODELS_URL, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to reach Groq API: {e}")
        sys.exit(1)

    data = response.json()
    active_model_ids = {model["id"] for model in data.get("data", [])}

    print(f"📋 Groq currently has {len(active_model_ids)} active models.\n")

    if CONFIGURED_MODEL in active_model_ids:
        print(f"✅ Your configured model is ACTIVE: '{CONFIGURED_MODEL}'")
        print("   No action needed — safe to deploy.")
    else:
        print(f"⚠️  Your configured model is NOT in the active list: '{CONFIGURED_MODEL}'")
        print("   This means Groq has deprecated or renamed it.")
        print("\n   Suggested active alternatives:")

        # Show a few commonly-used active models as suggestions
        suggestions = [
            m for m in sorted(active_model_ids)
            if any(keyword in m.lower() for keyword in ["gpt-oss", "llama", "qwen"])
        ]
        for s in suggestions[:8]:
            print(f"     • {s}")

        print(f"\n   👉 Update CONFIGURED_MODEL in this script AND the model=")
        print(f"      parameter in rag/chain.py's build_chain() function.")
        sys.exit(1)


if __name__ == "__main__":
    check_model()