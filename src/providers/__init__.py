"""Application-owned chat provider contract and independent adapters."""

import os

# Use LiteLLM's packaged metadata; local chat does not need a remote cost-map refresh.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
