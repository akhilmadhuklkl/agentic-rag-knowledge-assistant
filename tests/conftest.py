"""Keep the test suite independent from a developer's local model settings."""

import os

os.environ["LLM_PROVIDER"] = "deterministic"
os.environ["EMBEDDING_PROVIDER"] = "local"
