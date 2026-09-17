"""Keep unit tests independent of local observability credentials."""

import os

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
# The disabled SDK still validates key presence before checking tracing_enabled.
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "unit-test-public-key")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "unit-test-secret-key")
