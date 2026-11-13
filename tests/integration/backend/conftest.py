"""Live services are opt-in; normal test runs never touch a developer's business database."""

import os
import pytest


def pytest_collection_modifyitems(items):
    if os.getenv("RUN_PLATFORM_INTEGRATION") == "1":
        return
    marker = pytest.mark.skip(
        reason="Set RUN_PLATFORM_INTEGRATION=1 with dedicated platform test services"
    )
    for item in items:
        if "/integration/backend/" in str(item.path):
            item.add_marker(marker)
