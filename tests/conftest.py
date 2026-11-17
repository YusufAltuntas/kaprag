import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load local settings (e.g. DATABASE_URL) without overriding the real environment.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


@pytest.fixture
def tmp_dir():
    """A fresh temporary directory.

    Used instead of pytest's `tmp_path`, whose "current" symlink creation takes
    about a minute per test on the development Windows machine.
    """
    with tempfile.TemporaryDirectory(prefix="kaprag-test-") as path:
        yield Path(path)
