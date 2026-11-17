from pathlib import Path

from dotenv import load_dotenv

# Load local settings (e.g. DATABASE_URL) without overriding the real environment.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
