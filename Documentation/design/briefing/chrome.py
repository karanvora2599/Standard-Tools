"""Where headless Chrome lives, shared by build.py and check.py.

Set CHROME to the browser's path on a machine where it is installed elsewhere
(any Chromium-based browser recent enough to support @page margin boxes works).
The profile directory keeps these runs away from your everyday browser session.
"""

import os
from pathlib import Path

CHROME = os.environ.get(
    "CHROME", r"C:/Program Files/Google/Chrome/Application/chrome.exe"
)
PROFILE = Path(__file__).resolve().parent / ".chrome-profile"
