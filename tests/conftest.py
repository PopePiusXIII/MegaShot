import os
import sys

# Ensure src is on sys.path for test imports.
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--videos",
        action="store",
        default=None,
        help="Comma-separated list of video files for golf ball detection tests.",
    )
    parser.addoption(
        "--expected",
        action="store",
        default=None,
        help="Path to expected detection data (JSON or CSV).",
    )
    parser.addoption(
        "--frame-index",
        action="store",
        type=int,
        default=0,
        help="Default frame index to use when expected data omits it.",
    )
