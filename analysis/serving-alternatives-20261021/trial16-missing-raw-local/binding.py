"""Pinned single-attempt collection context; no network at import."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'trial-16-candidate-s10-r4-operational-local'
OVERLAY=HERE.parent/'trial16-collection-decimal-overlay-local'
ASSESSMENT=ROOT/'remote-readback/partial-assessment-v1'
CURRENT_PRIMARY=ROOT/'remote-readback/current-primary-fallback-local/primary-current'
ACTUAL_RESULT='serving-candidate-s10-16/output/result.json'
