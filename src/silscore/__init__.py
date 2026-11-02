"""Public interface for Silscore."""

from .model import Silscore, load_default_model
from .prediction import normalize_target, predict_targets, target_records
from .records import SiRNARecord, reverse_complement, to_rna

__all__ = [
    "SiRNARecord",
    "Silscore",
    "load_default_model",
    "normalize_target",
    "predict_targets",
    "reverse_complement",
    "target_records",
    "to_rna",
]

__version__ = "0.1.0"
