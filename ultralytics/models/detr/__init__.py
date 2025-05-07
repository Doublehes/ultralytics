# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .model import DETR
from .predict import DETRPredictor
from .val import DETRValidator

__all__ = "DETRPredictor", "DETRValidator", "DETR"
