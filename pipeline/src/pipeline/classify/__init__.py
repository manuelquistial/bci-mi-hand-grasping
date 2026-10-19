"""Classifier and evaluation chosen after the feature step."""

from pipeline.classify.evaluate import SCHEMES, Evaluation, SubjectTrials, evaluate_classifier
from pipeline.classify.models import CLASSIFIERS

__all__ = [
    "CLASSIFIERS",
    "SCHEMES",
    "Evaluation",
    "SubjectTrials",
    "evaluate_classifier",
]
