"""Concept metadata for SC-CBM (Chapter 3).

Reuses the same 8-concept dermoscopic vocabulary and ordering as GroundDerm
(Chapter 4) so the two chapters share one concept ontology:

    order = [TPN, APN, BWV, ISTR, RSTR, RDG, IDG, RS]

The class-indicator matrix encodes which concepts point to which diagnosis,
following the 7-point checklist / ABCD rule (see thesis Table 3.2). It is used
to initialise the interpretable linear head so that, before any training, each
concept already pushes the prediction toward its clinically associated class.
"""

import numpy as np

CONCEPT_CODES = ["TPN", "APN", "BWV", "ISTR", "RSTR", "RDG", "IDG", "RS"]

CONCEPT_NAMES = {
    "TPN": "typical pigment network",
    "APN": "atypical pigment network",
    "BWV": "blue-whitish veil",
    "ISTR": "irregular streaks",
    "RSTR": "regular streaks",
    "RDG": "regular dots and globules",
    "IDG": "irregular dots and globules",
    "RS": "regression structures",
}

# Clinical polarity per concept: +1 = benign indicator, -1 = malignant indicator.
# Order matches CONCEPT_CODES.
#   TPN  benign    | APN  malignant | BWV  malignant | ISTR malignant
#   RSTR benign    | RDG  benign    | IDG  malignant | RS   malignant
BENIGN_CONCEPTS = {"TPN", "RSTR", "RDG"}
MALIGNANT_CONCEPTS = {"APN", "BWV", "ISTR", "IDG", "RS"}

NUM_CONCEPTS = len(CONCEPT_CODES)
NUM_CLASSES = 2  # 0 = Nevus / Non-Melanoma, 1 = Melanoma


def class_indicator_matrix() -> np.ndarray:
    """Return the (NUM_CONCEPTS, NUM_CLASSES) class-indicator matrix.

    Column 0 (benign) is 1 for benign concepts; column 1 (melanoma) is 1 for
    malignant concepts. Used to initialise the interpretable linear head.
    """
    ind = np.zeros((NUM_CONCEPTS, NUM_CLASSES), dtype=np.float32)
    for i, code in enumerate(CONCEPT_CODES):
        if code in BENIGN_CONCEPTS:
            ind[i, 0] = 1.0
        else:
            ind[i, 1] = 1.0
    return ind
