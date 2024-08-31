from enum import Enum
from typing import Optional

import numpy as np

RNG_TYPE = Optional[np.random.Generator | int]

class AnnotatorType(Enum):
    RELIABLE = "reliable" # better than random annotator
    NOCONTROL = "no-control" # No control over the reliability of the annotator
    UNIFORM = "uniform" # uniformly picks a random response 
    SAME = "same" # picks the same class for all data points
    ADVERSE = "adverse" # picks an answer that is different than true class

class AdversaryType(Enum):
    RANDOM = "random"
    DS = "ds"