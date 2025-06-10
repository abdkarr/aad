from enum import Enum
from typing import Optional

import numpy as np

RNGType = Optional[np.random.Generator | int]
""" 
The type alias for random number generator. 

- If an `int`, a random number generator whose seed number is set to
  the given number is created. 
- If `np.random.Generator`, it is used as the random number generator.
- If `None`, a random generator without setting any seed number is
  created.
"""

class AnnotatorType(Enum):
    RELIABLE = "reliable" # better than random annotator
    NOCONTROL = "no-control" # No control over the reliability of the annotator
    UNIFORM = "uniform" # uniformly picks a random response 
    SAME = "same" # picks the same class for all data points
    ADVERSE = "adverse" # picks an answer that is different than true class

class AdversaryType(Enum):
    RANDOM = "random"
    DS = "ds"