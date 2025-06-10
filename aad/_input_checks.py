import numpy as np

from .typing import RNGType

def _check_rng(rng: RNGType):
    """Checks if a given input for random number generator is valid.

    A valid rng input function can be either an int indicating seed number, 
    a `np.random.Generator` object, or `None`. 
    """
    
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    
    return rng