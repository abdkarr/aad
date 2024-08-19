import numpy as np

from .typing import RNG_TYPE

def _check_rng(rng: RNG_TYPE):
    """Checks if a given input for random number generator is valid.

    A valid rng input function can be either an int indicating seed number, 
    a `np.random.Generator` object, or `None`. 
    """
    
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    
    return rng