from . import datasets
from . import simulations
from .aggregators.majority_voting import _apply as majority_voting
from .aggregators.dawid_skene import _apply as dawid_skene
from .detection import detect_attacks
from .calc_agreement_mat import calc_agreement_mat

from pathlib import Path
PROJECT_DIR = str(Path(__file__).parents[1])
