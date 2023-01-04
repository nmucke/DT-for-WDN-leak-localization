from enum import Enum, auto
from pathlib import Path
from typing import Protocol


import numpy as np


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


