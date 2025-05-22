from AlgorithmImports import *
# endregion

from dataclasses import dataclass, field
from typing import List, Protocol, Literal
import pandas as pd
import time
from abc import ABC, abstractmethod
from ta.volatility import average_true_range
import scipy
from tqdm import tqdm
from .rules import *
import math
from itertools import combinations


class Rule(Protocol):
    """Interface: una regola restituisce True/False sui dati passati."""
    def check(self, window: pd.DataFrame) -> bool: ...

@dataclass
class BaseStrategy:
    id: str
    rules: List[Rule] = field(default_factory=list)
    bias: Literal["bullish", "bearish", "both"] = "both"

    