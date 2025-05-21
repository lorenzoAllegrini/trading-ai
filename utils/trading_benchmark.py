# region imports
from AlgorithmImports import *
# endregion

import pandas as pd
from typing import Optional, Callable, Tuple, Any, List

class MLTradingBenchmark:
    def __init__(self) -> None:
        """
        Inizializza un contenitore per gestire strategie di trading basate su machine learning.
        """
        self.strategies: Dict[str,Any] = {}

    def add_strategy(self, strategy: Any) -> None:
        """
        Aggiunge una nuova strategia al benchmark.

        Args:
            strategy (MLStrategy): La strategia di machine learning da aggiungere.
        """
        self.strategies[strategy.id] = strategy

    def run(self, data: pd.DataFrame) -> int:
        """
        Esegue tutte le strategie sul nuovo batch di dati e ritorna una decisione aggregata.

        Args:
            data (pd.DataFrame): Il batch di dati su cui fare predizioni.

        Returns:
            int: +1 per comprare, -1 per vendere, 0 per hold.
        """
        if not self.strategies:
            raise ValueError("Nessuna strategia inserita nel benchmark.")

        short_preds = []
        long_preds = []

        for strategy_id, strategy in self.strategies.items():
            prediction = strategy.predict(data)
            if strategy_id.startswith("short"):
                short_preds.append(prediction)
            elif strategy_id.startswith("long"):
                long_preds.append(prediction)

        any_short_signal = any(p == 1 for p in short_preds)
        any_long_signal = any(p == 1 for p in long_preds)

        if any_short_signal and not any_long_signal:
            return -1
        elif any_long_signal and not any_short_signal:
            return 1
        else:
            return 0