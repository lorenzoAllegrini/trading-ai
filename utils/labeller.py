# region imports
from AlgorithmImports import *
# endregion

from typing import Optional
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from utils.rules import *

@dataclass
class BaseStrategy:
    id: str
    rules: List[Rule] = field(default_factory=list)
    bias: Literal["bullish", "bearish", "both"] = "both"
class Labeller:
    """
    Etichetta 1 quando:
      • tutte le regole sono vere al tempo T
      • la variazione entro `lookahead` barre è ≥ move_threshold × avg_body5
    Elimina i dati che non rispettano la regola base.
    """

    def __init__(self,
        strategy: BaseStrategy,
        lookahead: int = 10,
        move_threshold: float = 1.0,
        sl:bool = False,
        sl_locality:int=5,
        remove:bool = False,
        slcoef: float = 0.2,
        tpsl_ratio: float = 2.0,
        atr_col: str = "atr",
        ) -> None:
        self.strategy = strategy
        self.lookahead = lookahead       
        self.move_threshold = move_threshold
        self.lookback = 2000
        self.sl = sl
        self.sl_locality = sl_locality
        self.remove = remove
        self.slcoef    = slcoef
        self.tpsl_ratio = tpsl_ratio
        self.atr_col    = atr_col

    def _required_move(self) -> float:
        """Soglia assoluta configurata dall’utente (in unità di prezzo)."""
        return float(self.move_threshold)
        

    def label_one(self, window: pd.DataFrame, future_close: pd.Series) -> Optional[int]:
        """
        Restituisce
        • None  se una rule di base è False (opzione remove=True)
        • 1     se TP colpito prima dello SL
        • 0     altrimenti
        """
        # ---------------- 0) filtro regole base ----------------
        if not all(rule.check(window) for rule in self.strategy.rules):
            return None if self.remove else 0

        # ---------------- 1) ATR e parametri dinamici ----------
        atr = window[self.atr_col].iat[-1]
        if np.isnan(atr) or atr == 0:
            return 0                           # nessuna informazione → scarta

        sl_points = self.slcoef * atr
        tp_points = sl_points * self.tpsl_ratio

        entry = window["close"].iat[-1]

        # direzione trade
        bias = self.strategy.bias
        is_long  = bias in ("bullish", "both")
        is_short = bias in ("bearish", "both")

        # ---------------- 2) simula esito nella future_window --
        if future_close.empty:
            return 0

        # Long --------------------------------------------------
        if is_long:
            tp_hit = (future_close >= entry + tp_points)
            sl_hit = (future_close <= entry - sl_points)

            # prima occorrenza
            idx_tp = tp_hit.idxmax() if tp_hit.any() else None
            idx_sl = sl_hit.idxmax() if sl_hit.any() else None

            if idx_tp is not None and (idx_sl is None or idx_tp < idx_sl):
              
                return 1
            return 0

        # Short -------------------------------------------------
        if is_short:
            tp_hit = (future_close <= entry - tp_points)
            sl_hit = (future_close >= entry + sl_points)

            idx_tp = tp_hit.idxmax() if tp_hit.any() else None
            idx_sl = sl_hit.idxmax() if sl_hit.any() else None

            if idx_tp is not None and (idx_sl is None or idx_tp < idx_sl):
                return 1
            return 0

        # se bias == "both" ma non specificato quale direzione → fallback
        return 0
    
    def _stop_loss_level(self, window: pd.DataFrame) -> Tuple[float, float]:
        """
        Ritorna (sl_long, sl_short):
            • sl_long  = minimo dei LOW nelle ultime sl_locality barre
            • sl_short = massimo dei HIGH   “     ”
        """
        sub = window.tail(self.sl_locality)
        return sub["low"].min(), sub["high"].max()

    def label_df(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        labels = [None] * n

        # determina il massimo lookback di tutte le rule
        max_lookback = max(rule.lookback for rule in self.strategy.rules) #modificare: massimo anche rispetto al lookback degli indicatori

        # se usi anche lookahead per filtrare future_window, potresti aggiungere anche quello
        warmup = max_lookback + 2
   
        for i in tqdm(range(n), desc="Labeling"):
            # finché siamo nel warmup, lasciamo label = None
       
            if i < warmup:
                labels[i] = None
                continue

            # sliding window fino a max_lookback barre
            start = i - self.lookback
            window = df.iloc[start : i + 1]

            # future window
            future_start = i + 1
            future_end   = min(n, i + 1 + self.lookahead)
            future_window = df["close"].iloc[future_start : future_end]

            labels[i] = self.label_one(window, future_window)

        # bring the old index into columns, rename first two → level_0, level_1
        out = df.copy().reset_index()
        out.rename(
            columns={
                out.columns[0]: "level_0",
                out.columns[1]: "level_1"
            },
            inplace=True
        )

        out["label"] = labels
   
        return out