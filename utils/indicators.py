# region imports
from AlgorithmImports import *
# endregion

import pandas as pd
from typing import Optional
from abc import ABC
from utils.rules import *
    

class CustomIndicator(ABC):
    """
    Base class per tutti gli indicatori: impone .update(df) → risultato
    """
    def Update(self, df: pd.DataFrame) -> float | None:
        pass


class PriceEmaDifferenceIndicator(CustomIndicator):
    """
    Calcola la differenza percentuale tra l'ultimo prezzo di chiusura
    e l'EMA su una finestra di `ma_period` barre.
    
    Valore positivo → prezzo sopra la EMA.
    Valore negativo → prezzo sotto la EMA.
    """
    def __init__(self, ma_period: int) -> None:
        self.ma_period = ma_period
        self.last_diff = None
        self.last_ema = None
        self.last_close = None

    def Update(self, window: pd.DataFrame) -> Optional[float]:
        if not isinstance(window, pd.DataFrame):
            return None
            
        # Cerca la colonna 'close'
        if "close" in window:
            closes = window["close"]
        elif "Close" in window:
            closes = window["Close"]
        else:
            raise KeyError("Nessuna colonna 'close' o 'Close' trovata.")

        if len(closes) == 0:
            return None

        # Calcola EMA sull'ultima finestra
        ema_series = closes.ewm(span=self.ma_period, adjust=False).mean()
        self.last_ema = ema_series.iloc[-1]
        self.last_close = closes.iloc[-1]

        # Differenza percentuale
        self.last_diff = (self.last_close - self.last_ema) / self.last_ema
        return self.last_diff


class TrendlineBreakoutIndicator(CustomIndicator):
    def __init__(
        self,
        finder: TrendlineFinder,
        lookback: Optional[int] = None,
        tol: float = 1e-5,
        bos_lookback: int = 5
    ) -> None:
        """
        finder:    TrendlineFinder già configurato (support o resistance)
        lookback:  quante barre guardare (default = finder.lookback)
        tol:       tolleranza in unità di prezzo sul breakout
        """
        super().__init__()
        self.finder   = finder
        self.lookback = lookback or finder.lookback
        self.tol      = tol
        self.Current  = 0.0  # ultimo segnale (0 or 1)
        self.bos_lookback = bos_lookback


    def Update(self, df: pd.DataFrame) -> float:
        """
        Ritorna 1.0 se si è verificato un breakout sull'ultima barra,
        0.0 altrimenti.
        """
        n = len(df)
        # non abbiamo abbastanza dati
        if n < self.lookback + 1:
            self.Current = 0.0
            return self.Current

        # prendiamo l'ultima finestra di lookback barre
        window_df = df.iloc[-self.lookback:]

        # calcoliamo la trendline migliore
        res = self.finder.find_best_trendline(window_df)
        if res is None:
            self.Current = 0.0
            return self.Current

        slope, intercept = res

        # proiettiamo il valore della linea sulla (lookback)-esima barra
        # ossia sull'indice x = lookback - 1
        x = self.lookback - 1
        y_line = slope * x + intercept

        # prendiamo il prezzo effettivo dell'ultima barra
        sig = 0.0
        for i in range(1, self.bos_lookback):
            last_low   = df["low"].iloc[-i]
            last_high  = df["high"].iloc[-i]

            
            if self.finder.kind == "resistance":
                # breakout rialzista: il minimo supera la linea + tol
                if last_low > y_line + self.tol:
                    sig = 1.0
            else:  # support
                # breakout ribassista: il massimo scende sotto la linea - tol
                if last_high < y_line - self.tol:
                    sig = 1.0

        self.Current = sig
        return sig


class TrendlineSlopeIndicator(CustomIndicator):
    """
    Ritorna l’inclinazione (slope) della trend‑line di supporto
    o resistenza calcolata da `TrendlineFinder` sull’ultima finestra.

    ▸ Current  → slope (float)     | None se non disponibile
    """

    def __init__(
        self,
        finder: TrendlineFinder,
        lookback: Optional[int] = None
    ) -> None:
        super().__init__()
        self.finder   = finder
        self.lookback = lookback or finder.lookback
        self.Current  = None        # slope dell’ultimo update

    def Update(self, df: pd.DataFrame) -> Optional[float]:
        """
        Calcola e salva in `self.Current` la slope della trend‑line;
        se non c’è abbastanza storia o la linea non è valida -> None.
        """
        if len(df) < self.lookback:
            self.Current = None
            return self.Current

        window_df = df.iloc[-self.lookback:]

        result = self.finder.find_best_trendline(window_df)
        if result is None:
            self.Current = None
        elif isinstance(result, tuple):
            slope, _intercept = result
            self.Current = float(slope)
        else:                       # caso kind="both": result è un dict
            # per coerenza scegliamo di salvare comunque due slope
            # (potresti anche suddividerlo in due indicatori separati)
            self.Current = {
                "support":    float(result["support"][0]),
                "resistance": float(result["resistance"][0])
            }

        return self.Current

    def check(self, df: pd.DataFrame) -> bool:
        """
        Torna True se nell’ultima finestra è stata calcolata una slope.
        """
        return self.Update(df) is not None


class WickReversalIndicator(CustomIndicator):
    """
    Rileva l’ultimo pivot (massimo o minimo locale) in una finestra di `lookback`
    e restituisce:

        ▸ se kind="high"  → lunghezza upper–wick  = high – max(open, close)
        ▸ se kind="low"   → lunghezza lower–wick  = min(open, close) – low

    Il valore è espresso in unità di prezzo; se non si trova alcun pivot
    nell’intervallo restituisce None.
    """

    def __init__(
        self,
        kind: Literal["high", "low"] = "high",
        lookback: int = 50,
        locality: int = 3           # ampiezza del “vicinato” per il pivot
    ) -> None:
        if kind not in ("high", "low"):
            raise ValueError("kind deve essere 'high' o 'low'")
        self.kind     = kind
        self.lookback = lookback
        self.locality = locality
        self.Current  = None        # ultimo valore calcolato

    def Update(self, df: pd.DataFrame) -> Optional[float]:
        """
        df deve avere colonne: open, high, low, close  e indice ordinato.

        Ritorna lunghezza wick (float) oppure None se non trovato.
        """
        if len(df) < self.lookback:
            self.Current = None
            return self.Current

        window = df.iloc[-self.lookback:]
        n      = len(window)

        # ------------------------------------------------ pivot search
        last_pivot_idx = None
        if self.kind == "high":
            # scorriamo dall’ultima barra all’indietro
            for i in range(n - self.locality, self.locality - 1, -1):
                seg = window.iloc[i - self.locality : i + self.locality + 1]
                if window.high.iat[i] == seg.high.max():
                    last_pivot_idx = i
                    break
        else:  # kind == "low"
            for i in range(n - self.locality, self.locality - 1, -1):
                seg = window.iloc[i - self.locality : i + self.locality + 1]
                if window.low.iat[i] == seg.low.min():
                    last_pivot_idx = i
                    break

        if last_pivot_idx is None:          # nessun pivot trovato
            self.Current = None
            return self.Current

        c = window.iloc[last_pivot_idx]

        if self.kind == "high":             # upper‑wick
            wick = c.high - max(c.open, c.close)
        else:                               # lower‑wick
            wick = min(c.open, c.close) - c.low

        self.Current = float(wick)
        return self.Current

    def check(self, df: pd.DataFrame) -> bool:
        return (self.Update(df) or 0.0) > 0.0

class FVGSizeIndicator(CustomIndicator):
    """
    Restituisce la *dimensione* (gap_high − gap_low) dell’ultimo
    Fair‑Value‑Gap (FVG) valido individuato da un’istanza di `FVGRule`.

    • Se non esiste un FVG che soddisfi la regola ⇒ `None`
    • Altrimenti ⇒ float (ampiezza del gap, stessa unità dei prezzi)

    Parametri facoltativi
    ---------------------
    normalize_by_atr : se True la dimensione è divisa per l’ATR
                       calcolato sulla stessa finestra (valore “percentuale”)
    """
    def __init__(
        self,
        fvg_rule: FVGRule,
        normalize_by_atr: bool = False,
        atr_period: int | None = None
    ) -> None:
        self.rule              = fvg_rule              # oggetto già configurato
        self.norm              = normalize_by_atr
        self.atr_period        = atr_period or fvg_rule.lookback
        self.Current: float | None = None              # ultimo valore calcolato

    # ------------------------------------------------------------ #
    def Update(self, df: pd.DataFrame) -> float | None:            # stile QC
        """
        df deve contenere almeno `fvg_rule.lookback` barre con colonne
        open, high, low, close.
        """
        look = self.rule.lookback
        if len(df) < look:
            self.Current = None
            return self.Current

        window = df.tail(look)

        # 1) trova tutti i gap (già filtrati per body‑multiplier ecc.)
        gaps = self.rule.detect_fvg(window)
        if not gaps:
            self.Current = None
            return self.Current

        # 2) prendi il PIÙ RECENTE che rispetta direzione/bias
        last_gap = None
        for g in reversed(gaps):
            if self.rule.direction == "both" or g["type"] == self.rule.direction:
                last_gap = g
                break
        if last_gap is None:                       # nessun gap nel bias richiesto
            self.Current = None
            return self.Current

        size = last_gap["gap_high"] - last_gap["gap_low"]

        # 3) opzionale: normalizza con ATR → dimensione “in ATR”
        if self.norm:
            sub = df.tail(self.atr_period)
            atr = ta.volatility.average_true_range(
                high   = sub["high"],
                low    = sub["low"],
                close  = sub["close"],
                window = self.atr_period
            ).iloc[-1]
            size = size / (atr or 1e-8)            # evita div./0

        self.Current = float(size)
        return self.Current

    # ------------------------------------------------------------ #
    def check(self, df: pd.DataFrame) -> bool:                      # interfaccia Rule‑like
        """True se esiste un FVG e la sua dimensione è > 0."""
        return (self.Update(df) or 0.0) > 0.0