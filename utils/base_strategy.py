from AlgorithmImports import *
# endregion

from dataclasses import dataclass, field
from typing import List, Protocol, Literal
import pandas as pd

class Rule(Protocol):
    """Interface: una regola restituisce True/False sui dati passati."""
    def check(self, window: pd.DataFrame) -> bool: ...

@dataclass
class BaseStrategy:
    id: str
    rules: List[Rule] = field(default_factory=list)
    bias: Literal["bullish", "bearish", "both"] = "both"

import time
from dataclasses import dataclass, field
from typing import List, Protocol, Literal
import pandas as pd
from abc import ABC, abstractmethod
from ta.volatility import average_true_range
import scipy

class Rule(Protocol):
    """Interface: una regola restituisce True/False sui dati passati."""
    def check(self, window: pd.DataFrame) -> bool: ...

@dataclass
class BaseStrategy:
    id: str
    rules: List[Rule] = field(default_factory=list)
    bias: Literal["bullish", "bearish", "both"] = "both"

class CustomIndicator(ABC):
    """
    Base class per tutti gli indicatori: impone .update(df) → risultato
    """
    def Update(self, df: pd.DataFrame) -> Any:
        pass

class FVGRule:
    def __init__(
        self,
        lookback: int = 20,
        must_retest: bool = False,
        direction: Literal["bullish", "bearish", "both"] = "bullish",
        body_multiplier: float = 1.5,
        retest_lookback: int = 4
    ):
        self.lookback = lookback
        self.must_retest = must_retest
        self.direction = direction
        self.body_multiplier = body_multiplier
        self.retest_lookback = retest_lookback

    def detect_fvg(self, data: pd.DataFrame) -> List[dict]:
        df = data.tail(self.lookback).reset_index(drop=True)
        fvg_list: List[Dict[str, Any]] = []
        for i in range(2, len(df)):
            c0, c1, c2 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
            avg_body = (
                (df["close"].iloc[max(0, i - 1 - self.lookback): i - 1]
                 - df["open"].iloc[max(0, i - 1 - self.lookback): i - 1])
                .abs()
                .mean()
                or 1e-4
            )
            mid_body = abs(c1.close - c1.open)
            if mid_body < avg_body * self.body_multiplier:
                continue

            # bullish gap
            if c2.low > c0.high:
                fvg_list.append({
                    "type": "bullish",
                    "gap_low":  c0.high,
                    "gap_high": c2.low,
                    "index":    data.index[-len(df) + i]
                })
            # bearish gap
            elif c2.high < c0.low:
                fvg_list.append({
                    "type": "bearish",
                    "gap_low":  c2.high,
                    "gap_high": c0.low,
                    "index":    data.index[-len(df) + i]
                })

        return fvg_list

    def check(self, window: pd.DataFrame) -> bool:
        # servono almeno 3 barre per formare un gap
        window = window[-self.lookback:]
        if len(window) < 3:
            return False

        gaps = self.detect_fvg(window)
        if not gaps:
            return False

        last_close = window["close"].iloc[-1]

        # ciclo dal più recente
        for gap in reversed(gaps):
            t = gap["type"]
            # filtro per bias
            if self.direction != "both" and t != self.direction:
                continue

            low_b, high_b = gap["gap_low"], gap["gap_high"]
            creation_idx = gap["index"]

            # barra di creazione → tutte le barre dall’indice creato fino all’ultima
            sub = window.loc[creation_idx:]

            # 1) invalidazione: controlla separatamente per bullish o bearish
            if t == "bullish":
                # il gap è invalidato se qualcuno ha LOW ≤ gap_low
                if (sub["low"] <= low_b).any():
                    continue
            else:  # bearish
                # invalidato se qualcuno ha HIGH ≥ gap_high
                if (sub["high"] >= high_b).any():
                    continue

            # 2) se non serve retest, è già OK
            if not self.must_retest:
               
                return True

            # 3) breakout: prima dell’ultima barra, almeno una barra FUORI dal gap
            breakout_window = sub.iloc[:-1]
            if t == "bullish":
                if not (breakout_window["low"] > high_b).any():
                    continue
            else:
                if not (breakout_window["high"] < low_b).any():
                    continue

            # 4) retest: nelle ultime self.retest_lookback barre (escl. ultima)
            candidates = (
                sub.iloc[-self.retest_lookback-1:-1]
                if len(sub) > self.retest_lookback
                else sub.iloc[:-1]
            )
            if t == "bullish":
                if not candidates["low"].between(low_b, high_b).any():
                    continue
            else:
                if not candidates["high"].between(low_b, high_b).any():
                    continue

            # 5) ultimo prezzo di nuovo FUORI (al rialzo per bullish, al ribasso per bearish)
            if t == "bullish" and last_close > high_b:
   
                return True
            if t == "bearish" and last_close < low_b:

                return True

        return False



class BosIndicator(CustomIndicator):
    def __init__(
        self,
        lookback: int = 150,
        locality: int = 5,
        direction: Literal["bullish", "bearish", "both"] = "bullish",
        key_level_validation: int = 2,
        C: float = 0.3,
        levels_lag: int = 10,
        bos_lookback: int = 5,
        reversal: bool = True
    ):
        self.lookback = lookback
        self.locality = locality
        self.direction = direction
        self.key_level_validation = key_level_validation
        self.C = C
        self.levels_lag = levels_lag
        self.bos_lookback = bos_lookback

    def detect_key_levels(self, data: pd.DataFrame) -> Dict[str, List[Tuple[Any, float]]]:
        """
        Identifica i livelli locali e li valida:
        - serve almeno `key_level_validation` tocchi
        - due tocchi si considerano dello stesso livello se |p1-p2| ≤ C * body del rispettivo livello
        """
        raw_levels = []  # lista di dict {type, index, price, body}
        n = len(data)

        # 1) Rileva tutti i picchi locali
        for i in range(self.locality, n - self.locality):
            window = data.iloc[i - self.locality : i + self.locality + 1]
            idx = data.index[i]
            high, low = data["high"].iloc[i], data["low"].iloc[i]
            body = high - low

            if self.direction in ("bullish", "both") and high == window["high"].max():
                raw_levels.append({"type":"resistance","index":idx,"price":high,"body":body})
            if self.direction in ("bearish", "both") and low == window["low"].min():
                raw_levels.append({"type":"support","index":idx,"price":low,"body":body})

        # 2) Valida i livelli con almeno `key_level_validation` tocchi
        validated = {"support": [], "resistance": []}
        for lvl in raw_levels:
            # conta quanti raw_levels dello stesso type cadono entro C*body da questo livello
            same_type = [o for o in raw_levels if o["type"] == lvl["type"]]
            hits = sum(abs(o["price"] - lvl["price"]) <= lvl["body"] * self.C for o in same_type)
            if hits >= self.key_level_validation:
                validated[lvl["type"]].append((lvl["index"], lvl["price"]))

        # 3) Rimuovi eventuali duplicati (stesso prezzo, stesso index)
        for t in ("support","resistance"):
            seen = set()
            uniq = []
            for idx, price in validated[t]:
                if (idx, price) not in seen:
                    seen.add((idx, price))
                    uniq.append((idx, price))
            validated[t] = uniq

        return validated


    def check(self, data: pd.DataFrame) -> Tuple[bool, Optional[Tuple[Any, float]]]:
        """
        Ritorna (True, (idx, level_price)) se l'ultima candela
        rompe completamente un key level, altrimenti (False, None).
        Per "completamente" si intende che tutto il corpo
        (min(open,close) > level_price per bullish,
         max(open,close) < level_price per bearish).
        """
        key_levels = self.detect_key_levels(data[:-self.levels_lag])
        last = data.iloc[-1]
        body_low  = min(last["open"], last["close"])
        body_high = max(last["open"], last["close"])

        if self.direction in ("bullish", "both"):
            for lvl_idx, lvl_price in key_levels["resistance"]:
                if body_low > lvl_price:
                    return True

        if self.direction in ("bearish", "both"):
            for lvl_idx, lvl_price in key_levels["support"]:
                if body_high < lvl_price:
                    return True

        return False




class KeyLevelsIndicator(CustomIndicator):
    def __init__(
        self,
        kind: Literal["support","resistance"],   # ora definiamo anche il tipo
        lookback: int = 150,
        first_w: float = 0.1,
        atr_mult: float = 3.0,
        prom_thresh: float = 0.05,
        C: float = 0.0001,
        break_lookback: int = 3,
        locality: int = 5,
        key_level_validation: int = 2,
        num_levels: int = 1
    ):
        """
        kind                 : 'support' o 'resistance'
        lookback             : barre su cui costruire il market-profile
        first_w              : peso del prezzo più vecchio
        atr_mult             : moltiplicatore ATR per bandwidth KDE
        prom_thresh          : soglia prominenza
        C                    : tolleranza breakout in unità di prezzo
        break_lookback       : quante barre all’indietro guardare per breakout
        locality             : (se usassi la logica pivot) raggio per pivot locali
        key_level_validation : tocchi minimi per convalidare un pivot
        """
        self.kind = kind
        self.lookback = lookback
        self.first_w = first_w
        self.atr_mult = atr_mult
        self.prom_thresh = prom_thresh
        self.C = C
        self.break_lookback = break_lookback
        self.locality = locality
        self.key_level_validation = key_level_validation
        self.num_levels            = num_levels

    def _merge_levels_with_prom(
        self,
        peaks: np.ndarray,
        price_grid: np.ndarray,
        prominences: np.ndarray,
        tol: float
    ) -> list[dict]:
        clusters: list[dict] = []
        raw = sorted(
            [
                {
                  "price": np.exp(price_grid[p]),
                  "prom": prominences[i]
                }
                for i, p in enumerate(peaks)
            ],
            key=lambda x: x["price"]
        )
        for lvl in raw:
            if not clusters:
                clusters.append({
                    "price": lvl["price"],
                    "count": 1,
                    "prom_sum": lvl["prom"]
                })
                continue
            last = clusters[-1]
            if abs(lvl["price"] - last["price"]) <= tol:
                total = last["price"] * last["count"] + lvl["price"]
                last["count"] += 1
                last["price"] = total / last["count"]
                last["prom_sum"] += lvl["prom"]
            else:
                clusters.append({
                    "price": lvl["price"],
                    "count": 1,
                    "prom_sum": lvl["prom"]
                })
        return clusters

    def detect_key_levels(self, data: pd.DataFrame) -> list[dict]:
        # 0) warm-up guard
        if len(data) < self.lookback:
            return []

        window = data.tail(self.lookback)
        # 1) ATR
        atr_series = average_true_range(
            high   = window["high"],
            low    = window["low"],
            close  = window["close"],
            window = self.lookback
        )
        atr = atr_series.iloc[-1]

        # 2) log-close + pesi + KDE
        prices = np.log(window["close"].to_numpy())
        n      = len(prices)
        w_step = (1.0 - self.first_w) / n
        weights= np.clip(self.first_w + np.arange(n)*w_step, 0.0, 1.0)

        kernel = scipy.stats.gaussian_kde(
            prices,
            bw_method = atr * self.atr_mult,
            weights   = weights
        )

        # 3) grid e picchi
        mn, mx    = prices.min(), prices.max()
        grid      = np.linspace(mn, mx, 200)
        pdf       = kernel(grid)
        prom_min  = pdf.max() * self.prom_thresh
        peaks, props = scipy.signal.find_peaks(pdf, prominence=prom_min)

        # 4) clustering
        lin       = np.exp(grid)
        tol_merge = self.prom_thresh * (lin.max() - lin.min())
        clusters  = self._merge_levels_with_prom(
            peaks       = peaks,
            price_grid  = grid,
            prominences = props["prominences"],
            tol         = tol_merge
        )

        # 5) tests e score
        tol_test = self.C
        closes   = window["close"]
        for c in clusters:
            lvl_price     = c["price"]
            c["tests"]    = int(((closes - lvl_price).abs() <= tol_test).sum())
            c["score"]    = c["prom_sum"]

        # 6) ordino per score decrescente
        clusters = sorted(clusters, key=lambda x: x["score"], reverse=True)

        # 7) applico il filtro num_levels in base a kind
        if self.num_levels is not None and clusters:
            # ordino i livelli solo per prezzo
            if self.kind == "support":
                # prendo i più bassi
                clusters = sorted(clusters, key=lambda x: x["price"])[: self.num_levels]
            else:  # resistance
                # prendo i più alti
                clusters = sorted(clusters, key=lambda x: x["price"], reverse=True)[: self.num_levels]

        return clusters

    def Update(self, data: pd.DataFrame) -> float:
        """
        Controlla nelle ultime `break_lookback` barre se c’è stato breakout:
          - resistance: high/close entro ±C di un level
          - support   : low/close  entro ±C di un level

        Restituisce (score, price) del primo breakout, 
        oppure (0.0, None) se nessuno trovato.
        """
        
        key_levels = self.detect_key_levels(data)
      
        if len(key_levels) == 0:
            return 0.0
    
        for i in range(1, self.break_lookback + 1):
            bar = data.iloc[-i]
            for lvl in key_levels:
                price = lvl["price"]
                if self.kind == "resistance":
                    if abs(bar["high"] - price) <= self.C \
                    or abs(bar["close"] - price) <= self.C:
                        return lvl["score"]

                else:  # support
                    if abs(bar["low"] - price) <= self.C \
                    or abs(bar["close"] - price) <= self.C:
                        return lvl["score"]

        return 0.0
    
    def check(self, data: pd.DataFrame):

        res = self.Update(data)
    
        return res > 0.0

    def check_breakout(self, data: pd.DataFrame) -> Tuple[bool, Optional[Tuple[pd.Timestamp, float]]]:
        """
        True + (idx,price) se l'ultima candela rompe un livello:
        - bullish: close_low > resistance_price
        - bearish: close_high < support_price
        """
        window = data.tail(self.lookback)
        levels = self.detect_key_levels(window)
        last = window.iloc[-1]
        low_body  = min(last["open"], last["close"])
        high_body = max(last["open"], last["close"])

        # verifica resistenza (rialzo)
        if self.direction in ("bullish","both"):
            for idx, price in levels["resistance"]:
                if low_body > price:
                    return True, (idx, price)

        # verifica supporto (ribasso)
        if self.direction in ("bearish","both"):
            for idx, price in levels["support"]:
                if high_body < price:
                    return True, (idx, price)

        return False, None



class PriceEmaDifferenceIndicator(CustomIndicator):
    """
    Calcola la differenza percentuale tra l'ultimo prezzo di chiusura
    e l'EMA su una finestra di `ma_period` barre.
    
    Valore positivo → prezzo sopra la EMA.
    Valore negativo → prezzo sotto la EMA.
    """
    def __init__(self, ma_period: int):
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

import math

class PriceEmaDifferenceRule:
    """
    Regola su PriceEmaDifferenceIndicator.

    direction: "positive" (prezzo sopra EMA), 
               "negative" (prezzo sotto EMA),
               "both"     (esorso assoluto >= threshold)
    threshold: differenza percentuale minima (es. 0.02 = 2%)
    """
    def __init__(
        self,
        indicator: PriceEmaDifferenceIndicator,
        direction: Literal["positive", "negative", "both"],
        threshold: float,
    
    ):
        if direction not in ("positive", "negative", "both"):
            raise ValueError("direction deve essere 'positive', 'negative' o 'both'")
        self.indicator = indicator
        self.direction = direction
        self.threshold = threshold
        self.last_diff: Optional[float] = None
        self.lookback = self.indicator.ma_period

    def check(self, window: pd.DataFrame) -> bool:
        """
        Restituisce True se la differenza EMA meets la condizione impostata.
        """
        diff = self.indicator.Update(window)
        self.last_diff = diff

        if diff is None:
            return False

        if self.direction == "positive":
            return diff >= self.threshold
        elif self.direction == "negative":
            return diff <= -self.threshold
        else:  # both
            return abs(diff) >= self.threshold

from itertools import combinations





class TrendlineFinder:
    """
    Trova, su una finestra di `lookback` barre, la miglior trendline di:
     - supporto   (kind="support")
     - resistenza (kind="resistance")
     - entrambe    (kind="both") → ritorna un dict con le due linee
    """
    def __init__(
        self,
        kind: Literal["support","resistance","both"] = "support",
        lookback: int = 300,
        C: float = 0.0005,
        min_touches: int = 2
    ):
        if kind not in ("support","resistance","both"):
            raise ValueError("kind must be 'support','resistance' or 'both'")
        self.kind        = kind
        self.lookback    = lookback
        self.C           = C
        self.min_touches = min_touches

        # ultimo risultato
        self.last_slope     = None
        self.last_intercept = None

    def find_best_trendline(
        self,
        df: pd.DataFrame
    ) -> Optional[Tuple[float,float]]:
        """
        Analizza gli ultimi `lookback` record di df[["high","low","close"]],
        calcola support & resistance coefs e poi ritorna:
         - (slope, intercept) per il self.kind scelto
         - oppure None se non si trova nulla
        """
        data = df.iloc[-self.lookback :]

        highs  = data["high"].values
        lows   = data["low"].values
        closes = data["close"].values

        # fit delle due linee
        (sup_coef, res_coef) = self.fit_trendlines_high_low(highs, lows, closes)
        # sup_coef = (slope_sup, int_sup), res_coef = (slope_res, int_res)

        out = None
        if self.kind == "support":
            out = sup_coef
        elif self.kind == "resistance":
            out = res_coef
        else:  # both
            out = {"support": sup_coef, "resistance": res_coef}

        # se out è una tupla valida, salvalo
        if isinstance(out, tuple):
            slope, intercept = out
            self.last_slope, self.last_intercept = slope, intercept

        return out

    def fit_trendlines_high_low(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Tuple[Tuple[float,float], Tuple[float,float]]:
        """
        1) fit lineare su `close` per trovare pivot upper/lower
        2) ottimizza slope/intercept su high, low usando optimize_slope
        Ritorna ((s_sup, i_sup), (s_res, i_res))
        """
        x = np.arange(len(close))
        slope0, intercept0 = np.polyfit(x, close, 1)
        line0 = slope0*x + intercept0

        upper_pivot = int((high - line0).argmax())
        lower_pivot = int((low  - line0).argmin())

        sup_coef = self.optimize_slope(True,  lower_pivot, slope0, low)
        res_coef = self.optimize_slope(False, upper_pivot, slope0, high)
        return sup_coef, res_coef

    def optimize_slope(
        self,
        support: bool,
        pivot: int,
        init_slope: float,
        y: np.ndarray
    ) -> Tuple[float,float]:
        """
        Discesa numerica per minimizzare l'errore sui punti `y`:
         - mantiene la linea valida (mai oltrepassa i prezzi)
         - minimizza la somma dei quadrati delle differenze
        Ritorna (best_slope, best_intercept)
        """
        # passo base proporzionale all'ampiezza
        slope_unit = (y.max() - y.min()) / len(y)
        best_slope = float(init_slope)
        best_err   = self.check_trend_line(support, pivot, best_slope, y)
        step       = 1.0

        while step > 1e-4:
            # numerically estimate gradient
            err_up = self.check_trend_line(support, pivot, best_slope + slope_unit*1e-4, y)
            grad = err_up - best_err

            # tentativo di discesa
            delta = -step * (1 if grad>0 else -1)
            candidate = best_slope + delta*slope_unit
            err_cand  = self.check_trend_line(support, pivot, candidate, y)

            if err_cand>=0 and err_cand < best_err:
                best_err   = err_cand
                best_slope = candidate
            else:
                step *= 0.5

        best_int = -best_slope*pivot + y[pivot]
        return best_slope, best_int

    def check_trend_line(
        self,
        support: bool,
        pivot: int,
        slope: float,
        y: np.ndarray
    ) -> float:
        """
        Se la linea va fuori dal vincolo (support/resistance), ritorna -1.
        Altrimenti somma quadrati diffs.
        """
        intercept = -slope*pivot + y[pivot]
        x = np.arange(len(y))
        preds = slope*x + intercept
        diffs = preds - y

        if support:
            if diffs.max() > 1e-8: return -1.0
        else:
            if diffs.min() < -1e-8: return -1.0

        return float((diffs**2).sum())


class TrendlineBreakoutIndicator(CustomIndicator):
    def __init__(
        self,
        finder: TrendlineFinder,
        lookback: Optional[int] = None,
        tol: float = 1e-5,
        bos_lookback: int = 5
    ):
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


    def update(self, df: pd.DataFrame) -> float:
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

    def check(self, df: pd.DataFrame) -> bool:
        """
        Restituisce True se c'è stato un breakout nell'ultima barra.
        """
        return self.update(df) > 0


class TrendlineSlopeIndicator(CustomIndicator):
    def __init__(self, trendline_finder: TrendlineFinder, backward_delay:int = 5, direction: Literal["positive", "negative", "both"] = "negative"):
        self.finder = trendline_finder
        self.backward_delay = backward_delay
        self.last_slope = None
        self.last_intercept = None
        self.valid_pivots = 0
        self.direction = direction

    def Update(self, df: pd.DataFrame) -> Optional[float]:
        result = self.finder.find_best_trendline(df[:-self.backward_delay])
        if result is None:
            self.last_slope = 0
            self.last_intercept = None
            self.valid_pivots = 0
            return None

        self.last_slope, self.last_intercept, self.valid_pivots = result
        return self.last_slope

    def check(self, df: pd.DataFrame) -> bool:
        signal = self.Update(df)

        if self.direction == "negative":
            if signal < 0:
                return True
            else:
                return False

        if self.direction == "positive":
            if signal > 0:
                return True
            else:
                return False



class BosIndicator(CustomIndicator):
    """
    CustomIndicator that identifies support/resistance levels
    based on wick pivots and signals breakouts.
    """
    def __init__(
        self,
        lookback: int = 150,
        locality: int = 20,
        key_level_validation: int = 2,
        direction: Literal["bullish", "bearish", "both"] = "bullish",
        C: float = 0.001,
        wick_threshold: float = 0.001,
        level_backCandles: int = 6,
        close_lim: float = 150e-5,
        n_levels: Optional[int] = None,
        breakout_lookback: int = 5,
        levels_lag: int = 10
    ):
        self.lookback = lookback
        self.locality = locality
        self.key_level_validation = key_level_validation
        self.direction = direction
        self.C = C
        self.wick_threshold = wick_threshold
        self.level_back = level_backCandles
        self.close_lim = close_lim
        self.n_levels = n_levels
        self.break_lookback = breakout_lookback
        self.levels_lag = levels_lag

    
    def detect_key_levels(self, df: pd.DataFrame) -> Dict[str, List[Tuple[pd.Timestamp, float]]]:
        """
        Identifica key levels che:
         - sono pivot locali (high per resistenza, low per supporto)
         - sono toccati almeno 2 volte entro la finestra
         - non vengono mai invalidati (nessuna rottura sopra/sotto) dalla prima al current bar
        """
        n = len(df)
        # memorizza per ogni livello (price) gli indici di pivot che lo toccano
        touches: Dict[float, List[int]] = {}

        # 1) trova tutti i pivot locali e raggruppa per prezzo approssimato
        for i in range(self.locality, n - self.locality):
            # finestre locali
            win_h = df.high.iloc[i - self.locality : i + self.locality + 1]
            win_l = df.low.iloc[i - self.locality : i + self.locality + 1]

            # pivot resistenza
            if self.direction in ("bullish","both") and df.high.iloc[i] == win_h.max():
                price = df.high.iloc[i]
                touches.setdefault(price, []).append(i)
            # pivot supporto
            if self.direction in ("bearish","both") and df.low.iloc[i] == win_l.min():
                price = df.low.iloc[i]
                touches.setdefault(price, []).append(i)

        validated: Dict[str, List[Tuple[pd.Timestamp, float]]] = {"resistance": [], "support": []}

        # 2) per ogni candidato, controlla tocchi >=2 e no invalidazione
        for price, inds in touches.items():
            if len(inds) < self.key_level_validation:
                continue
            first = min(inds)
            # tolleranza proporzionale
            tol = price * self.C
            # determina tipo da primo pivot
            typ = "resistance" if price in df.high.values else "support"
            # invalidazione: da first al termine
            if typ == "resistance":
                if (df.high.iloc[first:] > price + tol).any():
                    continue
                validated["resistance"].append((df.index[first], price))
            else:
                if (df.low.iloc[first:] < price - tol).any():
                    continue
                validated["support"].append((df.index[first], price))

        # 3) ordina e limita il numero se richiesto
        if self.n_levels is not None:
            if validated["resistance"]:
                validated["resistance"] = sorted(validated["resistance"], key=lambda x: x[1], reverse=True)[:self.n_levels]
            if validated["support"]:
                validated["support"] = sorted(validated["support"], key=lambda x: x[1])[:self.n_levels]

        return validated


    def merge(self,pivots):
        pivots = sorted(pivots, key=lambda x: x[1])
        merged: List[Tuple[int,float]] = []
        for l,p in pivots:
            if not merged or abs(p-merged[-1][1])>self.C*p:
                merged.append((l,p))
        return merged



    def check(self, data: pd.DataFrame) -> Tuple[bool, Optional[Tuple[Any, float]]]:
        """
        Ritorna (True, (idx, level_price)) se l'ultima candela
        rompe completamente un key level, altrimenti (False, None).
        Per "completamente" si intende che tutto il corpo
        (min(open,close) > level_price per bullish,
         max(open,close) < level_price per bearish).
        """
        n = len(data)
        if n <= self.levels_lag:
            return False
        key_levels = self.detect_key_levels(data[-self.lookback:-self.levels_lag])
        
        last = data.iloc[-1]
 
        body_low  = min(last["open"], last["close"])
        body_high = max(last["open"], last["close"])
        if self.direction in ("bullish", "both"):
            for lvl_idx, lvl_price in key_levels["resistance"]:
                if body_low > lvl_price:
                    print(f"body_low: {body_low}, price: {lvl_price}")
                    return True

        if self.direction in ("bearish", "both"):
            for lvl_idx, lvl_price in key_levels["support"]:
                if body_high < lvl_price:
                    return True
        return False
    

class FvgBosRule:
    def __init__(
        self,
        fvg_rule: FVGRule,
        bos_rule: BosIndicator,
        tol_factor: float = 2.0
    ):
        self.fvg = fvg_rule
        self.bos = bos_rule
        self.tol_factor = tol_factor
        self.lookback = 205

    def check(self, data: pd.DataFrame) -> bool:
        # 1) Prendi abbastanza barre
        look = max(self.fvg.lookback, self.bos.lookback)
        window = data.tail(look)

        # 2) Trova resistenze BOS
        key_res = self.bos.detect_key_levels(window).get("resistance", [])
        
        if not key_res:
            return False
 
        # 3) Trova gap bullish
        gaps = self.fvg.detect_fvg(window)
        if not gaps:
            return False

        # 4) Cicla sui gap (dal più recente)
        for gap in reversed(gaps):
            if gap["type"] != "bullish":
                continue

            low_b, high_b = gap["gap_low"], gap["gap_high"]
            # sottofinestra da barra di creazione in poi
            sub = window.loc[gap["index"]:]

            # 4a) invalidazione: nessuna barra deve avere low <= gap_low
            if (sub["low"] <= low_b).any():
                continue

            # 5) “Appena sopra” una resistenza BOS
            for lvl_idx, lvl_price in key_res:
                # ricavo il body di quel livello
                c = window.loc[lvl_idx]
                body = abs(c["high"] - c["low"])
                tol  = self.tol_factor * body

                # gap_low deve cadere tra (lvl_price, lvl_price+tol]
                if not (lvl_price < low_b <= lvl_price + tol):
                    continue

                # 6) se non serve retest, abbiamo il segnale
                if not self.fvg.must_retest:
                    return True

                # 7) breakout: almeno una barra (escl. l'ultima) con low > gap_high
                breakout_window = sub.iloc[:-1]
                if not (breakout_window["low"] > high_b).any():
                    continue

                # 8) retest: nelle ultime retest_lookback barre (escl. l'ultima)
                if len(sub) > self.fvg.retest_lookback:
                    candidates = sub.iloc[-self.fvg.retest_lookback - 1 : -1]
                else:
                    candidates = sub.iloc[:-1]

                # deve capitare un tocco tra low_b e high_b
                if not candidates["low"].between(low_b, high_b).any():
                    continue

                # 9) ultimo close fuori gap (>= high_b)
                last_close = window["close"].iloc[-1]
                if last_close > high_b:
                    return True

        # nessuna combinazione valida
        return False
        


class InverseFVGRule(FVGRule):
    def __init__(self, *args, inverse: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse = inverse

    def detect_inverse_fvg(self, data: pd.DataFrame) -> List[dict]:
        """
        Trasforma ogni gap completamente invalidato in un 'gap inverso'
        da retestare come se fosse nella direzione opposta.
        """
        df = data.tail(self.lookback).reset_index(drop=True)
        orig = self.detect_fvg(data)
        inv = []
        for gap in orig:
            low_b, high_b, idx0 = gap["gap_low"], gap["gap_high"], gap["index"]
            sub = data.loc[idx0:]
            # gap bullish pienamente invaso → diventa bearish inverse
            if gap["type"] == "bullish" and (sub["close"] <= low_b).any():
                inv.append({
                    "type":      "bearish",
                    "gap_low":   low_b,
                    "gap_high":  high_b,
                    "index":     idx0
                })
            # gap bearish pienamente invaso → diventa bullish inverse
            elif gap["type"] == "bearish" and (sub["close"] >= high_b).any():
                inv.append({
                    "type":      "bullish",
                    "gap_low":   low_b,
                    "gap_high":  high_b,
                    "index":     idx0
                })
        return inv

    def check(self, window: pd.DataFrame) -> bool:
        """
        Se inverse=False usa il comportamento standard di FVGRule.check,
        altrimenti applica la logica su detect_inverse_fvg.
        """
        window = window[-self.lookback:]
        if len(window) < 3:
            return False

        gaps = (
            self.detect_inverse_fvg(window)
            if self.inverse
            else self.detect_fvg(window)
        )

        if not gaps:
            return False

        last_close = window["close"].iloc[-1]

        for gap in reversed(gaps):
            t = gap["type"]

            if self.direction != "both" and t != self.direction:
                continue

            low_b, high_b = gap["gap_low"], gap["gap_high"]
            idx0 = gap["index"]
            sub = window.loc[idx0:]

            # 2) breakout (stesso di FVGRule)
            breakout_window = sub.iloc[:-1]
            if t == "bullish":
                if not (breakout_window["low"] < low_b).any():
                    continue
            else:
                if not (breakout_window["high"] > high_b).any():
                    continue
            # 3) retest
            candidates = (
                sub.iloc[-self.retest_lookback - 1 : -1]
                if len(sub) > self.retest_lookback
                else sub.iloc[:-1]
            )
            if t == "bearish":
                if not candidates["low"].between(low_b, high_b).any():
                    continue
            else:
                if not candidates["high"].between(low_b, high_b).any():
                    continue
            # 4) ultimo prezzo di nuovo FUORI
            if (t == "bearish" and last_close > high_b) or \
               (t == "bullish" and last_close < low_b):
                return True

        return False
