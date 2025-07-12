# region imports
from AlgorithmImports import *
# endregion
import pandas as pd
# Your New Python File

class Rule(Protocol):
    """Interface: una regola restituisce True/False sui dati passati."""
    def check(self, window: pd.DataFrame) -> bool: ...

class KeyLevelFinder(Rule):
    """
    Trova supporti / resistenze basandosi su pivot di high/low.

    Un livello è valido se:
      • è massimo locale (resistenza) o minimo locale (supporto)
      • viene toccato ≥ `min_touches` volte nella finestra
      • non viene mai “rotto” oltre la tolleranza `tolerance_pct`
    """

    def __init__(
        self,
        lookback: int = 150,
        locality: int = 20,
        min_touches: int = 2,
        direction: Literal["bullish", "bearish", "both"] = "both",
        tolerance_pct: float = 0.001,
        n_levels: Optional[int] = None,
        levels_lag: int = 10,
    ) -> None:
        self.lookback       = lookback
        self.locality       = locality
        self.min_touches    = min_touches
        self.direction      = direction
        self.tolerance_pct  = tolerance_pct
        self.n_levels       = n_levels
        self.levels_lag     = levels_lag

    # ---------------------------------------------------------------------
    def _pivot_candidates(self, df: pd.DataFrame) -> Dict[float, List[int]]:
        """
        Ritorna dizionario {price: [indici_pivot]} senza ancora validazione.
        """
        touches: Dict[float, List[int]] = {}
        n = len(df)

        for i in range(self.locality, n - self.locality):
            win = df.iloc[i - self.locality : i + self.locality + 1]

            if self.direction in ("bullish", "both") and df.high.iat[i] == win.high.max():
                price = df.high.iat[i]
                touches.setdefault(price, []).append(i)

            if self.direction in ("bearish", "both") and df.low.iat[i] == win.low.min():
                price = df.low.iat[i]
                touches.setdefault(price, []).append(i)

        return touches

    # ---------------------------------------------------------------------
    def detect_key_levels(self, df: pd.DataFrame) -> Dict[str, List[Tuple[pd.Timestamp, float]]]:
     
        if len(df) < self.lookback + self.levels_lag:
            return {"resistance": [], "support": []}
        # finestra “storica” su cui cercare i pivot
        window = df.iloc[-self.lookback - self.levels_lag :-self.levels_lag].copy()
        touches = self._pivot_candidates(window)
        

        validated: Dict[str, List[Tuple[pd.Timestamp, float]]] = {"resistance": [], "support": []}
        tol_mask = (1 + self.tolerance_pct)
       
        for price, idxs in touches.items():
            if len(idxs) < self.min_touches:
                continue
            first_idx = idxs[0]
            tol = price * self.tolerance_pct

            # distingui il tipo guardando se price è in colonna high/min low del pivot d'origine
            is_res = np.isclose(window.high.iloc[first_idx], price)
            series = window.high if is_res else window.low
            violated = (series.iloc[first_idx:] > price + tol) if is_res \
                       else (series.iloc[first_idx:] < price - tol)

            if violated.any():
                continue

            kind = "resistance" if is_res else "support"
            validated[kind].append((window.index[first_idx], price))

        # ordina e limita
        if self.n_levels is not None:
            validated["resistance"] = sorted(validated["resistance"], key=lambda x: x[1], reverse=True)[: self.n_levels]
            validated["support"]    = sorted(validated["support"],    key=lambda x: x[1])[: self.n_levels]

        return validated
    
    def merge(self, pivots: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        pivots = sorted(pivots, key=lambda x: x[1])
        merged: List[Tuple[int,float]] = []
        for l,p in pivots:
            if not merged or abs(p-merged[-1][1])>self.C*p:
                merged.append((l,p))
        return merged

class KeyLevelBounceIndicator(KeyLevelFinder):

    def __init__(
        self,
        *args: Any,
        bounce_lookback: int = 5,
        breakout: bool = False,
        breakout_tol_pct: float = 0.0004,   # extra % oltre tolerance per lo spike
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bounce_lookback  = bounce_lookback
        self.breakout         = breakout
        self.breakout_tol_pct = breakout_tol_pct   # usato solo se breakout=True

    def check(self, data: pd.DataFrame) -> Tuple[bool, Optional[Tuple[Any, float]]]:
        if len(data) <= self.levels_lag + self.bounce_lookback:
            return False
        
        past = data.iloc[-self.lookback - self.levels_lag :]

        levels = self.detect_key_levels(past)
        recent = data.iloc[-self.bounce_lookback:]
        if recent.empty:
            return False

        tol  = self.tolerance_pct
        brk  = tol + self.breakout_tol_pct
      
        if self.direction == "bullish":
            for ts_lvl, lvl in levels["resistance"]:
                touched_idx = None
                for i, bar in enumerate(recent.itertuples(index=False)):
                    prev = recent.iloc[i-1]
                    curr = recent.iloc[i]
                    if not self.breakout:
                        touch  = prev.high >= lvl - tol
                        reject = curr.close  <  lvl - tol
                    else:
                        touch  = prev.high  >= lvl * (1 + brk)
                        reject = curr.close < lvl 

                    if touch and touched_idx is None:
                        touched_idx = i                # abbiamo il primo touch
                        continue 

                    if reject and touched_idx is not None and i > touched_idx:
                        return True

        else:  # SUPPORTO -----------------------------------------------------------
            for ts_lvl, lvl in levels["support"]:
                touched_idx = None  
                for i, bar in enumerate(recent.itertuples(index=False)):
                    if not self.breakout:
                        touch  = bar.close <= lvl + tol
                        reject = bar.low   >  lvl + tol
                    else:
                        touch  = bar.low  <= lvl * (1 - brk)
                        reject = bar.close > lvl 

                    if touch and touched_idx is None:
                        touched_idx = i        
                        continue             

                    if reject and touched_idx is not None and i > touched_idx:
                        return True
        return False



class BosIndicator(KeyLevelFinder):
    def __init__(self, *args: Any, breakout_lookback: int = 10, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.break_lookback = breakout_lookback
    def check(self, data: pd.DataFrame) -> Tuple[bool, Optional[Tuple[Any, float]]]:

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
                    return True

        if self.direction in ("bearish", "both"):
            for lvl_idx, lvl_price in key_levels["support"]:
                if body_high < lvl_price:
                    return True
        return False

class TrendlineFinder(Rule):
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
    ) -> None:
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


class PriceEmaDifferenceRule(Rule):
    """
    Regola su PriceEmaDifferenceIndicator.

    direction: "positive" (prezzo sopra EMA), 
               "negative" (prezzo sotto EMA),
               "both"     (esorso assoluto >= threshold)
    threshold: differenza percentuale minima (es. 0.02 = 2%)
    """
    def __init__(
        self,
        direction: Literal["positive", "negative", "both"],
        threshold: float,
        lookback: int
    
    ) -> None:
        if direction not in ("positive", "negative", "both"):
            raise ValueError("direction deve essere 'positive', 'negative' o 'both'")
        self.direction = direction
        self.threshold = threshold
        self.last_diff: Optional[float] = None
        self.lookback = lookback

    def check(self, window: pd.DataFrame) -> bool:
        """
        Restituisce True se la differenza EMA meets la condizione impostata.
        """
        if not isinstance(window, pd.DataFrame):
            return None
            
        # Cerca la colonna 'close'
        closes = window["close"]

        if len(closes) == 0:
            return None

        # Calcola EMA sull'ultima finestra
        ema_series = closes.ewm(span=self.lookback, adjust=False).mean()
        last_ema = ema_series.iloc[-1]
        last_close = closes.iloc[-1]
        diff = (last_close - last_ema) / last_ema

        if diff is None:
            return False

        if self.direction == "positive":
            return diff >= self.threshold
        elif self.direction == "negative":
            return diff <= -self.threshold
        else:  # both
            return abs(diff) >= self.threshold


class PriceBelowMARule(Rule):
    """Regola che verifica se il prezzo di chiusura e' al di sotto di una
    media mobile semplice di ``ma_period`` barre di almeno ``threshold_pct``.
    """

    def __init__(self, ma_period: int, threshold_pct: float = 0.0) -> None:
        self.ma_period = ma_period
        self.threshold_pct = threshold_pct
        self.lookback = ma_period

    def check(self, window: pd.DataFrame) -> bool:
        if len(window) < self.ma_period:
            return False

        ma = window["close"].rolling(self.ma_period).mean().iloc[-1]
        current = window["close"].iloc[-1]
        return current < ma * (1 - self.threshold_pct)


class FVGRule(Rule):
    def __init__(
        self,
        lookback: int = 20,
        must_retest: bool = False,
        direction: Literal["bullish", "bearish", "both"] = "bullish",
        body_multiplier: float = 1.5,
        retest_lookback: int = 4
    ) -> None:
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



class InverseFVGRule(FVGRule):
    def __init__(self, *args: Any, inverse: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.inverse = inverse


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
        
        inv = []
        for gap in fvg_list:
            low_b, high_b, idx0 = gap["gap_low"], gap["gap_high"], gap["index"]
            sub = data.loc[idx0:]                                # barre dalla creazione in poi

            if gap["type"] == "bullish":
                # la prima candela che invalida ha HIGH ≤ gap_low
                mask = sub["high"] <= low_b
                if mask.any():
                    first_inv_idx = sub.index[mask.argmax()]     # <-- indice del primo invalido
                    inv.append({
                        "type":     "bearish",
                        "gap_low":  low_b,
                        "gap_high": high_b,
                        "index":    first_inv_idx                # <-- salvo questo indice
                    })

            else:  # gap["type"] == "bearish"
                # la prima candela che invalida ha LOW ≥ gap_high
                mask = sub["low"] >= high_b
                if mask.any():
                    first_inv_idx = sub.index[mask.argmax()]
                    inv.append({
                        "type":     "bullish",
                        "gap_low":  low_b,
                        "gap_high": high_b,
                        "index":    first_inv_idx
                    })

        return inv


class FvgBosRule:
    def __init__(
        self,
        fvg_rule: FVGRule,
        bos_rule: BosIndicator,
        tol_factor: float = 2.0
    ) -> None:
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
    