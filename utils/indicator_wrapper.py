# region imports
from AlgorithmImports import *
# endregion

# Your New Python File
import pandas as pd
from typing import Any, Dict, Optional
from datetime import timedelta

# utils/logger_utils.py
import logging

from tqdm import tqdm
from typing import Any, Dict, Optional
from datetime import timedelta

from utils.indicators import *

    


class IndicatorWrapper:
    """
    Converte un qualunque indicatore QC (o CustomIndicator) in un
    data‑frame di feature, aggiornandolo riga‑per‑riga da un DataFrame
    (deve contenere almeno le colonne timestamp, open/high/low/close …).

    • Per RSI, EMA, MACD, … si usa Update(time, value)
    • Per ATR si costruisce un TradeBar
    • Per BollingerBands si aggiorna con (time, value) e salva 3 colonne
    • Per Stochastic si costruisce un QuoteBar (ask/bid)
    • Per qualsiasi CustomIndicator: self.indicator.Update(window_df)
      e si memorizza il valore di ritorno
    """

    def __init__(self, indicator: Any, name: str, symbol: Symbol):
        self.indicator = indicator
        self.name: str = name.lower()          # es. "atr", "bb", …
        self.symbol: Symbol = symbol
        self._results: List[Dict[str, float]] = []

    def update(self, df: pd.DataFrame) -> None:
        """
        Aggiorna l’indicatore con TUTTE le righe di `df`.
        Il DataFrame **DEVE** avere un DatetimeIndex; le colonne minime
        richieste dipendono dall’indicatore.

        I valori calcolati vengono salvati in `self._results`.
        """
        self._results.clear()


        for idx, row in df.iterrows():
            # ------------- ATR (TradeBar) -------------------------------
            if isinstance(self.indicator, AverageTrueRange):

                bar = TradeBar(
                    time   = row["level_1"].to_pydatetime(),
                    symbol = self.symbol,
                    open   = float(row["open"]),
                    high   = float(row["high"]),
                    low    = float(row["low"]),
                    close  = float(row["close"]),
                    volume = float(row.get("volume", 0)),
                    period = timedelta(minutes=1)      # <‑ usa la tua resolution
                )
                self.indicator.Update(bar)
                self._results.append(
                    {self.name: float(self.indicator.Current.Value)}
                )
                continue

            # ------------- Stochastic (QuoteBar) -----------------------
            if isinstance(self.indicator, Stochastic):
                bar = QuoteBar(
                    row["level_1"].to_pydatetime(),
                    self.symbol,
                    Bar(row["bidclose"], row["bidhigh"], row["bidlow"], row["bidopen"]), 0,
                    Bar(row["askclose"], row["askhigh"], row["asklow"], row["askopen"]), 0,
                    timedelta(minutes=1)
                )
                self.indicator.Update(bar)
                self._results.append(
                    {self.name: float(self.indicator.Current.Value)}
                )
                continue

            # ------------- Bollinger Bands -----------------------------
            if isinstance(self.indicator, BollingerBands):
                self.indicator.Update(row["level_1"], row["close"])
                self._results.append({
                    f"{self.name}_lower": float(self.indicator.LowerBand.ToString()),
                    f"{self.name}_upper": float(self.indicator.UpperBand.ToString()),
                    f"{self.name}_sd": float(self.indicator.StandardDeviation.ToString())
                })
                continue
        
            # ------------- CustomIndicator ----------------------------
       
            if isinstance(self.indicator, CustomIndicator):
      
                val = self.indicator.Update(df.loc[:idx])   # passa finestra fino a ts
                self._results.append({self.name: val})
                continue

            # ------------- Indicatori “normali” (RSI, EMA, MACD, …) ---
           
            self.indicator.Update(row["level_1"].to_pydatetime(), float(row["close"]))
            self._results.append(
                {self.name: float(self.indicator.Current.Value)}
            )


    def to_dataframe(self) -> pd.DataFrame:
        """
        Converte i risultati in DataFrame.
        Ritorna DataFrame vuoto se update non è stato ancora chiamato.
        """
        return pd.DataFrame(self._results)


class IndicatorManager:
    def __init__(self, wrappers: Dict[str, IndicatorWrapper]):
        self.wrappers = wrappers

    def update_all(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_frames = []
        for wrapper in tqdm(self.wrappers.values(), desc="Updating indicators"):
            wrapper.update(df)
            feat_df = wrapper.to_dataframe()
            #print(f"wrapper: {wrapper}, feat_df: {len(feat_df)}")
            feature_frames.append(feat_df)
        # concateno tutti lungo le colonne
        combined = pd.concat(feature_frames, axis=1)
        return combined

    def get_latest(self, df: pd.DataFrame) -> pd.DataFrame:
        combined = self.update_all(df)
        if combined.empty:
            return combined
        return combined.tail(1).reset_index(drop=True)