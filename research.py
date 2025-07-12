
from AlgorithmImports import *
import zipfile
import os
import pandas as pd
import numpy as np



# se il tuo notebook NON è nella root, sblocca il path:
# root = pathlib.Path(__file__).resolve().parent.parent
# sys.path.append(str(root))

# import interni
from utils.indicator_wrapper import IndicatorWrapper
from utils.base_strategy     import BaseStrategy, FVGRule, PriceBelowMARule
from utils.labeller          import Labeller
from utils. strategy import MLStrategy

from utils.callbacks import SystemMonitorCallback
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBClassifier

data_path = "data/equity/usa/minute/spx"  # <-- il tuo path corretto!

# Prendi tutti i file zip
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".zip")]

# Carica tutti gli zip
dfs = []
for file in files:
    with zipfile.ZipFile(file) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
            dfs.append(df)

# Unisci i file
history = pd.concat(dfs)
history = history.sort_values(by="time").reset_index(drop=True)



window_size = 1000

ind_fact = {
    "rsi":  lambda: IndicatorWrapper(RelativeStrengthIndex(14), "rsi"),
    "macd": lambda: IndicatorWrapper(MovingAverageConvergenceDivergence(12,26,9), "macd"),
    "bb":   lambda: IndicatorWrapper(BollingerBands(20, 2), "bb"),
    "ema":  lambda: IndicatorWrapper(ExponentialMovingAverage(9), "ema")
}

spec_bull = BaseStrategy(
    id="long_ma20_fvg",
    bias="bullish",
    move_threshold=6.0,
    rules=[
        PriceBelowMARule(ma_period=1, threshold_pct=0.20),
        FVGRule(lookback=15, must_retest=True, direction="bullish", body_multiplier=1.5)
    ]
)

labeller_bull = Labeller(spec_bull, lookahead=15)

long_strategy = MLStrategy(
    indicator_factories=ind_fact,
    labeller=labeller_bull,
    strategy_id=spec_bull.id,
    window_size=window_size
)

param_space = {
    "classifier__scale_pos_weight": Real(0.4, 1.0),
    "classifier__n_estimators": Integer(1000, 1500),
    "classifier__max_depth": Integer(4, 9),
    "classifier__learning_rate": Real(0.001, 0.01),
    "classifier__min_child_weight": Integer(1, 3),
}
callbacks = [SystemMonitorCallback()]
classifier = XGBClassifier(base_score=0.5)

search = BayesSearchCV(classifier, param_space, n_iter=10, random_state=42)
long_strategy.model_selection(
    train_df=history_df,
    model=search,
    pattern="default",
    callbacks=callbacks
)
benchmark = Benchmark(run_id="fx_run",
                                    exp_dir="fx_experiments")
benchmark.add_strategy(long_strategy)


warmup_period = 100  # ad esempio aspetti 100 barre prima di generare segnali

window = pd.DataFrame()
signals = []          
times   = []

for row in history.itertuples():
    # aggiorna la rolling window
    window = pd.concat([window, pd.DataFrame([{
        "time":  row.time,
        "open":  row.open,
        "high":  row.high,
        "low":   row.low,
        "close": row.close
    }])]).tail(window_size)

    # aspetta che si riempia bene la finestra e superi anche il warmup
    if len(window) < window_size + warmup_period:
        signals.append(0)
        times.append(row.time)
        continue

    # ora puoi calcolare la decisione
    sig = benchmark.run(window)
    signals.append(sig)
    times.append(row.time)

import matplotlib.pyplot as plt

price = history.set_index("time")["close"]
signal_series = pd.Series(signals, index=times)

plt.figure()
price.plot(label="AUDUSD close")
(signal_series.replace({-1:-0.005, 1:0.005, 0:np.nan}) * price).plot(style='o', label="signals")
plt.legend()
plt.title("Segnali strategy vs prezzo")
plt.show()