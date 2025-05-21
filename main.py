from AlgorithmImports import *
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# importa le tue classi
from utils.indicator_wrapper import IndicatorWrapper, IndicatorManager
from utils.strategy import MLStrategy
from utils.trading_benchmark import MLTradingBenchmark
from utils.labeller import Labeller 
from utils.base_strategy import BaseStrategy, FVGRule, PriceBelowMARule
from utils.callbacks import CallbackHandler, SystemMonitorCallback
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
from functools import partial
from sklearn.model_selection import TimeSeriesSplit
from skopt.space import Real, Categorical, Integer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from datetime import timedelta

class BasicTemplateAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2010, 10, 7)
        self.SetEndDate(2024, 10, 11)
        self.SetCash(100_000)
        self.symbol = self.AddEquity("SPY", Resolution.MINUTE).Symbol
        self.strategies = {}
        self.window_size = 1000  
        self.history_window = pd.DataFrame()

        # Setup indicator factories
        self.indicator_factories = {
            "rsi":  lambda: IndicatorWrapper(RelativeStrengthIndex(14), "rsi"),
            "macd": lambda: IndicatorWrapper(MovingAverageConvergenceDivergence(12,26,9), "macd"),
            "bb":   lambda: IndicatorWrapper(BollingerBands(20, 2, MovingAverageType.Simple), "bb"),
            "ema":  lambda: IndicatorWrapper(ExponentialMovingAverage(9), "ema")
        }

        # Setup strategy
        spec_bull = BaseStrategy(
            id="long_ma20_fvg",
            bias="bullish",
            rules=[
                PriceBelowMARule(ma_period=1, threshold_pct=0.20),
                FVGRule(lookback=15, must_retest=True, direction="bullish", body_multiplier=1.5)
            ]
        )

        labeller_bull = Labeller(spec_bull, lookahead=10)

        self.long_strategy = MLStrategy(
            indicator_factories=self.indicator_factories,
            labeller=labeller_bull,
            id=spec_bull.id,
            window_size=self.window_size
        )

        # Hyperparameter optimization
        param_space = {
            "classifier__scale_pos_weight": Real(0.4, 1.0),
            "classifier__n_estimators": Integer(1000, 1500),
            "classifier__max_depth": Integer(4, 9),
            "classifier__learning_rate": Real(0.001, 0.01),
            "classifier__min_child_weight": Integer(1, 3),
        }
        callbacks = [SystemMonitorCallback()]
        classifier = XGBClassifier(base_score=0.5)
        self.Debug("sfioasdfsahadsfiuhiadfhsiuadsadfshiadshiufiuadsdfhsu")
        history = self.History([self.symbol], timedelta(days=365*5), Resolution.MINUTE).loc[self.symbol]
        history_df = pd.DataFrame({
            "open": history["open"],
            "high": history["high"],
            "low":  history["low"],
            "close": history["close"]
        })
        self.Debug(str(history_df.head(5)))
        search = BayesSearchCV(classifier, param_space, n_iter=10, random_state=42)
        res = self.long_strategy.model_selection(
            train_df=history_df,
            model=search,
            callbacks=callbacks
        )
        self.Debug(str(res))
        self.strategies[self.long_strategy.id] = self.long_strategy



    def OnData(self, data):
        if self.symbol not in data:
            return

        bar = data[self.symbol]

        # Aggiorna la finestra rolling
        self.history_window = (
            pd.concat([self.history_window, pd.DataFrame([{
                "time":  self.Time,
                "open":  bar.Open,
                "high":  bar.High,
                "low":   bar.Low,
                "close": bar.Close
            }])])
            .tail(self.window_size)
        )

        if len(self.history_window) < self.window_size:
            return

        # Esegui la strategia direttamente
        decision = self.long_strategy.predict(self.history_window)

        if decision == 1:
            self.SetHoldings(self.symbol, 0.5)
            self.Debug(f"BUY {self.symbol} @ {bar.Close}")
        elif decision == -1:
            self.SetHoldings(self.symbol, -0.5)
            self.Debug(f"SELL {self.symbol} @ {bar.Close}")

    def direction_change(self, raw_type, data, num_lookback):
        
        '''
        num_lookback is the number of previous prices
        raw_type is a string: open, high, low or close
        Data is a series
        Returns a dataframe of previous prices
        '''
        
        prices = []
        length = len(data)
    
        for i in range(num_lookback, length+1):
            this_data = np.array(data[i-num_lookback : i])

            # input is change in priice
            prices.append(np.diff(this_data.copy()))
        
        prices_df = pd.DataFrame(prices)
            
        columns = {}
        
        for index in prices_df.columns:
            columns[index] = "{0}_shifted_by_{1}".format(raw_type, num_lookback - index-1)
        
        prices_df.rename(columns = columns, inplace=True)
        
        prices_df.index += num_lookback - 1
        
        return prices_df
    
    def _previous_prices(self, raw_type, data, num_lookback):
        
            '''
            num_lookback is the number of previous prices
            Data is open, high, low or close
            Data is a series
            Returns a dataframe of previous prices
            '''
            
            prices = []
            length = len(data)
        
            for i in range(num_lookback, length+1):
                this_data = np.array(data[i-num_lookback : i])
                prices.append(this_data)
        
            prices_df = pd.DataFrame(prices)
                
            columns = {}
            
            for index in prices_df.columns:
                columns[index] = "{0}_shifted_by_{1}".format(raw_type, num_lookback - index)
            
            prices_df.rename(columns = columns, inplace=True)
            prices_df.index += num_lookback
            
            return prices_df
    


