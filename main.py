from AlgorithmImports import *
import pandas as pd
from datetime import timedelta

from utils.indicator_wrapper import IndicatorWrapper
from utils.strategy import MLStrategy
from utils.labeller import Labeller, BaseStrategy
from utils.rules import FVGRule, PriceBelowMARule
from utils.callbacks import SystemMonitorCallback

from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBClassifier


class BasicTemplateAlgorithm(QCAlgorithm):
    """Esempio di algoritmo QuantConnect che addestra un modello di machine
    learning all'avvio e poi lo utilizza per generare segnali."""

    def Initialize(self) -> None:
        # Periodo di backtest
        self.SetStartDate(2010, 10, 7)
        self.SetEndDate(2024, 10, 11)
        self.SetCash(100_000)

        # Asset principale
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Parametri finestra dati
        self.window_size = 1000
        self.history_window = pd.DataFrame()

        # --------------------------- Indicatori ----------------------------
        self.indicator_factories = {
            "rsi":  lambda: IndicatorWrapper(RelativeStrengthIndex(14), "rsi", self.symbol),
            "macd": lambda: IndicatorWrapper(MovingAverageConvergenceDivergence(12, 26, 9), "macd", self.symbol),
            "bb":   lambda: IndicatorWrapper(BollingerBands(20, 2, MovingAverageType.Simple), "bb", self.symbol),
            "ema":  lambda: IndicatorWrapper(ExponentialMovingAverage(9), "ema", self.symbol),
        }

        # --------------------------- Strategia ----------------------------
        base = BaseStrategy(
            id="long_ma20_fvg",
            bias="bullish",
            rules=[
                PriceBelowMARule(ma_period=20, threshold_pct=0.02),
                FVGRule(lookback=15, must_retest=True, direction="bullish", body_multiplier=1.5),
            ],
        )

        labeller = Labeller(base, lookahead=10)
        self.long_strategy = MLStrategy(
            indicator_factories=self.indicator_factories,
            labeller=labeller,
            id=base.id,
            window_size=self.window_size,
            symbol=self.symbol,
        )

        # ------------------------- Addestramento --------------------------
        history = self.History([self.symbol], timedelta(days=365 * 5), Resolution.Daily)
        history = history.loc[self.symbol][["open", "high", "low", "close"]]

        param_space = {
            "classifier__scale_pos_weight": Real(0.4, 1.0),
            "classifier__n_estimators": Integer(100, 300),
            "classifier__max_depth": Integer(3, 8),
            "classifier__learning_rate": Real(0.001, 0.01),
            "classifier__min_child_weight": Integer(1, 3),
        }

        callbacks = [SystemMonitorCallback()]
        classifier = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        search = BayesSearchCV(classifier, param_space, n_iter=5, random_state=42)

        self.long_strategy.model_selection(
            train_df=history,
            model=search,
            callbacks=callbacks,
        )

        # ------------------------- Risk Management ----------------------
        self.stop_loss_multiple = 2.0
        self.take_profit_multiple = 4.0
        self.stop_loss_ticket = None
        self.take_profit_ticket = None
        self.atr = self.ATR(self.symbol, 14, MovingAverageType.Wilders, Resolution.Daily)

    # ------------------------------------------------------------------
    def OnData(self, data: Slice) -> None:
        if self.symbol not in data:
            return

        bar = data[self.symbol]
        self.history_window = (
            pd.concat([
                self.history_window,
                pd.DataFrame([
                    {
                        "time": self.Time,
                        "open": bar.Open,
                        "high": bar.High,
                        "low": bar.Low,
                        "close": bar.Close,
                    }
                ]),
            ])
            .tail(self.window_size)
        )

        if len(self.history_window) < self.window_size:
            return

        if not self.atr.IsReady:
            return

        signal = int(self.long_strategy.predict(self.history_window)[0])
        invested = self.Portfolio[self.symbol].Invested

        if signal == 1 and not invested:
            quantity = self.CalculateOrderQuantity(self.symbol, 1.0)
            self.MarketOrder(self.symbol, quantity)

            atr_value = self.atr.Current.Value
            stop_price = data[self.symbol].Close - atr_value * self.stop_loss_multiple
            take_price = data[self.symbol].Close + atr_value * self.take_profit_multiple

            self.stop_loss_ticket = self.StopMarketOrder(self.symbol, -quantity, stop_price)
            self.take_profit_ticket = self.LimitOrder(self.symbol, -quantity, take_price)

        elif signal != 1 and invested:
            self.Liquidate(self.symbol)
            if self.stop_loss_ticket:
                self.Transactions.CancelOrder(self.stop_loss_ticket.OrderId)
                self.stop_loss_ticket = None
            if self.take_profit_ticket:
                self.Transactions.CancelOrder(self.take_profit_ticket.OrderId)
                self.take_profit_ticket = None

    def OnOrderEvent(self, orderEvent: OrderEvent) -> None:
        if self.stop_loss_ticket and orderEvent.OrderId == self.stop_loss_ticket.OrderId:
            if orderEvent.Status != OrderStatus.Submitted:
                self.stop_loss_ticket = None
                self.take_profit_ticket = None
        if self.take_profit_ticket and orderEvent.OrderId == self.take_profit_ticket.OrderId:
            if orderEvent.Status != OrderStatus.Submitted:
                self.take_profit_ticket = None
                self.stop_loss_ticket = None

