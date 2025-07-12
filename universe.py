from AlgorithmImports import *
from statsmodels.tsa.stattools import adfuller
#endregion

class StationarySelectionModel(ETFConstituentsUniverseSelectionModel):
    def __init__(self, algorithm: QCAlgorithm, etf: str, lookback: int = 10, universe_settings: UniverseSettings | None = None) -> None:
        self.algorithm = algorithm
        self.lookback = lookback
        self.symbol_data = {}
        self.prices = {}

        symbol = Symbol.create(etf, SecurityType.EQUITY, Market.USA)
        super().__init__(symbol, universe_settings, self.etf_constituents_filter)

    def etf_constituents_filter(self, constituents: List[ETFConstituent]) -> List[Symbol]:
        stationarity = {}

        for c in constituents:
            symbol = c.symbol
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = SymbolData(self.algorithm, symbol, self.lookback)
            data = self.symbol_data[symbol]

            # Update with the last price
            if c.market_value and c.shares_held:
                price = c.market_value / c.shares_held
                data.update(price)
            elif c.price != 0:
                data.update(c.price)
            elif symbol in self.prices:
                data.update(self.prices[symbol])
            # Cache the stationarity test statistics in the dict
            if data.test_statistics is not None:
                stationarity[symbol] = data.test_statistics

        # Return the top 10 lowest test statistics stocks (more negative stat means higher prob to have no unit root)
        selected = sorted(stationarity.items(), key=lambda x: x[1])
        return [x[0] for x in selected[:10]]

class PriceGetter(CoarseFundamentalUniverseSelectionModel):
    def __init__(self, universe: StationarySelectionModel) -> None:
        self.universe = universe
        super().__init__(self.selection)

    def selection(self, coarse: List[CoarseFundamental]) -> List[Symbol]:
        self.universe.prices = {c.symbol: c.price for c in coarse}
        return []

class SymbolData:
    def __init__(self, algorithm: QCAlgorithm, symbol: Symbol, lookback: int) -> None:
        # RollingWindow to hold log price series for stationary testing
        self.window = RollingWindow[float](lookback)
        self.model = None

        # Warm up RollingWindow
        history = algorithm.history[TradeBar](symbol, lookback, Resolution.DAILY)
        for bar in list(history)[:-1]:
            self.window.add(np.log(bar.close))

    def update(self, value: float) -> None:
        if value == 0: return

        # Update RollingWindow with log price
        self.window.add(np.log(value))
        if self.window.is_ready:
            # Test stationarity for log price series by augmented dickey-fuller test
            price = np.array(list(self.window))[::-1]
            self.model = adfuller(price, regression='ct', autolag='BIC')

    @property
    def test_statistics(self) -> float | None:
        return self.model[0] if self.model is not None else None