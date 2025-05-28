# region imports
from AlgorithmImports import *
# endregion

import pandas as pd
from typing import Optional, Callable, Tuple, Any, List
from .indicator_wrapper import IndicatorManager, IndicatorWrapper
from .labeller import Labeller
from .callbacks import Callback, CallbackHandler
from skopt import BayesSearchCV
from tqdm import tqdm

class MLStrategy:
    def __init__(
        self,
        indicator_factories: dict[str, Callable[[], IndicatorWrapper]],
        labeller: Labeller,
        id: str,
        window_size: int = 10000,
        atr_period: int = 100,              # ← periodo ATR di default
        symbol: Symbol | None = None    
    ):
        if "atr" not in indicator_factories:
            if symbol is None:
                raise ValueError("Serve `symbol=` se vuoi aggiungere l'ATR auto.")
            indicator_factories = indicator_factories.copy()   # no side‑effect
            indicator_factories["atr"] = lambda: IndicatorWrapper(
                AverageTrueRange(atr_period, MovingAverageType.Wilders),
                "atr",
                symbol
            )

        wrappers = {name: factory() for name, factory in indicator_factories.items()}
        self.indicator_manager = IndicatorManager(wrappers)

        # ------------------------------------------------ 2)  altri parametri
        self.labeller    = labeller
        self.window_size = window_size
        self.model       = None
        self.id          = id
        self.symbol = symbol
        
    def preprocess(self, history_window: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # 1) Etichetta l’intera finestra
        price_data_all = history_window.copy().reset_index()
        price_data_all.rename(
            columns={
                price_data_all.columns[0]: "level_0",
                price_data_all.columns[1]: "level_1"
            },
            inplace=True
        )
        feature_df_all = self.indicator_manager.update_all(price_data_all)

        # DataFrame “completo” su cui lavorerà il Labeller
        combined_all = pd.concat([price_data_all, feature_df_all], axis=1)

        # ------------------------------------------------ 2) etichettatura sull’intera serie
        labelled = self.labeller.label_df(combined_all)

        # ------------------------------------------------ 3) Seleziona righe valide
        mask_valid   = labelled["label"].notna()
        labels       = labelled.loc[mask_valid, "label"].reset_index(drop=True)
        combined_out = combined_all.loc[mask_valid].reset_index(drop=True)

        # (opzionale) rimuovi colonne di indice multi‑level se presenti
        combined_out = combined_out.drop(columns=["level_0", "level_1"],
                                        errors="ignore")

        return combined_out, labels


    def model_selection( 
        self,
        train_df: pd.DataFrame,
        model: Any,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
    ):
        callback_handler = CallbackHandler(
            callbacks=callbacks or [],
            call_every_ms=call_every_ms,
        )
        train_df, labels = self.preprocess(train_df)
        #callback_handler.start()
        model.fit(X=train_df, y=labels)
        #callback_handler.stop()

        if isinstance(model, BayesSearchCV):
            fitted_model =  model.best_estimator_
            
        else:
            fitted_model = model

        self.model = fitted_model
        return {
            "train_df": train_df,
            "labels": labels,
            "best_score": model.best_score_,
            "best_params": model.best_params_}

    def predict(
        self,
        data:pd.DataFrame
    ):
        indicator_df, _ = self.preprocess(data)
        data_point = indicator_df.tail(1)

        return self.model.predict(data_point)

