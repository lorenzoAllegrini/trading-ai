# Trading AI

Questo repository contiene un framework sperimentale per generare segnali di trading tramite tecniche di machine learning integrato con la piattaforma QuantConnect.

## Struttura

- **`main.py`** – implementa `BasicTemplateAlgorithm`, una strategia `QCAlgorithm` che carica dati storici, calcola vari indicatori, esegue un'ottimizzazione bayesiana degli iperparametri e produce segnali long/short tramite un modello XGBoost.
  La logica di `OnData` applica ora stop loss e take profit dinamici basati sull'ATR per ogni operazione.
- **`research.py`** – script di esempio che mostra come utilizzare i moduli per addestrare e testare la strategia in locale partendo da file di dati compressi.
- **`research.ipynb`** – notebook usato nelle fasi di sperimentazione e analisi.
- **`universe.py`** – definisce modelli di Universe Selection che filtrano i titoli sulla base di criteri di stazionarietà dei prezzi.
- **`utils/`** – raccolta di componenti ausiliarie:
  - `indicator_wrapper.py` – converte gli indicatori di QuantConnect in DataFrame e permette di aggiornarli in batch.
  - `indicators.py` – indicatori custom (differenza prezzo‑EMA, breakout di trendline, dimensione dei Fair Value Gap ecc.).
  - `rules.py` – regole di trading che individuano pattern, breakout e altri segnali.
  - `labeller.py` – etichetta il dataset utilizzando le regole definite dalla strategia.
  - `strategy.py` – classe `MLStrategy` che gestisce calcolo delle feature, labeling e addestramento del modello.
  - `callbacks.py` – callback per monitorare CPU e memoria durante le fasi di ottimizzazione.

## Flusso operativo

1. Preparare i dati storici (ad es. file zip nella cartella `data/equity/...`).
2. Eseguire `research.py` o lavorare con `research.ipynb` per generare feature e label tramite `Labeller` e `IndicatorWrapper`.
3. L'`MLStrategy` addestra un modello `XGBClassifier` eseguendo una ricerca bayesiana degli iperparametri.
4. `main.py` impiega gli stessi moduli in un contesto QuantConnect/LEAN e genera i segnali di trading durante l'esecuzione.

## Requisiti

Il progetto richiede il framework LEAN di QuantConnect e librerie come `pandas`, `numpy`, `scikit-learn`, `skopt` e `xgboost`. Alcuni moduli fanno inoltre riferimento a classi e costrutti interni di QuantConnect.

