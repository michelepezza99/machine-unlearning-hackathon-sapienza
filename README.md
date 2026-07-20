# TIM x Sapienza Machine Unlearning Hackathon

Questo repository contiene una pipeline riproducibile per rimuovere l'influenza
degli utenti del forget set da un classificatore multilabel PyTorch. La challenge
bilancia utility, privacy e tempo di esecuzione; la Precision@10 e' calcolabile
localmente, mentre la Membership Inference Attack ufficiale e' nascosta.

La proxy privacy inclusa nel workflow di ricerca confronta il comportamento del
modello candidato con un retraining di riferimento. Serve a ordinare gli
esperimenti, ma non e' la metrica ufficiale e non garantisce il risultato in
leaderboard.

## Struttura

```text
.
|-- main.py                         # esecuzione finale autonoma
|-- machine_unlearning/
|   |-- data.py                     # caricamento, split e preprocessing
|   |-- model.py                    # DynamicMLP e artifact
|   |-- metrics.py                  # utility e proxy privacy locale
|   |-- training.py                 # retraining e baseline
|   |-- unlearning.py               # Fisher, SSD, GA e repair
|   |-- submission.py               # creazione e validazione submission
|   `-- workflow.py                 # orchestrazione del metodo finale
|-- configs/
|   |-- final_config.json           # metodo fisso usato da main.py
|   `-- search_configs.json          # spazio di ricerca sperimentale
|-- scripts/
|   |-- search_configs.py           # ricerca e selezione separata
|   `-- validate_submission.py      # validator indipendente
|-- notebooks/
|   `-- 01_experiments.ipynb        # analisi e presentazione
|-- tests/                          # fixture sintetiche e smoke test
|-- data/                           # shard, forget_data.csv, model_artifact
|-- outputs/                        # diagnostica generata, ignorata da Git
`-- submission/                     # esattamente i tre file finali
```

## Ambiente

Il progetto supporta Python 3.11. Il workflow terminale non richiede Jupyter o
IPython.

Con `venv` e pip:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Con Conda:

```bash
conda env create -f environment.yml
conda activate hackathon-tim-env
```

Per eseguire il notebook e' possibile installare separatamente `jupyterlab` e
`ipykernel`.

## Dati

La cartella `data/` deve contenere:

```text
data/
|-- *c000.csv          # partizioni originali, separate da punto e virgola
|-- forget_data.csv    # righe forget con user_id, feature e target
`-- model_artifact     # artifact originale senza estensione
```

Il loader preserva l'ordine delle colonne dei CSV, converte le feature in valori
numerici e sostituisce NaN o infinito con zero, coerentemente con il preprocessing
originale. Le target non vengono imputate: valori mancanti, non finiti o non
binari causano un errore. Gli ID duplicati e gli split sovrapposti sono rifiutati.

Gli shard estratti in `data/` sono la rappresentazione canonica: non manteniamo
archivi ZIP duplicati. Prima di pubblicare o distribuire una copia del repository
occorre verificare autonomamente le condizioni di licenza e condivisione dei dati
della challenge.

## Esecuzione finale

La configurazione attualmente fissata usa il retraining da zero per due epoche,
scelto dalla migliore epoca della baseline reale disponibile. Non esegue ricerca
o early stopping durante il run finale.

Con i percorsi predefiniti:

```bash
python main.py
```

Forma esplicita equivalente:

```bash
python main.py --data-dir data --output-dir outputs/final_run --submission-dir submission --config configs/final_config.json --seed 92
```

Il timer usa `time.perf_counter()`. Per il retraining include seed, costruzione
del modello, calcolo dei pesi di classe e tutte le epoche fisse. Per il metodo
ibrido include teacher logits, Fisher retain/forget, maschera, dampening,
eventuale gradient ascent, repair e ricalibrazione BatchNorm. Esclude caricamento
dati, ricerca, valutazione post-hoc, serializzazione e validator. Il valore viene
arrotondato per eccesso con `math.ceil`.

## Ricerca sperimentale

La ricerca e' intenzionalmente separata da `main.py`:

```bash
python scripts/search_configs.py --data-dir data --config configs/search_configs.json --output-dir outputs/search --selected-config configs/final_config.json
```

Lo script valuta modello originale, retraining, Fisher Dampening, repair,
distillazione e poche varianti di selective gradient ascent. La validation serve
alla selezione locale e resta esclusa dal training. Al termine propone una
configurazione fissa; rieseguire `main.py` non ripete la ricerca.

La proxy privacy usa feature di loss/confidenza e modelli di attacco out-of-fold
raggruppati per utente. Anche con questa precauzione resta una proxy costruita sul
forget set e sul retraining di riferimento, non una stima garantita della MIA
nascosta.

## Submission

La directory `submission/` contiene esclusivamente:

```text
model_artifact
execution_time.txt
validation_ids.csv
```

`model_artifact` conserva le chiavi obbligatorie `state_dict`, `architecture`,
`best_hyperparameters` e `model_class_source`; tutti i tensori sono su CPU.
`execution_time.txt` contiene un solo intero non negativo. `validation_ids.csv`
contiene la sola colonna `user_id`.

Validazione indipendente:

```bash
python scripts/validate_submission.py --submission-dir submission --data-dir data
```

Il validator ricostruisce `DynamicMLP`, carica lo stato con `strict=True`,
controlla valori finiti e device CPU, ricrea lo split deterministico, verifica le
disgiunzioni e prova un'inferenza su dati reali.

## Riproducibilita' e test

Il seed controlla Python, NumPy, PyTorch CPU/CUDA, split e generatori dei
DataLoader. La configurazione finale contiene tutti gli iperparametri necessari e
non dipende da variabili globali di un notebook.

```bash
python -m compileall .
python -m pytest -q
python main.py --help
```

`outputs/`, `submission/`, cache Python/Jupyter e artifact intermedi sono
ignorati da Git. Il repository, non lo ZIP storico della soluzione, e' la fonte
autorevole del codice.

## Limiti noti

- L'evaluator MIA ufficiale non e' pubblico.
- La proxy locale puo' non correlare con il punteggio leaderboard.
- La Fisher per-esempio e' costosa su CPU; il runner emette un avviso per
  configurazioni particolarmente ampie.
- L'artifact originale non dichiara l'ordine delle feature: usiamo l'ordine
  autorevole dei CSV e ne verifichiamo la dimensionalita'.
- Nessuna prestazione sul leaderboard puo' essere garantita senza una submission
  all'evaluator ufficiale.
