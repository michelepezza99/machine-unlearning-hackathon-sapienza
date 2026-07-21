# TIM x Sapienza Machine Unlearning Hackathon

Questo repository contiene un workflow riproducibile per rimuovere l'influenza
degli utenti del *forget set* da un classificatore multilabel PyTorch. Il
programma ufficiale per produrre la submission è `main.py`; la ricerca
sperimentale è separata e non modifica automaticamente la configurazione finale.

La metrica di privacy disponibile localmente è soltanto una proxy diagnostica:

```text
MIA ufficiale nascosta
!=
proxy locale rispetto a un retraining di riferimento
```

La proxy aiuta a confrontare esperimenti locali, ma non dimostra la cancellazione
esatta e non garantisce il punteggio della leaderboard.

## Concetti essenziali

- **Modello originale**: il modello consegnato dalla challenge in
  `data/model_artifact`.
- **Forget set**: gli utenti la cui influenza deve essere rimossa; si trova in
  `data/forget_data.csv`.
- **Retain set**: tutti gli utenti originali che non appartengono al forget set.
- **Validation set**: una parte deterministica del retain set, esclusa dal
  training e usata soltanto per misurare utility e selezionare esperimenti.
- **Retraining da zero**: costruzione di un nuovo modello usando soltanto il
  retain-training set. È il riferimento più semplice da interpretare.
- **Fisher dampening**: metodo sperimentale che individua parametri relativamente
  più importanti per il forget set e ne attenua selettivamente il valore, prima
  di una fase di repair sui soli dati retain.
- **Proxy locale di privacy**: confronto, tramite feature di loss e confidenza,
  tra un candidato e il retraining di riferimento. Un valore più alto indica
  maggiore somiglianza locale al riferimento; il retraining vale 1 per
  costruzione. Non è la MIA ufficiale.

## Struttura del repository

Al netto di `.git/` e dei file generati ignorati, l'albero del progetto è:

```text
.
|-- .github/
|   `-- workflows/
|       `-- tests.yml
|-- configs/
|   |-- final_config.json
|   |-- search_configs.json
|   `-- search_stage1_coarse.json
|-- data/
|   |-- asba_sample_hackhaton_sapienza.csv_part-00000-e1716396-b487-4445-acca-72abc35d4e34-c000.csv
|   |-- asba_sample_hackhaton_sapienza.csv_part-00001-e1716396-b487-4445-acca-72abc35d4e34-c000.csv
|   |-- asba_sample_hackhaton_sapienza.csv_part-00002-e1716396-b487-4445-acca-72abc35d4e34-c000.csv
|   |-- asba_sample_hackhaton_sapienza.csv_part-00003-e1716396-b487-4445-acca-72abc35d4e34-c000.csv
|   |-- asba_sample_hackhaton_sapienza.csv_part-00004-e1716396-b487-4445-acca-72abc35d4e34-c000.csv
|   |-- asba_sample_hackhaton_sapienza.csv_part-00005-e1716396-b487-4445-acca-72abc35d4e34-c000.csv
|   |-- asba_sample_hackhaton_sapienza.csv_part-00006-e1716396-b487-4445-acca-72abc35d4e34-c000.csv
|   |-- asba_sample_hackhaton_sapienza.csv_part-00007-e1716396-b487-4445-acca-72abc35d4e34-c000.csv
|   |-- asba_sample_hackhaton_sapienza.csv_part-00008-e1716396-b487-4445-acca-72abc35d4e34-c000.csv
|   |-- asba_sample_hackhaton_sapienza.csv_part-00009-e1716396-b487-4445-acca-72abc35d4e34-c000.csv
|   |-- forget_data.csv
|   `-- model_artifact
|-- machine_unlearning/
|   |-- __init__.py
|   |-- data.py
|   |-- hybrid_recommendation.py
|   |-- metrics.py
|   |-- model.py
|   |-- progressive.py
|   |-- search.py
|   |-- search_aggregation.py
|   |-- submission.py
|   |-- training.py
|   |-- unlearning.py
|   `-- workflow.py
|-- notebooks/
|   `-- 01_experiments.ipynb
|-- reports/
|   |-- final_experiment_summary.md
|   `-- progressive_hyperparameter_search.md
|-- scripts/
|   |-- generate_progressive_configs.py
|   |-- search_configs.py
|   |-- select_final_hybrid.py
|   |-- summarize_all_candidates.py
|   `-- validate_submission.py
|-- tests/
|   |-- conftest.py
|   |-- test_data_splits.py
|   |-- test_metrics.py
|   |-- test_model_artifact.py
|   |-- test_search_selection.py
|   |-- test_search_aggregation.py
|   |-- test_search_workflow.py
|   |-- test_progressive_configs.py
|   |-- test_hybrid_recommendation.py
|   |-- test_smoke_workflow.py
|   |-- test_submission.py
|   |-- test_training.py
|   |-- test_unlearning.py
|   `-- test_workflow_config.py
|-- .gitignore
|-- environment.yml
|-- main.py
|-- README.md
`-- requirements.txt
```

`machine_unlearning/` contiene l'unica implementazione autorevole. Il notebook
importa queste API e resta opzionale. `outputs/` e `submission/` compaiono
soltanto dopo un'esecuzione e sono ignorate da Git.

## Requisiti dei dati

La directory `data/` deve contenere:

- i dieci shard `*c000.csv` separati da punto e virgola;
- `forget_data.csv` con `user_id`, feature e target;
- l'artifact originale `model_artifact` senza estensione.

Il loader conserva l'ordine delle colonne, converte le feature in numeri e
sostituisce NaN o infinito con zero. Le target devono essere finite e binarie.
ID mancanti, duplicati o non riconducibili al training originale causano un
errore. Il retain set viene diviso in modo deterministico, ma con uno split
casuale ordinario: non è una stratificazione multilabel e target molto rare
possono richiedere particolare attenzione.

I dati inclusi sono quelli originali della challenge. Prima di redistribuirli,
verificare autonomamente licenza e condizioni di condivisione.

## Ambiente Python 3.11

Python 3.11 è la versione supportata e usata dalla CI. Dopo l'attivazione
dell'ambiente, verificare che `python --version` mostri `3.11.x`.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
```

### Linux, WSL e macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
```

### Conda

```bash
conda env create -f environment.yml
conda activate hackathon-tim-env
python --version
```

Il workflow terminale non richiede Jupyter. Per aprire il notebook, installare
separatamente `jupyterlab` e `ipykernel` nello stesso ambiente.

## Controlli prima di eseguire esperimenti

I test usano soltanto fixture sintetiche e non avviano la ricerca sui dati reali.

```bash
python -m compileall main.py machine_unlearning scripts tests
python -m pytest -q
```

Le tre interfacce disponibili possono essere ispezionate senza avviare lavoro
costoso:

```bash
python main.py --help
python scripts/search_configs.py --help
python scripts/validate_submission.py --help
```

## Stato della configurazione finale

`configs/final_config.json` usa attualmente
`retraining_from_scratch` per 12 epoche e seed 92. La scelta è **supportata
localmente ma resta in attesa dell'evaluator ufficiale**: una ricerca completa
su dati reali ha confrontato otto candidati base, due varianti gradient-ascent e
il retraining; la proposta è stata poi rieseguita da processo pulito e la
submission è risultata valida. In un confronto full ridotto sui seed 92, 93 e
94 il retraining ha vinto due volte, mentre al seed 94 ha vinto l'ibrido perché
il retraining non rispettava l'utility floor.

Metriche, configurazioni, comandi e limiti sono sintetizzati in
`reports/final_experiment_summary.md`. Questa evidenza locale non misura la MIA
ufficiale nascosta e non garantisce il risultato in leaderboard.

## Ricerca sperimentale

La ricerca è opzionale e separata da `main.py`. Per impostazione predefinita
scrive una proposta in `outputs/search/proposed_final_config.json`; non
sovrascrive `configs/final_config.json`.

Per la ricerca progressiva strutturale → repair/gradient ascent → robustezza su
cinque seed, usare le configurazioni e i generatori descritti in
[`reports/progressive_hyperparameter_search.md`](reports/progressive_hyperparameter_search.md).
Quel flusso aggrega ogni candidato di ogni seed, aggiunge analisi Pareto e può
scrivere per default la configurazione reviewable
`configs/final_config_hybrid.json`; qualunque destinazione canonica viene
rifiutata e `configs/final_config.json` resta invariato.

I generatori degli stage 2, 3 e 4 accettano soltanto evidenza dello stage
precedente accompagnata da `search_metadata.json` con `status="completed"` e
`mode="full"`, e da una `effective_search_config.json` semanticamente coerente
con il template.
L'aggregatore ammette soltanto run per-seed completi in modalità full; il
selettore finale ricalcola dal raw le statistiche del summary e rifiuta
discrepanze, evidenza non-full, collisioni tra input/output e destinazioni
protette. Dopo una raccomandazione selezionata, l'esecuzione finale resta
separata dagli artefatti canonici:

```bash
python main.py --config configs/final_config_hybrid.json --output-dir outputs/final_hybrid --submission-dir submission_hybrid --device cpu
python scripts/validate_submission.py --submission-dir submission_hybrid --data-dir data
```

### Ricerca rapida

La modalità rapida riduce epoche, pazienza, campioni Fisher e numero di candidati
prima delle operazioni costose. Con i percorsi predefiniti usa
`outputs/search/quick/`:

```bash
python scripts/search_configs.py --quick --device cpu
```

`--max-candidates` può imporre un limite ulteriore, per esempio:

```bash
python scripts/search_configs.py --quick --max-candidates 1 --device cpu
```

I valori ridotti effettivi sono stampati all'avvio e salvati nei metadati. Una
ricerca rapida controlla l'integrazione; non basta per scegliere il metodo finale.

### Ricerca completa

```bash
python scripts/search_configs.py --data-dir data --config configs/search_configs.json --output-dir outputs/search --proposed-config outputs/search/proposed_final_config.json --device auto
```

Su una macchina CUDA è possibile sostituire `auto` con `cuda`; `cpu` forza
l'esecuzione su processore.

### Stabilità su più seed

```bash
python scripts/search_configs.py --data-dir data --config configs/search_configs.json --output-dir outputs/search_multiseed --proposed-config outputs/search_multiseed/proposed_final_config.json --device auto --seeds 92 93 94
```

Il riepilogo multi-seed riporta media, deviazione standard, minimo e massimo
delle metriche di utility, forget, proxy locale, score e tempo. `main.py` resta
sempre single-seed.

### Evidenza generata

Una ricerca produce piccoli file leggibili nella directory scelta, tra cui:

- `search_comparison.csv`: confronto dei candidati eseguiti;
- `finalists.csv`: retraining e miglior candidato ibrido;
- `search_metadata.json`: modalità, seed, device, metriche del modello originale,
  soglia utility assoluta, Fisher e configurazione effettiva;
- `proposed_final_config.json`: configurazione proposta, non promossa;
- `best_candidate_summary.json`: riepilogo riproducibile del vincitore;
- storie di retraining, repair e gradient ascent quando applicabili;
- un riepilogo aggregato quando vengono richiesti più seed.

Questi file restano ignorati da Git. Se risultati reali devono diventare evidenza
versionata, sintetizzarli senza modificarli in
`reports/final_experiment_summary.md`, includendo commit, data, device, seed,
configurazione, metriche e limiti. Creare quel report soltanto dopo una vera
esecuzione.

### Ispezione e promozione sicura

Con PowerShell:

```powershell
Import-Csv outputs/search/finalists.csv | Format-Table -AutoSize
Get-Content outputs/search/search_metadata.json
Get-Content outputs/search/proposed_final_config.json
```

Con Linux, WSL o macOS:

```bash
sed -n '1,20p' outputs/search/finalists.csv
python -m json.tool outputs/search/search_metadata.json
python -m json.tool outputs/search/proposed_final_config.json
```

Prima della promozione, eseguire e validare la proposta in percorsi separati:

```bash
python main.py --config outputs/search/proposed_final_config.json --output-dir outputs/proposed_run --submission-dir outputs/proposed_submission --device auto
python scripts/validate_submission.py --submission-dir outputs/proposed_submission --data-dir data
git diff --no-index -- configs/final_config.json outputs/search/proposed_final_config.json
```

`git diff --no-index` restituisce codice 1 quando trova differenze: in questo
caso è il comportamento atteso. Dopo la revisione, la promozione è una copia
manuale ed esplicita:

```powershell
Copy-Item -LiteralPath outputs/search/proposed_final_config.json -Destination configs/final_config.json
```

oppure:

```bash
cp outputs/search/proposed_final_config.json configs/final_config.json
```

Controllare quindi `git diff -- configs/final_config.json` e rieseguire test,
`main.py` e validator. La cronologia Git permette di revisionare o annullare la
modifica; non riscrivere la cronologia.

## Esecuzione finale

Con la configurazione canonica e i percorsi predefiniti:

```bash
python main.py
```

Forma esplicita equivalente:

```bash
python main.py --data-dir data --output-dir outputs/final_run --submission-dir submission --config configs/final_config.json --seed 92 --device auto
```

`main.py` carica la configurazione, ricostruisce lo split, esegue un solo metodo
fisso, salva diagnostica in `outputs/final_run/`, crea la submission e la valida.
Non effettua ricerca né selezione di checkpoint.

## Submission

`submission/` deve contenere **esattamente**:

```text
submission/
|-- model_artifact
|-- execution_time.txt
`-- validation_ids.csv
```

- `model_artifact` contiene almeno `state_dict`, `architecture`,
  `best_hyperparameters` e `model_class_source`; i tensori sono finiti e su CPU.
- `execution_time.txt` contiene un solo intero non negativo, arrotondato per
  eccesso.
- `validation_ids.csv` contiene soltanto la colonna `user_id`, senza duplicati.

Validazione indipendente:

```bash
python scripts/validate_submission.py --submission-dir submission --data-dir data
```

Il validator rifiuta file mancanti o extra, ricostruisce il modello con
`strict=True`, ricrea lo split deterministico, controlla le disgiunzioni e prova
un'inferenza su dati reali.

## Politica di misurazione del tempo

Il timer usa `time.perf_counter()`. Per il retraining include:

- inizializzazione del seed;
- costruzione del modello;
- calcolo dei pesi positivi;
- creazione dell'optimizer;
- tutte le epoche fisse.

Per il metodo ibrido include teacher logits, Fisher retain e forget, costruzione
della maschera, dampening, eventuale gradient ascent, repair e eventuale
ricalibrazione BatchNorm.

Sono esclusi caricamento dei file, ricerca e confronto candidati, valutazione
post-hoc, serializzazione e validator. Il tempo esatto in virgola mobile resta
nei diagnostici; soltanto `execution_time.txt` viene arrotondato per eccesso.

## Riproducibilità, limiti e costi

- Il seed controlla Python, NumPy, PyTorch, split e DataLoader, ma hardware,
  versione delle librerie e kernel CUDA possono ancora produrre piccole
  differenze.
- Un solo seed non misura la stabilità; usare `--seeds` per una decisione
  sperimentale più solida.
- Lo split è deterministico ma non multilabel-stratificato.
- La Fisher per esempio e il retraining completo possono richiedere molto tempo
  su CPU. La modalità rapida serve soltanto come controllo funzionale.
- L'artifact originale non dichiara l'ordine delle feature; il progetto usa
  l'ordine dei CSV e ne verifica la dimensionalità.
- La proxy locale può non correlare con la MIA ufficiale o con la leaderboard.
- Nessun risultato locale garantisce l'esito dell'evaluator nascosto.

## Risoluzione dei problemi

- **`python` non trovato**: installare Python 3.11 e usare `py -3.11` su
  Windows o `python3.11` su Unix per creare l'ambiente.
- **Attivazione PowerShell bloccata**: eseguire
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` nella stessa
  finestra, senza cambiare la policy globale.
- **CUDA non disponibile**: usare `--device cpu`. Richiedere `cuda` senza una
  GPU compatibile produce intenzionalmente un errore.
- **File dati mancante o schema errato**: confrontare `data/` con l'albero sopra;
  non rinominare gli shard e non aggiungere CSV estranei.
- **Submission con file extra**: spostare i file estranei o scegliere una
  directory nuova. Il programma non cancella automaticamente dati sconosciuti.
- **Risultati di ricerca assenti**: eseguire prima la modalità rapida o completa;
  notebook e README non contengono metriche simulate.
- **Memoria o tempo insufficienti**: iniziare con `--quick` e `--device cpu`,
  poi usare una copia di `configs/search_configs.json` per modificare limiti
  sperimentali senza alterare la configurazione canonica.

## Continuous integration

`.github/workflows/tests.yml` esegue compilazione e test sintetici con Python
3.11 sui push a `Michele` e sulle pull request dirette a `Michele`. Non esegue
ricerca Fisher né richiede una submission reale.
