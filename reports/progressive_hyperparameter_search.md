# Ricerca progressiva per l'ibrido Fisher

## Scopo e limiti

Questo workflow confronta configurazioni del metodo
`hybrid_fisher_dampening` in modo progressivo, riproducibile e verificabile. La
ricerca locale usa una proxy di privacy costruita rispetto a un retraining di
riferimento:

```text
proxy locale retrained-reference != MIA ufficiale nascosta
```

La proxy locale serve a ordinare esperimenti locali. Non costituisce una
validazione privacy ufficiale, non anticipa il risultato della leaderboard e non
permette di definire una configurazione "perfetta" o "ufficialmente ottima".

La configurazione canonica `configs/final_config.json` non viene modificata da
questo processo. L'eventuale raccomandazione ibrida viene scritta soltanto in
`configs/final_config_hybrid.json`.

## Perché una ricerca progressiva

Una griglia cartesiana completa mescolerebbe contemporaneamente struttura
Fisher, gradient ascent, repair e BatchNorm. Il costo crescerebbe rapidamente e
sarebbe difficile attribuire un miglioramento a una scelta precisa. La ricerca
progressiva cambia invece pochi fattori per volta:

1. individua regioni strutturali promettenti;
2. raffina quantile Fisher e BatchNorm soltanto in tre famiglie;
3. studia gradient ascent e repair soltanto in due strutture;
4. ripete quattro finalisti su cinque seed;
5. applica vincoli di completezza e stabilità prima del ranking finale.

Teacher logits e Fisher retain/forget continuano a essere calcolati una sola
volta per seed dal runner esistente. I candidati restano sequenziali: il workflow
non introduce parallelismo non deterministico e non duplica l'implementazione
dell'unlearning.

## Stage 1 — ricerca strutturale ampia

Configurazione: `configs/search_stage1_coarse.json`.

Varia principalmente:

- `top_fraction`: 0,25%, 0,5%, 1% e 2%;
- `minimum_dampening_factor`: 0,95, 0,90, 0,85 e 0,82 nelle combinazioni
  previste;
- assenza/presenza di gradient ascent tramite l'espansione progressiva esistente.

Gli altri campi importanti restano ai valori condivisi della ricerca corrente.
Il budget full è composto da 12 candidati base e fino a 4 varianti GA, quindi 16
esecuzioni ibride quando almeno quattro basi sono valide. Ogni variante automatica
copia una base valida, imposta quattro step e usa
`max(repair_learning_rate * 0.1, 1e-7)` come learning rate. Un fallimento resta
nel CSV e non può generare una variante.

In quick mode vengono eseguiti soltanto i primi quattro candidati base; epoche,
pazienza e campioni Fisher sono ridotti e le varianti GA automatiche sono
disabilitate. Quick è un test d'integrazione, non evidenza per la selezione.

PowerShell:

```powershell
python scripts\search_configs.py `
  --quick `
  --config configs\search_stage1_coarse.json `
  --output-dir outputs\stage1_quick `
  --proposed-config outputs\stage1_quick\proposed_final_config.json `
  --device cpu

python scripts\search_configs.py `
  --config configs\search_stage1_coarse.json `
  --output-dir outputs\stage1_full `
  --proposed-config outputs\stage1_full\proposed_final_config.json `
  --device cpu
```

Bash:

```bash
python scripts/search_configs.py \
  --quick \
  --config configs/search_stage1_coarse.json \
  --output-dir outputs/stage1_quick \
  --proposed-config outputs/stage1_quick/proposed_final_config.json \
  --device cpu

python scripts/search_configs.py \
  --config configs/search_stage1_coarse.json \
  --output-dir outputs/stage1_full \
  --proposed-config outputs/stage1_full/proposed_final_config.json \
  --device cpu
```

## Stage 2 — raffinamento strutturale

Il generatore legge il vero `outputs/stage1_full/search_comparison.csv`, conserva
soltanto righe valide che superano l'utility floor e raggruppa per
`(top_fraction, minimum_dampening_factor)`. In questo modo una base e le sue
varianti GA non diventano famiglie diverse. Le tre famiglie migliori sono
selezionate con ranking esplicito e tie-break stabile; il JSON registra
rappresentante, metriche e motivazione.

Per ogni famiglia vengono provati:

- `forget_absolute_quantile`: 0,40, 0,50 e 0,60;
- `recalibrate_batchnorm`: `false` e `true`.

Il budget è 3 × 3 × 2 = 18 basi, più fino a 6 varianti GA automatiche: massimo
24 candidati. I nomi sono deterministici e leggibili, per esempio
`tf_0p5_d090_q040_no_bn`.

```powershell
python scripts\generate_progressive_configs.py stage2
python scripts\search_configs.py `
  --config configs\search_stage2_refinement.json `
  --output-dir outputs\stage2_refinement `
  --proposed-config outputs\stage2_refinement\proposed_final_config.json `
  --device cpu
```

```bash
python scripts/generate_progressive_configs.py stage2
python scripts/search_configs.py \
  --config configs/search_stage2_refinement.json \
  --output-dir outputs/stage2_refinement \
  --proposed-config outputs/stage2_refinement/proposed_final_config.json \
  --device cpu
```

Il generatore si arresta se mancano tre famiglie valide distinte: non inventa
vincitori e non rilassa l'utility floor.

Questa regola di provenienza vale per tutti i generatori successivi (`stage2`,
`stage3` e `stage4`): il solo CSV non basta. Nella stessa directory devono
esistere `search_metadata.json`, con `status="completed"` e `mode="full"`, ed
`effective_search_config.json`. La configurazione effettiva viene validata e
deve coincidere con il template nelle impostazioni semantiche condivise
(schema, seed, split e batch di valutazione, utility floor, numero di varianti
GA automatiche, retraining, Fisher e candidato comune). Il JSON generato
registra l'esito in `progressive_generation.evidence_validation`. Il generatore
rifiuta inoltre output che sovrascrivano risultati o template, la configurazione
canonica `configs/final_config.json`, oppure file sotto `outputs/final_run` e
`submission`.

## Stage 3 — gradient ascent e repair

Il generatore legge il vero `outputs/stage2_refinement/search_comparison.csv`,
filtra validità e utility floor e raggruppa le varianti di nome/GA secondo la
struttura effettiva:

```text
top_fraction, minimum_dampening_factor,
forget_absolute_quantile, recalibrate_batchnorm
```

Seleziona due strutture con priorità a proxy privacy locale, Precision@10 e
tempo. Per ciascuna struttura crea dieci candidati:

- repair di riferimento con sei profili GA: nessun GA; 2 step a 5e-6; 2 step a
  1e-5; 4 step a 1e-5; 4 step a circa 1,3559e-5; 8 step a 5e-6;
- nessun GA con quattro repair alternativi: conservativo, flessibile, forte e
  fortemente regolarizzato.

L'intersezione "nessun GA + repair di riferimento" compare una sola volta. Il
totale è quindi 20 candidati, non il prodotto completo 6 × 5 × 2. Le varianti GA
automatiche sono impostate a zero perché ogni profilo è esplicito. Ogni candidato
ha una motivazione registrata nei metadati del JSON.

```powershell
python scripts\generate_progressive_configs.py stage3
python scripts\search_configs.py `
  --config configs\search_stage3_finalists.json `
  --output-dir outputs\stage3_refinement `
  --proposed-config outputs\stage3_refinement\proposed_final_config.json `
  --device cpu
```

```bash
python scripts/generate_progressive_configs.py stage3
python scripts/search_configs.py \
  --config configs/search_stage3_finalists.json \
  --output-dir outputs/stage3_refinement \
  --proposed-config outputs/stage3_refinement/proposed_final_config.json \
  --device cpu
```

## Stage 4 — robustezza multi-seed

`stage3` è un insieme di raffinamento da 20 candidati; non va ripetuto
integralmente cinque volte. Un passaggio separato seleziona quattro finalisti,
bilanciando strutture e profili GA/repair, e scrive
`configs/search_stage4_multiseed.json`.

```powershell
python scripts\generate_progressive_configs.py stage4 --finalist-count 4
python scripts\search_configs.py `
  --config configs\search_stage4_multiseed.json `
  --seeds 92 93 94 95 96 `
  --output-dir outputs\stage4_multiseed `
  --proposed-config outputs\stage4_multiseed\proposed_final_config.json `
  --device cpu
```

```bash
python scripts/generate_progressive_configs.py stage4 --finalist-count 4
python scripts/search_configs.py \
  --config configs/search_stage4_multiseed.json \
  --seeds 92 93 94 95 96 \
  --output-dir outputs/stage4_multiseed \
  --proposed-config outputs/stage4_multiseed/proposed_final_config.json \
  --device cpu
```

Il retraining viene ancora eseguito per costruire la proxy locale, ma non è un
candidato della raccomandazione ibrida. Ogni seed conserva il proprio
`seed_<n>/search_comparison.csv` completo. Il `proposed_final_config.json`
prodotto dal runner generale resta soltanto una proposta per-seed e non sostituisce
la selezione multi-seed descritta sotto.

## File prodotti dagli stage

I generatori scrivono soltanto configurazioni reviewable:

- Stage 1: `configs/search_stage1_coarse.json`, già versionabile;
- dopo Stage 1 full: `configs/search_stage2_refinement.json`;
- dopo Stage 2: `configs/search_stage3_finalists.json`;
- dopo Stage 3: `configs/search_stage4_multiseed.json`.

Ogni ricerca single-seed conserva nella propria directory almeno
`effective_search_config.json`, `validation_ids.csv`, `retraining_history.csv`,
`search_comparison.csv`, `finalists.csv`, `search_metadata.json`,
`best_candidate_summary.json`, la proposta non promossa e gli storici del miglior
ibrido. Stage 4 conserva gli stessi file sotto ogni `seed_92/` … `seed_96/` e,
alla radice, metadati e riepilogo dei soli finalisti prodotti dal runner
esistente. L'aggregatore aggiunge poi le tabelle complete descritte nella sezione
successiva. Nessuna directory di risultato viene creata dal solo generatore.

## Come leggere `search_comparison.csv`

Ogni riga rappresenta un candidato ibrido, inclusi i fallimenti. Le colonne
principali sono:

- `status`, `valid`, `error_type`, `error_message`: esito tecnico;
- `utility_floor_pass`: rispetto del floor calcolato dal runner;
- `precision_at_10`, `utility_ratio`, `validation_bce`: utility;
- `forget_bce`, `local_privacy_proxy`: diagnostica forget/privacy locale;
- `execution_time_seconds`: tempo del metodo secondo la policy del repository;
- `selected_parameter_fraction`: frazione di parametri Fisher selezionati;
- `best_epoch`: checkpoint di repair scelto durante la ricerca;
- `local_search_score`: combinazione locale di utility, proxy e tempo.

`local_search_score` non deve essere l'unico criterio: comprime obiettivi diversi
in un numero, dipende dalla proxy non ufficiale e può nascondere una pessima
prestazione nel seed peggiore. Il workflow finale usa filtri rigidi, minimi
multi-seed, tempi e complessità prima dei tie-break.

## Aggregazione completa e fingerprint

Il riepilogo usa tutti i `seed_*/search_comparison.csv`, non `finalists.csv`:

```powershell
python scripts\summarize_all_candidates.py `
  --input-dir outputs\stage4_multiseed `
  --expected-seeds 92 93 94 95 96
```

```bash
python scripts/summarize_all_candidates.py \
  --input-dir outputs/stage4_multiseed \
  --expected-seeds 92 93 94 95 96
```

Produce:

- `all_candidates_all_seeds.csv`: ogni esecuzione, anche fallita;
- `all_candidates_summary.csv`: copertura seed, validità, utility floor e
  statistiche di tutte le metriche richieste;
- `all_candidates_pareto.csv`: candidati validi che superano il floor con
  `is_pareto_optimal` e numero di dominatori;
- `all_candidates_metadata.json`: sorgenti, scope del fingerprint, deviazione
  standard e limitazione privacy.

Le deviazioni standard sono di popolazione (`ddof=0`). Le metriche sono
aggregate sulle esecuzioni valide; fallimenti e run mancanti restano conteggiati
separatamente.

Prima di aggregare, lo script richiede per ogni directory `seed_<n>` il relativo
`search_metadata.json` con `status="completed"` e `mode="full"`; evidenza quick,
incompleta o priva di metadati viene rifiutata. Gli output raw, summary, Pareto e
metadati devono essere distinti e non possono sovrascrivere le evidenze
sorgente, `configs/final_config.json` o file sotto `outputs/final_run` e
`submission`.

Il fingerprint SHA-256 usa JSON canonico dei campi semantici del candidato e,
quando disponibile, della configurazione effettiva condivisa (split, evaluation
batch e Fisher). Esclude seed, nomi, percorsi, motivazioni, stato e metriche. Se
gli step GA sono zero, i relativi learning rate/batch non fanno identità; se
BatchNorm è disabilitata, il batch di ricalibrazione non fa identità. Lo script
rifiuta duplicati seed/configurazione e lo stesso nome associato a configurazioni
effettive diverse.

L'analisi Pareto massimizza le medie di Precision@10 e proxy privacy locale,
minimizza il tempo medio e, quando disponibile per tutti gli eleggibili, la
frazione media di parametri selezionati. Anche questa frontiera dipende dalla
proxy locale e non è una frontiera rispetto alla MIA ufficiale.

## Raccomandazione ibrida deterministica

```powershell
python scripts\select_final_hybrid.py `
  --summary outputs\stage4_multiseed\all_candidates_summary.csv `
  --raw outputs\stage4_multiseed\all_candidates_all_seeds.csv `
  --search-config configs\search_stage4_multiseed.json `
  --expected-seeds 92 93 94 95 96 `
  --recommendation-output outputs\stage4_multiseed\hybrid_recommendation.json `
  --config-output configs\final_config_hybrid.json
```

```bash
python scripts/select_final_hybrid.py \
  --summary outputs/stage4_multiseed/all_candidates_summary.csv \
  --raw outputs/stage4_multiseed/all_candidates_all_seeds.csv \
  --search-config configs/search_stage4_multiseed.json \
  --expected-seeds 92 93 94 95 96 \
  --recommendation-output outputs/stage4_multiseed/hybrid_recommendation.json \
  --config-output configs/final_config_hybrid.json
```

Un candidato è eleggibile soltanto se:

- compare esattamente nei cinque seed richiesti;
- ha `valid_rate = 1.0`;
- ha `utility_floor_pass_rate = 1.0`;
- dispone di almeno un `best_epoch` positivo.

Il selettore non considera il summary come fonte indipendente: ricalcola dalle
righe grezze copertura, tassi e metriche aggregate, poi rifiuta ogni discrepanza
con il summary. Richiede inoltre `evidence_mode="full"` per ogni seed. I tre
output (configurazione, raccomandazione e diagnostica) devono essere distinti,
non possono sovrascrivere summary, raw o search config, né usare
`configs/final_config.json` o percorsi sotto `outputs/final_run` e `submission`.
Se nessun candidato è eleggibile, scrive raccomandazione e diagnostica
provvisorie, restituisce codice 2 e non scrive una nuova configurazione finale;
i vincoli non vengono rilassati.

Fra gli eleggibili il ranking gerarchico preferisce, nell'ordine: minimo e media
della proxy locale più alti; minimo di Precision@10 più alto; minimo di
`utility_ratio` più alto; tempo medio e massimo più bassi; frazione media di
parametri più bassa; quindi minore complessità (top fraction, step GA, epoche,
niente ricalibrazione BatchNorm) e infine nome/fingerprint.

`fixed_repair_epochs` è la moda dei `best_epoch` positivi osservati. In caso di
più mode viene scelto il valore positivo minore; zero non è mai selezionato. Il
JSON di raccomandazione conserva tutti i valori per seed e la regola usata.

## Esecuzione e validazione finale ibrida

Soltanto dopo una raccomandazione con stato selezionato:

```powershell
python main.py `
  --config configs\final_config_hybrid.json `
  --output-dir outputs\final_hybrid `
  --submission-dir submission_hybrid `
  --device cpu

python scripts\validate_submission.py `
  --submission-dir submission_hybrid `
  --data-dir data
```

```bash
python main.py \
  --config configs/final_config_hybrid.json \
  --output-dir outputs/final_hybrid \
  --submission-dir submission_hybrid \
  --device cpu

python scripts/validate_submission.py \
  --submission-dir submission_hybrid \
  --data-dir data
```

Questi percorsi non toccano `outputs/final_run` o `submission`. Il validatore
richiede esattamente `model_artifact`, `execution_time.txt` e
`validation_ids.csv`, ricostruisce lo split e prova un'inferenza reale.

## Sequenza end-to-end sintetica

Prima degli esperimenti reali:

```powershell
python -m compileall main.py machine_unlearning scripts tests
python -m pytest -q
python scripts\search_configs.py --quick --config configs\search_stage1_coarse.json --output-dir outputs\stage1_quick --proposed-config outputs\stage1_quick\proposed_final_config.json --device cpu
```

Sequenza reale, da continuare soltanto quando lo stage precedente è completo:

```powershell
python scripts\search_configs.py --config configs\search_stage1_coarse.json --output-dir outputs\stage1_full --proposed-config outputs\stage1_full\proposed_final_config.json --device cpu
python scripts\generate_progressive_configs.py stage2
python scripts\search_configs.py --config configs\search_stage2_refinement.json --output-dir outputs\stage2_refinement --proposed-config outputs\stage2_refinement\proposed_final_config.json --device cpu
python scripts\generate_progressive_configs.py stage3
python scripts\search_configs.py --config configs\search_stage3_finalists.json --output-dir outputs\stage3_refinement --proposed-config outputs\stage3_refinement\proposed_final_config.json --device cpu
python scripts\generate_progressive_configs.py stage4 --finalist-count 4
python scripts\search_configs.py --config configs\search_stage4_multiseed.json --seeds 92 93 94 95 96 --output-dir outputs\stage4_multiseed --proposed-config outputs\stage4_multiseed\proposed_final_config.json --device cpu
python scripts\summarize_all_candidates.py --input-dir outputs\stage4_multiseed --expected-seeds 92 93 94 95 96
python scripts\select_final_hybrid.py
python main.py --config configs\final_config_hybrid.json --output-dir outputs\final_hybrid --submission-dir submission_hybrid --device cpu
python scripts\validate_submission.py --submission-dir submission_hybrid --data-dir data
```

Ogni comando di generazione usa soltanto evidenza realmente presente. Nessun
CSV viene simulato e nessun vincitore viene codificato in anticipo.
