# Riepilogo dell'esperimento finale

- Data: 21 luglio 2026
- Branch: `Michele`
- Commit del workflow finale: `c364305`
- Commit della ricerca: `9e3e4cc`

## Stato della decisione

La configurazione promossa è `retraining_from_scratch` con seed 92 e 12 epoche
fisse. La scelta è supportata dagli esperimenti locali descritti qui, ma resta
in attesa dell'evaluator ufficiale. La proxy locale rispetto al retraining non è
la Membership Inference Attack nascosta e non garantisce il risultato in
leaderboard.

## Ambiente e dati

- esecuzione reale su CPU;
- interprete locale: Python 3.14.4; il progetto e la CI sono configurati per
  Python 3.11;
- 129.783 righe totali, 120.698 retain e 9.085 forget;
- split seed 92: 107.421 retain-training e 13.277 validation;
- 381 feature e 28 target;
- utility floor: 98,5% della Precision@10 del modello originale.

I file grezzi generati restano sotto `outputs/` e sono ignorati da Git. Questo
documento conserva soltanto l'evidenza compatta necessaria a spiegare la
promozione.

## Comandi principali eseguiti

```powershell
py -m compileall -q main.py machine_unlearning scripts tests
py -m pytest -q
py scripts\search_configs.py --device cpu
py main.py --data-dir data --output-dir outputs\proposed_full_run --submission-dir outputs\proposed_full_submission --config outputs\search\proposed_final_config.json --device cpu
py scripts\validate_submission.py --submission-dir outputs\proposed_full_submission --data-dir data
py scripts\search_configs.py --max-candidates 1 --seeds 92 93 94 --output-dir outputs\search\full_multiseed --proposed-config outputs\search\full_multiseed\proposed_final_config.json --device cpu
py main.py --data-dir data --output-dir outputs\final_run --submission-dir submission --config configs\final_config.json --device cpu
py scripts\validate_submission.py --submission-dir submission --data-dir data
```

## Ricerca completa, seed 92

La ricerca ha valutato otto candidati ibridi base, due varianti con gradient
ascent e il retraining. Tutti i dieci candidati ibridi sono terminati senza
failure. La soglia utility assoluta era 0,037465. Il tempo wall complessivo della
ricerca è stato 346,94 s; il tempo di selezione del retraining è escluso dal
tempo del metodo finale.

| Metodo | Configurazione | P@10 | BCE validation | BCE forget | Proxy locale | Tempo metodo (s) | Utility floor | Decisione |
|---|---|---:|---:|---:|---:|---:|---|---|
| Modello originale | artifact fornito | 0,038036 | 0,399017 | 0,397467 | n/d | n/d | baseline | riferimento utility |
| Retraining | 12 epoche | 0,038103 | 0,394272 | 0,389736 | 1,000000 | 48,18 | sì | selezionato |
| Ibrido | Fisher 0,5%, dampening 0,90, 4 step GA, 1 epoca repair, no BN | 0,038036 | 0,391818 | 0,390452 | 0,039549 | 13,17 | sì | miglior ibrido |

Il retraining ha vinto secondo la priorità dichiarata: validità, utility floor,
score multi-obiettivo, proxy locale, Precision@10 e tempo. Il valore 1 della
proxy per il retraining è atteso perché quel modello costituisce il riferimento
locale stesso; non va interpretato come risultato della MIA ufficiale.

## Replay pulito e submission

La proposta da 12 epoche è stata rieseguita in un processo separato. Le metriche
hanno coinciso con quelle della ricerca. L'esecuzione canonica finale ha misurato
50,69 s e ha scritto 51 s in `execution_time.txt`, arrotondando verso l'alto.
Il validatore ha confermato:

- esattamente `model_artifact`, `execution_time.txt` e `validation_ids.csv`;
- 13.277 ID di validation;
- ricostruzione stretta dell'artifact;
- inferenza riuscita sui dati reali.

## Stabilità full ridotta, seed 92/93/94

Per contenere il costo, il confronto multi-seed ha ripetuto Fisher completa,
retraining e la famiglia del miglior ibrido emerso dalla ricerca completa: un
candidato base e la sua variante gradient-ascent. Non è una ripetizione di tutti
i dieci candidati.

| Seed | Metodo | Variante | P@10 | BCE validation | BCE forget | Proxy locale | Tempo (s) | Floor | Selezionato |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 92 | Retraining | epoca scelta 12 | 0,038103 | 0,394272 | 0,389736 | 1,000000 | 46,96 | sì | sì |
| 92 | Ibrido | GA | 0,038036 | 0,391818 | 0,390452 | 0,039549 | 13,01 | sì | no |
| 93 | Retraining | epoca scelta dalla search | 0,038917 | 0,429825 | 0,424194 | 1,000000 | 86,87 | sì | sì |
| 93 | Ibrido | base | 0,038714 | 0,404762 | 0,400311 | 0,087317 | 12,99 | sì | no |
| 94 | Retraining | epoca scelta dalla search | 0,038134 | 0,382228 | 0,379277 | 1,000000 | 47,78 | no | no |
| 94 | Ibrido | GA | 0,038638 | 0,395528 | 0,394018 | 0,035537 | 12,89 | sì | sì |

Il retraining è stato selezionato in due seed su tre. Le sue statistiche
aggregate sono P@10 `0,038385 ± 0,000377`, BCE validation
`0,402108 ± 0,020206`, BCE forget `0,397736 ± 0,019190`, proxy locale
`1,000000 ± 0` e tempo `60,54 ± 18,62 s`. La migliore variante ibrida di ogni
seed ha P@10 medio `0,038463 ± 0,000303` e tempo medio `12,97 ± 0,05 s`, ma una
proxy locale molto più distante dal riferimento (`0,054134 ± 0,023521`).

## Decisione e limiti

La promozione del retraining a 12 epoche privilegia cancellazione interpretabile,
semplicità, riproducibilità, validità della submission e vittoria in due seed su
tre. Rimangono limiti sostanziali:

- al seed 94 il retraining non ha rispettato l'utility floor;
- split diversi rendono BCE e tempo sensibili al seed e all'epoca selezionata;
- il confronto multi-seed ha ristretto la famiglia ibrida e non ha ripetuto tutti
  i candidati;
- nessuna metrica locale sostituisce la MIA ufficiale nascosta;
- l'esito definitivo dipende dall'evaluator della challenge.

Per questi motivi lo stato nel config è
`locally_supported_official_evaluator_pending`, non “migliore in assoluto”.
