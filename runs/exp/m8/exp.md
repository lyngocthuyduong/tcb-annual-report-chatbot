# Retrieval experiments

answerable=85 unanswerable=19 sets=['dev', 'val']

## E2 BM25-only tokenizer
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| default | dev | 57 | 0.8421 | 0.9123 | 0.9298 | 0.9825 | 0.7347 |
| default | val | 28 | 0.9286 | 0.9286 | 0.9643 | 1.0 | 0.8038 |
| default | all | 85 | 0.8706 | 0.9176 | 0.9412 | 0.9882 | 0.7575 |
| vi | dev | 57 | 0.8947 | 0.9123 | 0.9298 | 0.9825 | 0.7504 |
| vi | val | 28 | 0.9286 | 0.9286 | 0.9286 | 1.0 | 0.8419 |
| vi | all | 85 | 0.9059 | 0.9176 | 0.9294 | 0.9882 | 0.7806 |
| vi+bigram | dev | 57 | 0.9123 | 0.9123 | 0.9474 | 0.9825 | 0.7916 |
| vi+bigram | val | 28 | 0.9286 | 0.9286 | 0.9286 | 1.0 | 0.8063 |
| vi+bigram | all | 85 | 0.9176 | 0.9176 | 0.9412 | 0.9882 | 0.7964 |

## E3 first-stage fusion
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| bm25 | dev | 57 | 0.8947 | 0.9123 | 0.9298 | 0.9825 | 0.7417 |
| bm25 | val | 28 | 0.9286 | 0.9286 | 0.9286 | 1.0 | 0.824 |
| bm25 | all | 85 | 0.9059 | 0.9176 | 0.9294 | 0.9882 | 0.7688 |
| dense | dev | 57 | 0.5263 | 0.5965 | 0.6842 | 0.7544 | 0.3767 |
| dense | val | 28 | 0.6429 | 0.7143 | 0.8214 | 0.8571 | 0.5153 |
| dense | all | 85 | 0.5647 | 0.6353 | 0.7294 | 0.7882 | 0.4224 |
| rrf60 | dev | 57 | 0.7368 | 0.7895 | 0.9474 | 0.9474 | 0.5476 |
| rrf60 | val | 28 | 0.8929 | 0.9643 | 0.9643 | 0.9643 | 0.7125 |
| rrf60 | all | 85 | 0.7882 | 0.8471 | 0.9529 | 0.9529 | 0.6019 |
| rsf | dev | 57 | 0.8246 | 0.8772 | 0.9474 | 0.9474 | 0.6291 |
| rsf | val | 28 | 0.9286 | 0.9286 | 0.9643 | 0.9643 | 0.7063 |
| rsf | all | 85 | 0.8588 | 0.8941 | 0.9529 | 0.9529 | 0.6545 |

## E3b RRF k
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| 10 | dev | 57 | 0.8421 | 0.8772 | 0.9474 | 0.9474 | 0.5741 |
| 10 | val | 28 | 0.9643 | 0.9643 | 0.9643 | 0.9643 | 0.7226 |
| 10 | all | 85 | 0.8824 | 0.9059 | 0.9529 | 0.9529 | 0.623 |
| 30 | dev | 57 | 0.7544 | 0.7895 | 0.9474 | 0.9474 | 0.5485 |
| 30 | val | 28 | 0.8929 | 0.9643 | 0.9643 | 0.9643 | 0.7143 |
| 30 | all | 85 | 0.8 | 0.8471 | 0.9529 | 0.9529 | 0.6031 |
| 60 | dev | 57 | 0.7368 | 0.7895 | 0.9474 | 0.9474 | 0.5476 |
| 60 | val | 28 | 0.8929 | 0.9643 | 0.9643 | 0.9643 | 0.7125 |
| 60 | all | 85 | 0.7882 | 0.8471 | 0.9529 | 0.9529 | 0.6019 |
| 100 | dev | 57 | 0.7368 | 0.7895 | 0.9474 | 0.9474 | 0.5505 |
| 100 | val | 28 | 0.8929 | 0.9643 | 0.9643 | 0.9643 | 0.7125 |
| 100 | all | 85 | 0.7882 | 0.8471 | 0.9529 | 0.9529 | 0.6039 |

## E4 final k of the fused list
| variant | set | n | R@5 | R@8 | R@10 | R@12 | R@15 | R@20 |
|---|---|---|---|---|---|---|---|---|
| fused | dev | 57 | 0.7368 | 0.7895 | 0.8421 | 0.9123 | 0.9298 | 0.9474 |
| fused | val | 28 | 0.8929 | 0.9643 | 0.9643 | 0.9643 | 0.9643 | 0.9643 |
| fused | all | 85 | 0.7882 | 0.8471 | 0.8824 | 0.9294 | 0.9412 | 0.9529 |

## E5 glossary expansion (final top-k)
| variant | set | n | R@5 | R@12 | mrr |
|---|---|---|---|---|---|
| expansion_on | dev | 57 | 0.7368 | 0.9123 | 0.5454 |
| expansion_on | val | 28 | 0.8929 | 0.9643 | 0.7125 |
| expansion_on | all | 85 | 0.7882 | 0.9294 | 0.6004 |
| expansion_off | dev | 57 | 0.7368 | 0.8947 | 0.5474 |
| expansion_off | val | 28 | 0.8929 | 0.9643 | 0.7125 |
| expansion_off | all | 85 | 0.7882 | 0.9176 | 0.6018 |

## E8 KPI label ranking (final top-k)
| variant | set | n | R@3 | R@5 | R@12 | mrr |
|---|---|---|---|---|---|---|
| label_on | dev | 57 | 0.7193 | 0.7895 | 0.9123 | 0.5739 |
| label_on | val | 28 | 0.8571 | 0.8929 | 0.9643 | 0.6763 |
| label_on | all | 85 | 0.7647 | 0.8235 | 0.9294 | 0.6076 |
| label_off | dev | 57 | 0.6667 | 0.7368 | 0.9123 | 0.5454 |
| label_off | val | 28 | 0.8571 | 0.8929 | 0.9643 | 0.7125 |
| label_off | all | 85 | 0.7294 | 0.7882 | 0.9294 | 0.6004 |

## Suggested decisions (not applied)
```json
{
 "E2": {
  "best_bm25_only_R@20": "default",
  "production": "vi",
  "R@20": {
   "default": 0.9412,
   "vi": 0.9294,
   "vi+bigram": 0.9412
  },
  "action": "none (keep 'vi'; switching needs >= 2 pts R@20)"
 },
 "E3": {
  "R@20": {
   "bm25": 0.9294,
   "dense": 0.7294,
   "rrf60": 0.9529,
   "rsf": 0.9529
  },
  "keep": "rrf60",
  "rule": "keep RRF unless another method is >= 5 pts better at R@20"
 },
 "E3b": {
  "keep_k": 60,
  "beats_60_by_5pts_on_every_set": [],
  "rule": "keep 60 unless >= 5 pts R@20 on dev AND val (every set provided)",
  "note": "with a single set the rule cannot confirm 'dev and val' separately"
 },
 "E4": {
  "recall_at_k": {
   "5": 0.7882,
   "8": 0.8471,
   "10": 0.8824,
   "12": 0.9294,
   "15": 0.9412,
   "20": 0.9529
  },
  "choose_k": 15,
  "current": 12,
  "rule": "smallest k within 2 pts of the best recall (k blocks go to the LLM; more k = longer prompt)"
 },
 "E5": {
  "regressions": [],
  "gains": [
   "dev:h-kpi-car"
  ],
  "keep_expansion": true,
  "rule": "keep if no regressions (gold lost from the top-12 only when expansion is on)"
 },
 "E8": {
  "regressions": [],
  "gains": [],
  "keep_label_ranking": true,
  "rule": "keep if no gold page leaves the top-12 when label ranking is on"
 }
}
```
