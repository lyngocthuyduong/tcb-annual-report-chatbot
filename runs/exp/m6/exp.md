# Retrieval experiments

answerable=85 unanswerable=19 sets=['dev', 'val']

## E2 BM25-only tokenizer
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| default | dev | 57 | 0.8246 | 0.8772 | 0.9123 | 0.9825 | 0.6989 |
| default | val | 28 | 0.8929 | 0.8929 | 0.9286 | 1.0 | 0.7532 |
| default | all | 85 | 0.8471 | 0.8824 | 0.9176 | 0.9882 | 0.7168 |
| vi | dev | 57 | 0.8596 | 0.9123 | 0.9298 | 0.9649 | 0.7144 |
| vi | val | 28 | 0.8929 | 0.9286 | 0.9286 | 0.9643 | 0.8215 |
| vi | all | 85 | 0.8706 | 0.9176 | 0.9294 | 0.9647 | 0.7497 |
| vi+bigram | dev | 57 | 0.9123 | 0.9123 | 0.9474 | 0.9649 | 0.7927 |
| vi+bigram | val | 28 | 0.8929 | 0.9286 | 0.9286 | 0.9643 | 0.8403 |
| vi+bigram | all | 85 | 0.9059 | 0.9176 | 0.9412 | 0.9647 | 0.8083 |

## E3 first-stage fusion
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| bm25 | dev | 57 | 0.8772 | 0.9123 | 0.9298 | 0.9649 | 0.7114 |
| bm25 | val | 28 | 0.8929 | 0.9286 | 0.9286 | 0.9643 | 0.8155 |
| bm25 | all | 85 | 0.8824 | 0.9176 | 0.9294 | 0.9647 | 0.7457 |
| dense | dev | 57 | 0.4211 | 0.5088 | 0.6316 | 0.7544 | 0.3586 |
| dense | val | 28 | 0.5714 | 0.6429 | 0.7857 | 0.8571 | 0.4786 |
| dense | all | 85 | 0.4706 | 0.5529 | 0.6824 | 0.7882 | 0.3981 |
| rrf60 | dev | 57 | 0.7193 | 0.7895 | 0.9298 | 0.9474 | 0.519 |
| rrf60 | val | 28 | 0.8214 | 0.9286 | 0.9286 | 0.9643 | 0.692 |
| rrf60 | all | 85 | 0.7529 | 0.8353 | 0.9294 | 0.9529 | 0.576 |
| rsf | dev | 57 | 0.807 | 0.8772 | 0.9298 | 0.9298 | 0.5927 |
| rsf | val | 28 | 0.9286 | 0.9286 | 0.9286 | 0.9286 | 0.7155 |
| rsf | all | 85 | 0.8471 | 0.8941 | 0.9294 | 0.9294 | 0.6332 |

## E3b RRF k
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| 10 | dev | 57 | 0.8246 | 0.8421 | 0.9298 | 0.9474 | 0.5592 |
| 10 | val | 28 | 0.8929 | 0.9286 | 0.9286 | 0.9643 | 0.7128 |
| 10 | all | 85 | 0.8471 | 0.8706 | 0.9294 | 0.9529 | 0.6098 |
| 30 | dev | 57 | 0.7193 | 0.7895 | 0.9298 | 0.9474 | 0.5215 |
| 30 | val | 28 | 0.8214 | 0.9286 | 0.9286 | 0.9643 | 0.6929 |
| 30 | all | 85 | 0.7529 | 0.8353 | 0.9294 | 0.9529 | 0.5779 |
| 60 | dev | 57 | 0.7193 | 0.7895 | 0.9298 | 0.9474 | 0.519 |
| 60 | val | 28 | 0.8214 | 0.9286 | 0.9286 | 0.9643 | 0.692 |
| 60 | all | 85 | 0.7529 | 0.8353 | 0.9294 | 0.9529 | 0.576 |
| 100 | dev | 57 | 0.7018 | 0.7895 | 0.9298 | 0.9474 | 0.5171 |
| 100 | val | 28 | 0.8214 | 0.9286 | 0.9286 | 0.9643 | 0.6861 |
| 100 | all | 85 | 0.7412 | 0.8353 | 0.9294 | 0.9529 | 0.5728 |

## E4 final k of the fused list
| variant | set | n | R@5 | R@8 | R@10 | R@12 | R@15 | R@20 |
|---|---|---|---|---|---|---|---|---|
| fused | dev | 57 | 0.7193 | 0.7895 | 0.8246 | 0.8772 | 0.8947 | 0.9298 |
| fused | val | 28 | 0.8214 | 0.9286 | 0.9286 | 0.9286 | 0.9286 | 0.9286 |
| fused | all | 85 | 0.7529 | 0.8353 | 0.8588 | 0.8941 | 0.9059 | 0.9294 |

## E5 glossary expansion (final top-k)
| variant | set | n | R@5 | R@15 | mrr |
|---|---|---|---|---|---|
| expansion_on | dev | 57 | 0.7193 | 0.8947 | 0.5165 |
| expansion_on | val | 28 | 0.8214 | 0.9286 | 0.6911 |
| expansion_on | all | 85 | 0.7529 | 0.9059 | 0.5741 |
| expansion_off | dev | 57 | 0.7193 | 0.8947 | 0.5224 |
| expansion_off | val | 28 | 0.8214 | 0.9286 | 0.6881 |
| expansion_off | all | 85 | 0.7529 | 0.9059 | 0.577 |

## Suggested decisions (not applied)
```json
{
 "E2": {
  "best_bm25_only_R@20": "vi+bigram",
  "production": "vi",
  "R@20": {
   "default": 0.9176,
   "vi": 0.9294,
   "vi+bigram": 0.9412
  },
  "action": "none (keep 'vi'; switching needs >= 2 pts R@20)"
 },
 "E3": {
  "R@20": {
   "bm25": 0.9294,
   "dense": 0.6824,
   "rrf60": 0.9294,
   "rsf": 0.9294
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
   "5": 0.7529,
   "8": 0.8353,
   "10": 0.8588,
   "12": 0.8941,
   "15": 0.9059,
   "20": 0.9294
  },
  "choose_k": 20,
  "current": 15,
  "rule": "smallest k within 2 pts of the best recall (k blocks go to the LLM; more k = longer prompt)"
 },
 "E5": {
  "regressions": [],
  "gains": [],
  "keep_expansion": true,
  "rule": "keep if no regressions (gold lost from the top-15 only when expansion is on)"
 }
}
```
