# Retrieval experiments

answerable=85 unanswerable=19 sets=['dev', 'val']

## E2 BM25-only tokenizer
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| default | dev | 57 | 0.8421 | 0.8947 | 0.9298 | 0.9825 | 0.7341 |
| default | val | 28 | 0.8929 | 0.9286 | 0.9643 | 1.0 | 0.7726 |
| default | all | 85 | 0.8588 | 0.9059 | 0.9412 | 0.9882 | 0.7468 |
| vi | dev | 57 | 0.8947 | 0.9123 | 0.9298 | 0.9825 | 0.7505 |
| vi | val | 28 | 0.8929 | 0.9286 | 0.9286 | 1.0 | 0.8122 |
| vi | all | 85 | 0.8941 | 0.9176 | 0.9294 | 0.9882 | 0.7708 |
| vi+bigram | dev | 57 | 0.9123 | 0.9123 | 0.9474 | 0.9825 | 0.7916 |
| vi+bigram | val | 28 | 0.9286 | 0.9286 | 0.9286 | 1.0 | 0.7956 |
| vi+bigram | all | 85 | 0.9176 | 0.9176 | 0.9412 | 0.9882 | 0.7929 |

## E3 first-stage fusion
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| bm25 | dev | 57 | 0.8947 | 0.9123 | 0.9298 | 0.9825 | 0.7417 |
| bm25 | val | 28 | 0.8929 | 0.9286 | 0.9286 | 1.0 | 0.7943 |
| bm25 | all | 85 | 0.8941 | 0.9176 | 0.9294 | 0.9882 | 0.759 |
| dense | dev | 57 | 0.5088 | 0.5789 | 0.6667 | 0.7544 | 0.3706 |
| dense | val | 28 | 0.6071 | 0.6786 | 0.8214 | 0.8571 | 0.5074 |
| dense | all | 85 | 0.5412 | 0.6118 | 0.7176 | 0.7882 | 0.4156 |
| rrf60 | dev | 57 | 0.7193 | 0.7719 | 0.9298 | 0.9474 | 0.5403 |
| rrf60 | val | 28 | 0.8929 | 0.9286 | 0.9643 | 0.9643 | 0.6852 |
| rrf60 | all | 85 | 0.7765 | 0.8235 | 0.9412 | 0.9529 | 0.5881 |
| rsf | dev | 57 | 0.807 | 0.8772 | 0.9474 | 0.9649 | 0.6233 |
| rsf | val | 28 | 0.9286 | 0.9286 | 0.9643 | 0.9643 | 0.6956 |
| rsf | all | 85 | 0.8471 | 0.8941 | 0.9529 | 0.9647 | 0.6471 |

## E3b RRF k
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| 10 | dev | 57 | 0.807 | 0.8596 | 0.9474 | 0.9474 | 0.5679 |
| 10 | val | 28 | 0.9643 | 0.9643 | 0.9643 | 0.9643 | 0.694 |
| 10 | all | 85 | 0.8588 | 0.8941 | 0.9529 | 0.9529 | 0.6095 |
| 30 | dev | 57 | 0.7368 | 0.7719 | 0.9298 | 0.9474 | 0.5412 |
| 30 | val | 28 | 0.8929 | 0.9286 | 0.9643 | 0.9643 | 0.6834 |
| 30 | all | 85 | 0.7882 | 0.8235 | 0.9412 | 0.9529 | 0.5881 |
| 60 | dev | 57 | 0.7193 | 0.7719 | 0.9298 | 0.9474 | 0.5403 |
| 60 | val | 28 | 0.8929 | 0.9286 | 0.9643 | 0.9643 | 0.6852 |
| 60 | all | 85 | 0.7765 | 0.8235 | 0.9412 | 0.9529 | 0.5881 |
| 100 | dev | 57 | 0.7193 | 0.7719 | 0.9298 | 0.9474 | 0.5433 |
| 100 | val | 28 | 0.8929 | 0.9286 | 0.9643 | 0.9643 | 0.6852 |
| 100 | all | 85 | 0.7765 | 0.8235 | 0.9412 | 0.9529 | 0.59 |

## E4 final k of the fused list
| variant | set | n | R@5 | R@8 | R@10 | R@12 | R@15 | R@20 |
|---|---|---|---|---|---|---|---|---|
| fused | dev | 57 | 0.7193 | 0.7719 | 0.8246 | 0.9123 | 0.9123 | 0.9298 |
| fused | val | 28 | 0.8929 | 0.9286 | 0.9643 | 0.9643 | 0.9643 | 0.9643 |
| fused | all | 85 | 0.7765 | 0.8235 | 0.8706 | 0.9294 | 0.9294 | 0.9412 |

## E5 glossary expansion (final top-k)
| variant | set | n | R@5 | R@20 | mrr |
|---|---|---|---|---|---|
| expansion_on | dev | 57 | 0.7193 | 0.9298 | 0.5395 |
| expansion_on | val | 28 | 0.8929 | 0.9643 | 0.6852 |
| expansion_on | all | 85 | 0.7765 | 0.9412 | 0.5875 |
| expansion_off | dev | 57 | 0.7193 | 0.9298 | 0.5427 |
| expansion_off | val | 28 | 0.8929 | 0.9643 | 0.6852 |
| expansion_off | all | 85 | 0.7765 | 0.9412 | 0.5897 |

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
   "dense": 0.7176,
   "rrf60": 0.9412,
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
   "5": 0.7765,
   "8": 0.8235,
   "10": 0.8706,
   "12": 0.9294,
   "15": 0.9294,
   "20": 0.9412
  },
  "choose_k": 12,
  "current": 20,
  "rule": "smallest k within 2 pts of the best recall (k blocks go to the LLM; more k = longer prompt)"
 },
 "E5": {
  "regressions": [],
  "gains": [],
  "keep_expansion": true,
  "rule": "keep if no regressions (gold lost from the top-20 only when expansion is on)"
 }
}
```
