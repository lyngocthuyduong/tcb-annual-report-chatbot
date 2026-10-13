# Retrieval experiments

answerable=85 unanswerable=19 sets=['dev', 'val']

## E2 BM25-only tokenizer
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| default | dev | 57 | 0.8421 | 0.9123 | 0.9474 | 1.0 | 0.7071 |
| default | val | 28 | 0.8929 | 0.8929 | 0.9286 | 1.0 | 0.7563 |
| default | all | 85 | 0.8588 | 0.9059 | 0.9412 | 1.0 | 0.7233 |
| vi | dev | 57 | 0.8596 | 0.9298 | 0.9474 | 0.9649 | 0.7369 |
| vi | val | 28 | 0.8929 | 0.9286 | 0.9286 | 0.9643 | 0.8036 |
| vi | all | 85 | 0.8706 | 0.9294 | 0.9412 | 0.9647 | 0.7589 |
| vi+bigram | dev | 57 | 0.9298 | 0.9298 | 0.9474 | 0.9649 | 0.8079 |
| vi+bigram | val | 28 | 0.8929 | 0.9286 | 0.9286 | 0.9643 | 0.8224 |
| vi+bigram | all | 85 | 0.9176 | 0.9294 | 0.9412 | 0.9647 | 0.8127 |

## E3 first-stage fusion
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| bm25 | dev | 57 | 0.8947 | 0.9298 | 0.9474 | 0.9649 | 0.7355 |
| bm25 | val | 28 | 0.8929 | 0.9286 | 0.9286 | 0.9643 | 0.7977 |
| bm25 | all | 85 | 0.8941 | 0.9294 | 0.9412 | 0.9647 | 0.756 |
| dense | dev | 57 | 0.4561 | 0.5263 | 0.6316 | 0.7895 | 0.3673 |
| dense | val | 28 | 0.5357 | 0.6786 | 0.75 | 0.8571 | 0.4778 |
| dense | all | 85 | 0.4824 | 0.5765 | 0.6706 | 0.8118 | 0.4037 |
| rrf60 | dev | 57 | 0.7368 | 0.8246 | 0.9298 | 0.9474 | 0.5363 |
| rrf60 | val | 28 | 0.8214 | 0.9286 | 0.9286 | 0.9286 | 0.6554 |
| rrf60 | all | 85 | 0.7647 | 0.8588 | 0.9294 | 0.9412 | 0.5755 |
| rsf | dev | 57 | 0.8772 | 0.8947 | 0.9298 | 0.9474 | 0.6127 |
| rsf | val | 28 | 0.9286 | 0.9286 | 0.9286 | 0.9286 | 0.7036 |
| rsf | all | 85 | 0.8941 | 0.9059 | 0.9294 | 0.9412 | 0.6426 |

## E3b RRF k
| variant | set | n | R@5 | R@8 | R@20 | R@40 | mrr |
|---|---|---|---|---|---|---|---|
| 10 | dev | 57 | 0.8421 | 0.8947 | 0.9298 | 0.9474 | 0.566 |
| 10 | val | 28 | 0.8929 | 0.9286 | 0.9286 | 0.9286 | 0.6583 |
| 10 | all | 85 | 0.8588 | 0.9059 | 0.9294 | 0.9412 | 0.5964 |
| 30 | dev | 57 | 0.7544 | 0.8421 | 0.9298 | 0.9474 | 0.5393 |
| 30 | val | 28 | 0.8214 | 0.9286 | 0.9286 | 0.9286 | 0.6554 |
| 30 | all | 85 | 0.7765 | 0.8706 | 0.9294 | 0.9412 | 0.5775 |
| 60 | dev | 57 | 0.7368 | 0.8246 | 0.9298 | 0.9474 | 0.5363 |
| 60 | val | 28 | 0.8214 | 0.9286 | 0.9286 | 0.9286 | 0.6554 |
| 60 | all | 85 | 0.7647 | 0.8588 | 0.9294 | 0.9412 | 0.5755 |
| 100 | dev | 57 | 0.7193 | 0.8246 | 0.9298 | 0.9474 | 0.5331 |
| 100 | val | 28 | 0.8214 | 0.9286 | 0.9286 | 0.9286 | 0.6554 |
| 100 | all | 85 | 0.7529 | 0.8588 | 0.9294 | 0.9412 | 0.5734 |

## E4 final k of the fused list
| variant | set | n | R@5 | R@8 | R@10 | R@12 | R@15 | R@20 |
|---|---|---|---|---|---|---|---|---|
| fused | dev | 57 | 0.7368 | 0.8246 | 0.8421 | 0.8947 | 0.9298 | 0.9298 |
| fused | val | 28 | 0.8214 | 0.9286 | 0.9286 | 0.9286 | 0.9286 | 0.9286 |
| fused | all | 85 | 0.7647 | 0.8588 | 0.8706 | 0.9059 | 0.9294 | 0.9294 |

## E5 glossary expansion (final top-k)
| variant | set | n | R@5 | R@10 | mrr |
|---|---|---|---|---|---|
| expansion_on | dev | 57 | 0.7368 | 0.8421 | 0.5286 |
| expansion_on | val | 28 | 0.8214 | 0.9286 | 0.6554 |
| expansion_on | all | 85 | 0.7647 | 0.8706 | 0.5704 |
| expansion_off | dev | 57 | 0.7368 | 0.8421 | 0.5321 |
| expansion_off | val | 28 | 0.8214 | 0.9286 | 0.6614 |
| expansion_off | all | 85 | 0.7647 | 0.8706 | 0.5747 |

## Suggested decisions (not applied)
```json
{
 "E2": {
  "best_bm25_only_R@20": "vi",
  "production": "vi",
  "R@20": {
   "default": 0.9412,
   "vi": 0.9412,
   "vi+bigram": 0.9412
  },
  "action": "none (keep 'vi'; switching needs >= 2 pts R@20)"
 },
 "E3": {
  "R@20": {
   "bm25": 0.9412,
   "dense": 0.6706,
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
   "5": 0.7647,
   "8": 0.8588,
   "10": 0.8706,
   "12": 0.9059,
   "15": 0.9294,
   "20": 0.9294
  },
  "choose_k": 15,
  "current": 10,
  "rule": "smallest k within 2 pts of the best recall (k blocks go to the LLM; more k = longer prompt)"
 },
 "E5": {
  "regressions": [],
  "gains": [],
  "keep_expansion": true,
  "rule": "keep if no regressions (gold lost from the top-10 only when expansion is on)"
 }
}
```
