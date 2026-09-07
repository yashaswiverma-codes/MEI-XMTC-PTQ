"""
evaluate_annexml.py
-------------------
Evaluation script for AnnexML predictions.
Computes P@k, nDCG@k, PSP@k, PSnDCG@k.

PSP@k and PSnDCG@k match OFFICIAL AnnexML formula:
  Official: sum(all_numerators) / sum(all_denominators)
  NOT per-query average

Prediction file format (AnnexML output):
  label:score label:score ...   (one line per test instance)

Ground truth format (XMC libsvm):
  label0,label1,...  feat:val ...
"""

import argparse, math
from collections import defaultdict


def load_predictions(path, topk=5):
    """Load AnnexML result file.
    Supports two formats:
      1. AnnexML native: true_labels<TAB>label:score,label:score,...
      2. Our output:     label:score label:score ...
    """
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                preds.append([])
                continue
            if '\t' in line:
                score_part = line.split('\t', 1)[1]
                tokens = score_part.split(',')
            else:
                tokens = line.split()
            pairs = []
            for tok in tokens:
                tok = tok.strip()
                if ':' not in tok: continue
                lbl, sc = tok.rsplit(':', 1)
                try:
                    pairs.append((int(lbl), float(sc)))
                except ValueError:
                    continue
            pairs.sort(key=lambda x: -x[1])
            preds.append(pairs[:topk])
    return preds


def load_ground_truth(path):
    """Load XMC test file. Returns list of label sets."""
    y_true = []
    with open(path) as f:
        first = f.readline().strip().split()
        if not (len(first) == 3 and all(t.isdigit() for t in first)):
            f.seek(0)
        for line in f:
            line = line.strip()
            if not line: continue
            tok = line.split()[0]
            if ':' in tok:
                y_true.append([])
            else:
                labels = [int(l) for l in tok.split(',') if l.strip()]
                y_true.append(labels)
    return y_true


def calc_propensity(train_path, A=0.55, B=1.5):
    """
    Compute propensity scores — matches official AnnexML formula exactly.
    pw = 1 + C * (freq + B)^(-A)
    C  = (log(N) - 1) * (B+1)^A
    """
    num_inst = 0
    freqs    = defaultdict(int)
    with open(train_path) as f:
        for line in f:
            if ':' not in line: continue
            num_inst += 1
            idx    = line.find(' ')
            labels = line[:idx] if idx >= 0 else line
            for l in labels.split(','):
                l = l.strip()
                if not l: continue
                freqs[int(l)] += 1
    C          = (math.log(num_inst) - 1) * (B + 1) ** A
    pw         = {k: 1.0 + C * (v + B) ** (-A) for k, v in freqs.items()}
    default_pw = 1.0 + C * B ** (-A)
    return pw, default_pw


def precision_at_k(preds, y_true, k):
    """Standard P@k — same as official."""
    total, n = 0.0, 0
    for pred, labels in zip(preds, y_true):
        if not labels: continue
        n += 1
        pred_set = {p[0] for p in pred[:k]}
        total   += sum(1 for l in labels if l in pred_set) / k
    return total / n * 100 if n > 0 else 0.0


def ndcg_at_k(preds, y_true, k):
    """Standard nDCG@k — same as official."""
    total, n = 0.0, 0
    for pred, labels in zip(preds, y_true):
        if not labels: continue
        n += 1
        label_set = set(labels)
        dcg  = sum((1.0/math.log2(i+2))
                   for i, (l, _) in enumerate(pred[:k])
                   if l in label_set)
        idcg = sum(1.0/math.log2(i+2)
                   for i in range(min(k, len(labels))))
        if idcg > 0: total += dcg / idcg
    return total / n * 100 if n > 0 else 0.0


def psp_at_k(preds, y_true, pw, default_pw, k):
    """
    PSP@k — matches official AnnexML cumulative formula:
      PSP@k = sum_over_queries(numerator) / sum_over_queries(denominator)

    NOT per-query average.

    numerator   = sum of pw of correctly predicted labels in top-k
    denominator = sum of pw of top-k true labels sorted by pw descending
    """
    n_sum = 0.0
    d_sum = 0.0
    for pred, labels in zip(preds, y_true):
        if not labels: continue
        label_set = set(labels)

        # Numerator: pw of correct predictions in top-k
        for lbl, _ in pred[:k]:
            if lbl in label_set:
                n_sum += pw.get(lbl, default_pw)

        # Denominator: top-k true labels sorted by pw descending
        sorted_pw = sorted(
            [pw.get(l, default_pw) for l in label_set],
            reverse=True)[:k]
        d_sum += sum(sorted_pw)

    return (n_sum / d_sum * 100) if d_sum > 0 else 0.0


def psndcg_at_k(preds, y_true, pw, default_pw, k):
    """
    PSnDCG@k — matches official AnnexML cumulative formula:
      PSnDCG@k = sum_over_queries(n_dcg/idcg) / sum_over_queries(d_dcg/idcg)

    numerator DCG   = sum of pw(label) / log2(rank+2) for correct predictions
    denominator DCG = sum of pw(true_label) / log2(rank+2) for top-k true labels
                      sorted by pw descending
    idcg            = standard idcg (unweighted) for normalization
    """
    n_sum = 0.0
    d_sum = 0.0
    for pred, labels in zip(preds, y_true):
        if not labels: continue
        label_set = set(labels)

        # idcg — standard unweighted (matches official dcg_list/idcg_list)
        idcg = sum(1.0/math.log2(i+2)
                   for i in range(min(k, len(labels))))
        if idcg == 0: continue

        # Numerator DCG: pw × dcg_weight for correct predictions
        n_dcg = sum(pw.get(lbl, default_pw) / math.log2(i+2)
                    for i, (lbl, _) in enumerate(pred[:k])
                    if lbl in label_set)

        # Denominator DCG: top-k true labels sorted by pw descending
        sorted_true_pw = sorted(
            [pw.get(l, default_pw) for l in label_set],
            reverse=True)[:k]
        d_dcg = sum(p / math.log2(i+2)
                    for i, p in enumerate(sorted_true_pw))

        n_sum += n_dcg / idcg
        d_sum += d_dcg / idcg

    return (n_sum / d_sum * 100) if d_sum > 0 else 0.0


def evaluate(preds, y_true, pw, default_pw, ks=(1, 3, 5)):
    metrics = {}
    skip = sum(1 for y in y_true if not y)
    print(f"[Eval] Skipping {skip} samples with no labels")
    for k in ks:
        metrics[f'P@{k}']      = precision_at_k(preds, y_true, k)
        metrics[f'nDCG@{k}']   = ndcg_at_k(preds, y_true, k)
        metrics[f'PSP@{k}']    = psp_at_k(preds, y_true, pw, default_pw, k)
        metrics[f'PSnDCG@{k}'] = psndcg_at_k(preds, y_true, pw, default_pw, k)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred',  required=True)
    p.add_argument('--data',  required=True, help='XMC test file for ground truth')
    p.add_argument('--train', required=True, help='XMC train file for propensity')
    p.add_argument('--topk',  type=int, default=5)
    p.add_argument('--A',     type=float, default=0.55)
    p.add_argument('--B',     type=float, default=1.5)
    args = p.parse_args()

    print(f"[Eval] Loading predictions from {args.pred} ...")
    preds  = load_predictions(args.pred, args.topk)
    print(f"[Eval] Loading ground truth from {args.data} ...")
    y_true = load_ground_truth(args.data)
    print(f"[Eval] Computing propensity from {args.train} (A={args.A}, B={args.B}) ...")
    pw, default_pw = calc_propensity(args.train, args.A, args.B)

    print(f"[Eval] {len(preds)} predictions, {len(y_true)} ground truth")

    metrics = evaluate(preds, y_true, pw, default_pw)

    p1  = metrics['P@1'];      p3  = metrics['P@3'];      p5  = metrics['P@5']
    n1  = metrics['nDCG@1'];   n3  = metrics['nDCG@3'];   n5  = metrics['nDCG@5']
    sp1 = metrics['PSP@1'];    sp3 = metrics['PSP@3'];    sp5 = metrics['PSP@5']
    sn1 = metrics['PSnDCG@1']; sn3 = metrics['PSnDCG@3']; sn5 = metrics['PSnDCG@5']

    print(f"\nP@1:      {p1:.2f}    P@3:      {p3:.2f}    P@5:      {p5:.2f}")
    print(f"nDCG@1:   {n1:.2f}    nDCG@3:   {n3:.2f}    nDCG@5:   {n5:.2f}")
    print(f"PSP@1:    {sp1:.2f}    PSP@3:    {sp3:.2f}    PSP@5:    {sp5:.2f}")
    print(f"PSnDCG@1: {sn1:.2f}    PSnDCG@3: {sn3:.2f}    PSnDCG@5: {sn5:.2f}")


if __name__ == '__main__':
    main()
