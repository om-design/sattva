# sattva_mvp.py

import numpy as np
from itertools import combinations
from collections import defaultdict

class DealStream:
    def __init__(self, events):
        self.events = events

    def extract_motifs(self):
        motifs = []
        for e1, e2, e3 in combinations(self.events, 3):
            dt1 = e2["t"] - e1["t"]
            dt2 = e3["t"] - e2["t"]
            pattern = (e1["type"], e2["type"], e3["type"])
            normalized_time = (dt1 / (dt1 + dt2 + 1e-6),
                               dt2 / (dt1 + dt2 + 1e-6))
            motifs.append((pattern, normalized_time))
        return motifs

class MotifLearner:
    def __init__(self):
        self.counts = defaultdict(int)
        self.total = 0

    def fit(self, deal_streams):
        for ds in deal_streams:
            motifs = ds.extract_motifs()
            for m in motifs:
                self.counts[m[0]] += 1
            self.total += 1

    def get_primitives(self, min_support=5):
        return {k: v for k, v in self.counts.items() if v >= min_support}

class StreamScorer:
    def __init__(self, primitives):
        self.primitives = primitives

    def score(self, stream):
        motifs = stream.extract_motifs()
        score = 0
        for m in motifs:
            if m[0] in self.primitives:
                score += 1
        return score
