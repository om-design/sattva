# sattva_ga_engine.py

import numpy as np
from itertools import combinations
from collections import defaultdict

# ---------------------------
# Basic Multivector Structure
# ---------------------------

class Multivector:
    def __init__(self, components):
        self.components = components  # dict: blade -> value

    def __add__(self, other):
        result = self.components.copy()
        for k, v in other.components.items():
            result[k] = result.get(k, 0) + v
        return Multivector(result)

    def scalar(self):
        return self.components.get((), 0.0)

    def bivector_norm(self):
        return np.sqrt(sum(v**2 for k, v in self.components.items() if len(k) == 2))

# ---------------------------
# Event Node
# ---------------------------

class EventNode:
    def __init__(self, event_id, vector, timestamp, event_type):
        self.id = event_id
        self.vector = np.array(vector)
        self.timestamp = timestamp
        self.type = event_type

# ---------------------------
# Deal Geometry
# ---------------------------

class DealGeometry:
    def __init__(self, events):
        self.events = events
        self.relationships = []
        self.motifs = []

    def build_relationship_vectors(self):
        for e1, e2 in combinations(self.events, 2):
            r = e2.vector - e1.vector
            dt = e2.timestamp - e1.timestamp
            self.relationships.append((e1.id, e2.id, r, dt))

    def extract_triangular_motifs(self):
        for e1, e2, e3 in combinations(self.events, 3):
            v1 = e2.vector - e1.vector
            v2 = e3.vector - e2.vector
            v3 = e1.vector - e3.vector
            closure = np.linalg.norm(v1 + v2 + v3)
            self.motifs.append({
                "nodes": (e1.id, e2.id, e3.id),
                "closure_error": closure
            })

# ---------------------------
# Primitive Bank
# ---------------------------

class PrimitiveBank:
    def __init__(self):
        self.primitives = []

    def add_primitive(self, signature, weight):
        self.primitives.append({
            "signature": signature,
            "weight": weight
        })

    def match(self, deal_geometry):
        score = 0.0
        for p in self.primitives:
            for m in deal_geometry.motifs:
                dist = abs(m["closure_error"] - p["signature"])
                score += p["weight"] * np.exp(-dist)
        return score
