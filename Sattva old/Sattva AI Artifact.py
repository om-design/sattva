# SATTVA: Sequenced Learning AI Artifact
# Python Notebook / Artifact
# Purpose: combine descriptive text, proofs, and code for continued development

# -----------------------------------------
# Section 1: Overview & Conceptual Description
# -----------------------------------------
overview = """
SATTVA is a sequenced-learning, multi-scalar AI architecture inspired by human cognition.
Key concepts include:
- Physics-based primitive formation (early learning phase)
- Nested geometric structures mimicking visual cortex representations
- Resonance across primitives for pattern recognition and creativity
- Threshold-based flip mechanisms for self-correction
- Cascade collapse for rapid restructuring
"""

# -----------------------------------------
# Section 2: Primitive Formation (Physics-Based)
# -----------------------------------------
import numpy as np

class Primitive:
    def __init__(self, data, depth=1.0):
        self.data = data  # atomic observation or feature
        self.depth = depth  # strength / persistence
        self.coupling_strength = 1.0  # influence on higher-level patterns
        self.active = False

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

# Proof Sketch: Physics-Based Primitives are Unique and Verifiable
# Assume primitive corresponds to unambiguous measurement (e.g., distance, energy state)
# They are invariant under transformations preserving physical constraints

# Example primitive creation
primitive_example = Primitive(data=np.array([1.0, 0.0, 0.0]))

# -----------------------------------------
# Section 3: Nested Multi-Scalar Structures
# -----------------------------------------
class MultiScalarNode:
    def __init__(self, primitives=None):
        self.primitives = primitives or []
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def resonance_score(self, input_pattern):
        # Compute resonance between input and stored primitives
        scores = [np.dot(p.data, input_pattern) for p in self.primitives]
        child_scores = [child.resonance_score(input_pattern) for child in self.children]
        total_score = np.sum(scores) + np.sum(child_scores)
        return total_score

# Example structure
root_node = MultiScalarNode(primitives=[primitive_example])
child_node = MultiScalarNode(primitives=[Primitive(data=np.array([0.5, 0.5, 0.0]))])
root_node.add_child(child_node)

# -----------------------------------------
# Section 4: Threshold Flip Mechanism
# -----------------------------------------
class ThresholdFlip:
    def __init__(self, primitive, flip_threshold=0.6):
        self.primitive = primitive
        self.flip_threshold = flip_threshold
        self.mainstream_confidence = 1.0
        self.counter_confidence = 0.0
        self.has_flipped = False

    def log_counter_evidence(self, weight=0.1):
        self.counter_confidence += weight
        total = self.mainstream_confidence + self.counter_confidence
        if (self.counter_confidence / total) > self.flip_threshold and not self.has_flipped:
            self.flip()

    def flip(self):
        self.primitive.data = -self.primitive.data  # negate pattern as proof-of-concept
        self.has_flipped = True

# Example usage
flip_agent = ThresholdFlip(primitive_example)
flip_agent.log_counter_evidence(weight=0.7)

# -----------------------------------------
# Section 5: Cascade Collapse
# -----------------------------------------
class CascadeCollapse:
    def __init__(self, dependency_graph):
        self.dependency_graph = dependency_graph  # dict of primitive -> dependent primitives

    def trigger(self, root_primitive):
        # recursively reduce confidence in dependent primitives
        if root_primitive in self.dependency_graph:
            for dep in self.dependency_graph[root_primitive]:
                dep.coupling_strength *= 0.3
                self.trigger(dep)

# Example dependency graph
dep_graph = {primitive_example: [child_node.primitives[0]]}
cascade = CascadeCollapse(dep_graph)
cascade.trigger(primitive_example)

# -----------------------------------------
# Section 6: Resonance-Based Pattern Matching
# -----------------------------------------
def find_best_match(root_node, input_pattern):
    # Traverse tree and return node with max resonance
    nodes = [root_node]
    best_score = -np.inf
    best_node = None
    while nodes:
        node = nodes.pop()
        score = node.resonance_score(input_pattern)
        if score > best_score:
            best_score = score
            best_node = node
        nodes.extend(node.children)
    return best_node, best_score

# Example pattern
input_pattern = np.array([1.0, 0.0, 0.0])
best_node, score = find_best_match(root_node, input_pattern)
print(f"Best node score: {score}")

# -----------------------------------------
# Section 7: practical core utilities
# -----------------------------------------
import numpy as np

# ---------- Utilities ----------
def l2_similarity(a: np.ndarray, b: np.ndarray, eps=1e-9):
    """Cosine-like similarity robust to zero vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def geometric_overlap(pattern_a: np.ndarray, pattern_b: np.ndarray):
    """
    A simple geometric overlap for vector-pattern primitives.
    Works for vectors, small point clouds (flattened).
    Returns value in [0,1].
    """
    # If inputs are matrices (N x D), compute averaged best-match cosine
    a = np.atleast_2d(pattern_a)
    b = np.atleast_2d(pattern_b)
    # Normalize rows
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    # pairwise similarity matrix
    sim = np.dot(a_norm, b_norm.T)  # shape (na, nb)
    # greedy-match score: average of row-wise max
    row_max = np.max(sim, axis=1)
    return float(np.mean(row_max.clip(min=0.0)))

# ---------- Primitives ----------
class Primitive:
    def __init__(self, name, data: np.ndarray, depth=1.0):
        self.name = name
        self.data = np.asarray(data, dtype=float)
        self.depth = float(depth)
        self.coupling_strength = 1.0
        self.active = True

    def verify(self, input_value: np.ndarray, tol=1e-6) -> bool:
        """Deterministic check for physics-based primitives: exact-ish match."""
        input_value = np.asarray(input_value, dtype=float)
        if self.data.shape != input_value.shape:
            return False
        return float(np.linalg.norm(self.data - input_value)) <= tol

    def perturb(self, delta: np.ndarray, factor=1.0):
        """Apply small change to primitive data (useful for testing)."""
        self.data = self.data + factor * np.asarray(delta)

# ---------- Geometric Primitive for multi-scale ----------
class GeometricPrimitive(Primitive):
    def __init__(self, name, pattern: np.ndarray, scale=1.0, depth=1.0):
        super().__init__(name, np.asarray(pattern, dtype=float), depth=depth)
        self.scale = float(scale)

    def compare(self, other) -> float:
        """Compute geometric resonance (overlap * scale factor)."""
        ov = geometric_overlap(self.data, other.data)
        scale_factor = min(self.scale, getattr(other, "scale", 1.0))
        return float(ov * scale_factor)

# ---------- Resonance Engine ----------
class ResonanceEngine:
    def __init__(self):
        self.known_patterns = []  # list of lists of GeometricPrimitive or Primitive

    def add_pattern(self, primitives):
        self.known_patterns.append(primitives)

    def evaluate_new_input(self, input_primitives):
        """
        Evaluate input_primitives against stored patterns.
        input_primitives: list of GeometricPrimitive (same order or comparable length)
        Returns (best_score, best_index)
        """
        if not self.known_patterns:
            return 0.0, None
        scores = []
        for idx, pattern in enumerate(self.known_patterns):
            # pairwise matching: align by position, fallback to greedy pairing
            pair_count = min(len(pattern), len(input_primitives))
            if pair_count == 0:
                scores.append(0.0)
                continue
            s = 0.0
            for i in range(pair_count):
                a = pattern[i]
                b = input_primitives[i]
                # prefer geometric compare if available
                if hasattr(a, "compare"):
                    s += a.compare(b)
                else:
                    s += l2_similarity(np.ravel(a.data), np.ravel(b.data))
            # normalize by count and weight by average depth
            depth_weight = np.mean([getattr(p, "depth", 1.0) for p in pattern])
            scores.append((s / pair_count) * depth_weight)
        best_idx = int(np.argmax(scores))
        return float(scores[best_idx]), best_idx

# ---------- Threshold Flip (EMA + evidence) ----------
class ThresholdFlip:
    def __init__(self, primitive: Primitive, flip_threshold=0.6, ema_alpha=0.2):
        self.primitive = primitive
        self.flip_threshold = float(flip_threshold)
        self.ema_alpha = float(ema_alpha)
        self.evidence_ema = 0.0  # tracks counter-evidence in [0,1]
        self.has_flipped = False

    def log_counter_evidence(self, evidence_value: float):
        """
        evidence_value should be in [0,1], higher => more counter evidence.
        Use EMA to avoid brittle counts.
        """
        evidence_value = float(np.clip(evidence_value, 0.0, 1.0))
        self.evidence_ema = (self.ema_alpha * evidence_value +
                             (1.0 - self.ema_alpha) * self.evidence_ema)
        if (self.evidence_ema > self.flip_threshold) and (not self.has_flipped):
            self.flip()

    def flip(self):
        # A soft flip: reflect along zero and slightly reduce depth to mark uncertainty.
        self.primitive.data = -1.0 * self.primitive.data
        self.primitive.depth *= 0.6
        self.has_flipped = True

# ---------- Cascade Collapse ----------
class CascadeCollapse:
    def __init__(self, dependency_graph):
        # dependency_graph: dict mapping Primitive -> list of dependents
        self.dependency_graph = dependency_graph

    def trigger(self, root_primitive):
        if root_primitive in self.dependency_graph:
            for dep in self.dependency_graph[root_primitive]:
                dep.coupling_strength *= 0.3
                # propagate
                self.trigger(dep)

# ---------- External Observer (simple inference) ----------
class ExternalObserver:
    def detect_invisible_primitive(self, action_history: np.ndarray, n_components=1):
        """
        action_history: T x D array of observations/actions
        Return: inferred basis vectors (principal components) as candidate primitives
        """
        X = np.asarray(action_history, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Center data
        Xc = X - np.mean(X, axis=0, keepdims=True)
        # SVD for principal components (no sklearn dependency)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        components = Vt[:n_components]
        # wrap as Primitive-like objects
        inferred = [Primitive(name=f"inferred_pc_{i}", data=components[i], depth=0.8)
                    for i in range(components.shape[0])]
        return inferred

    def mirror_back(self, primitive: Primitive):
        """Return a simple universal rule representation of primitive for reporting."""
        return {"name": primitive.name, "norm": float(np.linalg.norm(primitive.data)), "depth": primitive.depth}

# ---------- Quick test / demo ----------
if __name__ == "__main__":
    # primitives
    p1 = GeometricPrimitive("p1", pattern=np.array([1.0, 0.0, 0.0]), scale=1.0)
    p2 = GeometricPrimitive("p2", pattern=np.array([0.9, 0.1, 0.0]), scale=0.8)
    p3 = GeometricPrimitive("p3", pattern=np.array([0.0, 1.0, 0.0]), scale=1.0)

    # resonance engine
    re = ResonanceEngine()
    re.add_pattern([p1, p3])   # stored pattern A
    re.add_pattern([p2])       # stored pattern B

    # input (close to p1)
    inp = [GeometricPrimitive("inp", pattern=np.array([0.98, 0.02, 0.0]), scale=1.0)]
    score, idx = re.evaluate_new_input(inp)
    print("Resonance -> score:", score, "best pattern idx:", idx)

    # threshold flip demo
    base = Primitive("base", data=np.array([1.0, 0.0, 0.0]), depth=1.0)
    tf = ThresholdFlip(base, flip_threshold=0.5, ema_alpha=0.4)
    tf.log_counter_evidence(0.2)
    tf.log_counter_evidence(0.7)
    tf.log_counter_evidence(0.8)
    print("Flip happened:", tf.has_flipped, "primitive now:", base.data, "depth:", base.depth)

    # observer demo
    hist = np.vstack([np.array([1.0, 0.0, 0.0]) + 0.01 * np.random.randn(3),
                      np.array([0.98, 0.02, 0.0]) + 0.01 * np.random.randn(3),
                      np.array([1.01, -0.01, 0.0]) + 0.01 * np.random.randn(3)])
    obs = ExternalObserver()
    inferred = obs.detect_invisible_primitive(hist, n_components=1)
    print("Inferred primitives:", [obs.mirror_back(x) for x in inferred])

# -----------------------------------------
# Section 7: Editable Sections / Next Steps
# -----------------------------------------
# 1. Add more primitives from real datasets
# 2. Adjust resonance scoring to include geometric correlations
# 3. Integrate external witness / observer module
# 4. Experiment with multi-scalar combination strategies for creativity
# 5. Implement nature-grounded external field stabilization

# This notebook is intended to be saved, edited, and re-uploaded for iterative development.
