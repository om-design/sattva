import numpy as np
import uuid


# -----------------------------
# Utility
# -----------------------------

def l2_distance_sq(a, b):
    return np.sum((a - b) ** 2)


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)


# -----------------------------
# Primitive
# -----------------------------

class Primitive:
    def __init__(self, embedding, complexity=1.0):
        self.id = str(uuid.uuid4())
        self.embedding = embedding.astype(float)

        # Energy dynamics
        self.energy = 0.0
        self.decay_rate = 0.001 / complexity

        # Bandwidth (myelination analogue)
        self.bandwidth = 1.0
        self.bandwidth_cap = 10.0
        self.bandwidth_decay = 0.0005

        # Structural relations
        self.components = []
        self.parents = []

    def inject_energy(self, amount):
        self.energy += amount

    def decay(self):
        self.energy -= self.decay_rate * self.energy
        self.bandwidth -= self.bandwidth_decay * self.bandwidth
        if self.bandwidth < 1.0:
            self.bandwidth = 1.0

    def reinforce_bandwidth(self, amount):
        self.bandwidth += amount
        if self.bandwidth > self.bandwidth_cap:
            self.bandwidth = self.bandwidth_cap


# -----------------------------
# Engine
# -----------------------------

class Engine:
    def __init__(self, dim=8):
        self.dim = dim
        self.primitives = {}

        # Coherence / crystallization parameters
        self.coherence_threshold = 0.8
        self.crystallization_margin = 0.05
        self.temperature = 1.0

        # Periodic consolidation
        self.consolidation_interval = 50
        self.step_count = 0

    # ---------------------------------
    # Creation
    # ---------------------------------

    def create_primitive(self, embedding, complexity=1.0):
        p = Primitive(embedding, complexity)
        self.primitives[p.id] = p
        return p.id

    # ---------------------------------
    # Activation
    # ---------------------------------

    def activate_input(self, input_vector, magnitude=1.0):
        """
        Activate all primitives via resonance.
        """
        for p in self.primitives.values():
            r = cosine_similarity(p.embedding, input_vector)
            if r > 0:
                p.inject_energy(magnitude * r)
                p.reinforce_bandwidth(0.1 * r)

    # ---------------------------------
    # Routing
    # ---------------------------------

    def routing_cost(self, pid_from, pid_to):
        p1 = self.primitives[pid_from]
        p2 = self.primitives[pid_to]
        d2 = l2_distance_sq(p1.embedding, p2.embedding)
        return (1.0 / p2.bandwidth) + d2

    def transition_probability(self, pid_from, pid_to):
        d2 = l2_distance_sq(
            self.primitives[pid_from].embedding,
            self.primitives[pid_to].embedding
        )
        return np.exp(-d2 / self.temperature)

    # ---------------------------------
    # Spectral Coherence Detection
    # ---------------------------------

    def compute_coherence_matrix(self):
        ids = list(self.primitives.keys())
        n = len(ids)
        M = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pi = self.primitives[ids[i]]
                pj = self.primitives[ids[j]]

                sim = cosine_similarity(pi.embedding, pj.embedding)
                energy_factor = (pi.energy + pj.energy) / 2
                M[i, j] = sim * energy_factor

        return M, ids

    # ---------------------------------
    # Multi-Node Crystallization
    # ---------------------------------

    def attempt_crystallization(self):
        if len(self.primitives) < 2:
            return

        M, ids = self.compute_coherence_matrix()

        # Spectral decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(M)

        dominant_index = np.argmax(eigenvalues)
        dominant_value = eigenvalues[dominant_index]

        if dominant_value < self.coherence_threshold:
            return

        # Nodes participating strongly in dominant eigenmode
        dominant_vector = eigenvectors[:, dominant_index]

        participating = [
            ids[i]
            for i, val in enumerate(dominant_vector)
            if abs(val) > 0.3
        ]

        if len(participating) < 2:
            return

        # Compute centroid
        centroid = np.mean(
            [self.primitives[pid].embedding for pid in participating],
            axis=0
        )

        # Compare routing efficiency
        cost_before = 0.0
        for i in participating:
            for j in participating:
                if i != j:
                    cost_before += self.routing_cost(i, j)

        new_id = self.create_primitive(centroid, complexity=len(participating))

        cost_after = 0.0
        for pid in participating:
            cost_after += self.routing_cost(pid, new_id)

        if cost_after + self.crystallization_margin < cost_before:
            new_p = self.primitives[new_id]
            new_p.components = participating

            for pid in participating:
                self.primitives[pid].parents.append(new_id)
        else:
            del self.primitives[new_id]

    # ---------------------------------
    # Decomposition
    # ---------------------------------

    def attempt_decomposition(self):
        to_remove = []

        for pid, p in self.primitives.items():
            if not p.components:
                continue

            composite_cost = 0.0
            for c in p.components:
                composite_cost += self.routing_cost(pid, c)

            direct_cost = 0.0
            for i in range(len(p.components)):
                for j in range(len(p.components)):
                    if i != j:
                        direct_cost += self.routing_cost(
                            p.components[i],
                            p.components[j]
                        )

            if direct_cost < composite_cost:
                to_remove.append(pid)

        for pid in to_remove:
            p = self.primitives[pid]
            for c in p.components:
                if pid in self.primitives[c].parents:
                    self.primitives[c].parents.remove(pid)
            del self.primitives[pid]

    # ---------------------------------
    # Periodic Eigenmode Consolidation
    # ---------------------------------

    def consolidate(self):
        if len(self.primitives) < 2:
            return

        embeddings = np.array(
            [p.embedding for p in self.primitives.values()]
        )

        # Global covariance
        cov = np.cov(embeddings.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Align primitives slightly toward dominant global mode
        dominant_vec = eigenvectors[:, np.argmax(eigenvalues)]

        for p in self.primitives.values():
            alignment = cosine_similarity(p.embedding, dominant_vec)
            p.embedding += 0.01 * alignment * dominant_vec

    # ---------------------------------
    # Global Step
    # ---------------------------------

    def step(self):
        self.step_count += 1

        # Continuous decay
        for p in self.primitives.values():
            p.decay()

        # Structural evolution
        self.attempt_crystallization()
        self.attempt_decomposition()

        # Periodic global consolidation
        if self.step_count % self.consolidation_interval == 0:
            self.consolidate()
