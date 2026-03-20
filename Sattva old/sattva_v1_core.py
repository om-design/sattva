"""
SATTVA v1.0
Semantic Attractor Training of Transforming Vector Associations

Core components:
- Sensorimotor sandbox
- Invariant primitive extraction (covariance eigenmodes)
- Primitive depth dynamics
- Multi-layer projections
- Attractor well clustering
- Repetition gradient
- Shear-driven curiosity
- Epiphany (well merge)
"""

import numpy as np
from numpy.linalg import eig, norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ============================================================
# I. Sensorimotor Sandbox
# ============================================================

class SensorimotorSandbox:
    """
    Simple physical environment:
    Objects fall under constant acceleration.
    Different restitution coefficients (bounce).
    """

    def __init__(self, g=9.81):
        self.g = g

    def simulate_object(self, restitution, mass, steps=50, dt=0.05):
        """
        Simulate vertical motion.
        restitution: bounce coefficient
        mass: irrelevant to acceleration (invariant test)
        """
        y = 10.0
        v = 0.0
        states = []

        for _ in range(steps):
            v -= self.g * dt
            y += v * dt

            if y <= 0:
                y = 0
                v = -restitution * v

            states.append([y, v, mass, restitution])

        return np.array(states)


# ============================================================
# II. Invariant Primitive Layer (ICL)
# ============================================================

class InvariantLayer:
    """
    Extract stable eigenvectors of sensorimotor covariance.
    """

    def __init__(self):
        self.primitives = []
        self.depths = []

    def extract_primitives(self, data, top_k=3):
        """
        data: NxD matrix of states
        """
        cov = np.cov(data.T)
        vals, vecs = eig(cov)

        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]

        for i in range(top_k):
            p = vecs[:, i]
            self.primitives.append(p / norm(p))
            self.depths.append(1.0)

    def project(self, x):
        P = np.array(self.primitives).T
        return P.T @ x

    def update_depths(self, x, prediction_error,
                      alpha=0.001, beta=0.01):

        coeffs = self.project(x)

        for i, c in enumerate(coeffs):
            resonance = c**2
            contradiction = max(0, prediction_error - 0.01)

            self.depths[i] += alpha * resonance - beta * contradiction
            self.depths[i] = max(self.depths[i], 0.0)


# ============================================================
# III. Predictive Consistency Layer (PCL)
# ============================================================

class PredictiveLayer:
    """
    Simple linear predictor for next state.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.trained = False

    def train(self, X, Y):
        self.model.fit(X, Y)
        self.trained = True

    def predict(self, x):
        if not self.trained:
            return x
        return self.model.predict(x.reshape(1, -1))[0]


# ============================================================
# IV. Attractor Well Field (AWF)
# ============================================================

class AttractorField:
    """
    Clusters coefficient space into wells.
    """

    def __init__(self, n_wells=2):
        self.n_wells = n_wells
        self.kmeans = None
        self.centers = None
        self.depths = None
        self.repetition = None

    def fit(self, coeffs):
        self.kmeans = KMeans(n_clusters=self.n_wells)
        labels = self.kmeans.fit_predict(coeffs)

        self.centers = self.kmeans.cluster_centers_
        self.depths = np.ones(self.n_wells)
        self.repetition = np.zeros(self.n_wells)

        return labels

    def assign(self, c):
        distances = np.linalg.norm(self.centers - c, axis=1)
        probs = np.exp(-distances)
        probs /= probs.sum()
        return probs

    def update_repetition(self, probs):
        self.repetition += probs

    def compute_shear(self, invariant_energy, repetition_energy):
        return abs(invariant_energy - repetition_energy)

    def check_epiphany(self, invariant_layer,
                       merge_threshold=0.95):

        if self.n_wells < 2:
            return False

        c1, c2 = self.centers[0], self.centers[1]
        similarity = np.dot(c1, c2) / (norm(c1)*norm(c2))

        if similarity > merge_threshold:
            new_center = (c1 + c2) / 2
            self.centers = np.array([new_center])
            self.n_wells = 1
            print("Epiphany: wells merged.")
            return True

        return False


# ============================================================
# V. SATTVA Core Engine
# ============================================================

class SattvaCore:

    def __init__(self):
        self.invariant = InvariantLayer()
        self.predictive = PredictiveLayer()
        self.attractor = AttractorField(n_wells=2)

    def developmental_phase(self, sandbox):

        data_all = []

        for restitution in [0.2, 0.8]:
            for mass in [1.0, 5.0]:
                sim = sandbox.simulate_object(restitution, mass)
                data_all.append(sim)

        data_all = np.vstack(data_all)

        # Extract invariant primitives
        self.invariant.extract_primitives(data_all)

        # Train predictive layer
        X = data_all[:-1]
        Y = data_all[1:]
        self.predictive.train(X, Y)

        return data_all

    def form_wells(self, data):

        coeffs = np.array([self.invariant.project(x)
                           for x in data])

        labels = self.attractor.fit(coeffs)
        return coeffs, labels

    def process_input(self, x):

        coeff = self.invariant.project(x)

        invariant_energy = np.sum(coeff**2)

        probs = self.attractor.assign(coeff)
        repetition_energy = np.max(probs)

        shear = self.attractor.compute_shear(
            invariant_energy, repetition_energy)

        pred = self.predictive.predict(x)
        prediction_error = norm(pred - x)

        self.invariant.update_depths(x, prediction_error)
        self.attractor.update_repetition(probs)

        curiosity = -np.sum(probs * np.log(probs + 1e-8)) \
                    + shear

        return {
            "coeff": coeff,
            "invariant_energy": invariant_energy,
            "shear": shear,
            "curiosity": curiosity,
            "prediction_error": prediction_error
        }


# ============================================================
# VI. Minimal Simulation
# ============================================================

if __name__ == "__main__":

    sandbox = SensorimotorSandbox()
    sattva = SattvaCore()

    print("=== Developmental Phase ===")
    data = sattva.developmental_phase(sandbox)

    coeffs, labels = sattva.form_wells(data)

    print("Invariant primitives:", len(sattva.invariant.primitives))

    print("\n=== Processing Inputs ===")

    for i in range(10):
        x = data[np.random.randint(0, len(data))]
        result = sattva.process_input(x)

        print(f"\nInput {i}")
        print("Invariant energy:", result["invariant_energy"])
        print("Shear:", result["shear"])
        print("Curiosity:", result["curiosity"])
        print("Prediction error:", result["prediction_error"])

    sattva.attractor.check_epiphany(sattva.invariant)
