from sattva_engine import SattvaEngine
import numpy as np

DIM = 16
np.random.seed(42)

# Create orthogonal base primitives
raw = np.random.randn(4, DIM)
Q, _ = np.linalg.qr(raw.T)
ortho = Q.T[:4]

bases = {'point': ortho[0], 'line': ortho[1], 'angle': ortho[2], 'circle': ortho[3]}

def make(weights, protos):
    v = sum(w * protos[n] for n, w in weights.items())
    v += np.random.randn(DIM) * 0.01
    return v / np.linalg.norm(v)

def sample(proto, noise=0.06):
    v = proto + np.random.randn(DIM) * noise
    return v / np.linalg.norm(v)

# Build prototype library
protos = dict(bases)
protos['triangle'] = make({'line': 0.5, 'angle': 0.5}, protos)
protos['arc'] = make({'line': 0.5, 'circle': 0.5}, protos)
protos['motion'] = make({'point': 0.5, 'line': 0.5}, protos)
protos['orbit'] = make({'circle': 0.5, 'point': 0.3, 'line': 0.2}, protos)
protos['beat'] = make({'point': 0.7, 'circle': 0.3}, protos)
protos['melody'] = make({'line': 0.5, 'point': 0.3, 'arc': 0.2}, protos)

engine = SattvaEngine(dim=DIM)

# Phase 1: Pure geometry
print("Phase 1: Geometry")
for i in range(200):
    name = np.random.choice(['point', 'line', 'angle', 'circle'])
    r = engine.observe(sample(protos[name]), tag='geometry')
    if r.get('crystallized'):
        print(f"  Step {engine.step}: crystallized prim #{r['crystallized']['prim_id']}")

# Phase 2: Composites + physics
print("\nPhase 2: Physics")
for i in range(200):
    if np.random.rand() < 0.6:
        name = np.random.choice(['motion', 'orbit']); tag = 'physics'
    else:
        name = np.random.choice(['triangle', 'arc']); tag = 'geometry'
    r = engine.observe(sample(protos[name]), tag=tag)
    if r.get('epiphany'):
        e = r['epiphany']
        print(f"  Step {engine.step}: EPIPHANY {e['tags']} mag={e['magnitude']:.1f}")
    if r.get('action') == 'branch_pillar':
        print(f"  Step {engine.step}: BRANCH -> pillar #{r['pillar']} ({tag})")

# Phase 3: Music
print("\nPhase 3: Music")
for i in range(200):
    if np.random.rand() < 0.5:
        name = np.random.choice(['beat', 'melody']); tag = 'music'
    else:
        name = np.random.choice(['point', 'line', 'circle']); tag = 'geometry'
    r = engine.observe(sample(protos[name]), tag=tag)
    if r.get('epiphany'):
        e = r['epiphany']
        print(f"  Step {engine.step}: EPIPHANY {e['tags']} mag={e['magnitude']:.1f}")
    if r.get('action') == 'branch_pillar':
        print(f"  Step {engine.step}: BRANCH -> pillar #{r['pillar']} ({tag})")

# Save all epiphanies
with open("epiphanies.json", "w") as f:
    json.dump(engine.epiphanies, f, indent=2)

print(f"\nFinal: {engine.status()}")
