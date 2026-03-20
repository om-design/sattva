from sattva_engine import SattvaEngine
import numpy as np
import json
import os
from datetime import datetime, timezone

def to_json_safe(obj):
    """Recursively convert numpy types to plain Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_json_safe(v) for v in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return to_json_safe(obj.tolist())
    return obj


DIM = 16
np.random.seed(42)

# === ORTHOGONAL BASE PRIMITIVES ===
raw = np.random.randn(4, DIM)
Q, _ = np.linalg.qr(raw.T)
ortho = Q.T[:4]
bases = {
    'point': ortho[0],
    'line': ortho[1],
    'angle': ortho[2],
    'circle': ortho[3],
}

protos = dict(bases)


def make(weights):
    """Compose a prototype from weighted existing prototypes."""
    v = sum(w * protos[n] for n, w in weights.items())
    v += np.random.randn(DIM) * 0.01
    return v / np.linalg.norm(v)


def sample(proto, noise=0.06):
    """Generate a noisy observation from a prototype."""
    v = proto + np.random.randn(DIM) * noise
    return v / np.linalg.norm(v)


# === GEOMETRY (level 2: composites) ===
protos['triangle'] = make({'line': 0.5, 'angle': 0.5})
protos['square'] = make({'line': 0.6, 'angle': 0.4})
protos['arc'] = make({'line': 0.5, 'circle': 0.5})
protos['polygon'] = make({'line': 0.55, 'angle': 0.45})
protos['ray'] = make({'point': 0.5, 'line': 0.5})
protos['tangent'] = make({'line': 0.4, 'circle': 0.4, 'point': 0.2})

# Geometry (level 3: composites of composites)
protos['pentagon'] = make({'polygon': 0.6, 'circle': 0.4})
protos['star'] = make({'triangle': 0.6, 'angle': 0.4})
protos['spiral'] = make({'arc': 0.5, 'circle': 0.5})
protos['sector'] = make({'arc': 0.4, 'angle': 0.3, 'line': 0.3})

# === PHYSICS ===
protos['motion'] = make({'point': 0.5, 'line': 0.5})
protos['velocity'] = make({'line': 0.6, 'point': 0.4})
protos['collision'] = make({'point': 0.4, 'angle': 0.4, 'line': 0.2})
protos['orbit'] = make({'circle': 0.5, 'point': 0.3, 'line': 0.2})
protos['projectile'] = make({'arc': 0.5, 'point': 0.3, 'line': 0.2})
protos['wave'] = make({'circle': 0.4, 'line': 0.3, 'arc': 0.3})
protos['pendulum'] = make({'arc': 0.5, 'point': 0.3, 'angle': 0.2})
protos['rotation'] = make({'circle': 0.5, 'angle': 0.3, 'point': 0.2})

# === MUSIC ===
protos['beat'] = make({'point': 0.7, 'circle': 0.3})
protos['rhythm'] = make({'circle': 0.6, 'point': 0.4})
protos['melody'] = make({'line': 0.5, 'point': 0.3, 'arc': 0.2})
protos['harmony'] = make({'angle': 0.5, 'point': 0.3, 'line': 0.2})
protos['syncopation'] = make({'circle': 0.4, 'angle': 0.4, 'point': 0.2})
protos['crescendo'] = make({'line': 0.6, 'arc': 0.4})
protos['vibrato'] = make({'circle': 0.3, 'line': 0.3, 'arc': 0.4})

# === BIOLOGY ===
protos['cell_division'] = make({'circle': 0.4, 'point': 0.3, 'line': 0.3})
protos['growth'] = make({'line': 0.5, 'arc': 0.3, 'point': 0.2})
protos['heartbeat'] = make({'circle': 0.5, 'point': 0.5})
protos['branching'] = make({'angle': 0.4, 'line': 0.4, 'point': 0.2})
protos['helix'] = make({'spiral': 0.6, 'line': 0.4})
protos['membrane'] = make({'circle': 0.5, 'line': 0.3, 'point': 0.2})

# === DOMAIN GROUPS ===
geo_l1 = ['point', 'line', 'angle', 'circle']
geo_l2 = ['triangle', 'square', 'arc', 'polygon', 'ray', 'tangent']
geo_l3 = ['pentagon', 'star', 'spiral', 'sector']
phys = ['motion', 'velocity', 'collision', 'orbit', 'projectile', 'wave', 'pendulum', 'rotation']
mus = ['beat', 'rhythm', 'melody', 'harmony', 'syncopation', 'crescendo', 'vibrato']
bio = ['cell_division', 'growth', 'heartbeat', 'branching', 'helix', 'membrane']

print(f"Prototypes: {len(protos)} across geometry, physics, music, biology")

# === ENGINE ===
engine = SattvaEngine(dim=DIM)


def run_phase(label, pool, steps):
    """
    pool: list of (names, tag, weight)
    """
    print(f"\n--- {label} ({steps} steps) ---")
    total_w = sum(w for _, _, w in pool)
    thresholds = []
    cum = 0.0
    for names, tag, w in pool:
        cum += w / total_w
        thresholds.append((cum, names, tag))

    branches = 0
    for _ in range(steps):
        r = np.random.rand()
        for thresh, names, tag in thresholds:
            if r <= thresh:
                name = np.random.choice(names)
                break

        # Pass prototype name into engine.observe so pillars can carry names
        result = engine.observe(
            sample(protos[name]),
            tag=tag,
            name=name,
        )

        if result.get('crystallized'):
            c = result['crystallized']
            print(f"  Step {engine.step}: CRYSTAL prim #{c['prim_id']} ({c['evidence']} evidence)")

        if result.get('epiphany'):
            e = result['epiphany']
            print(f"  Step {engine.step}: EPIPHANY {e['tags']} "
                  f"mag={e['magnitude']:.1f} shared={e['shared_prims']}")

        if result.get('action') == 'branch_pillar':
            branches += 1

    s = engine.status()
    print(f"  => prims={s['primitives']} pillars={s['pillars']} "
          f"epi={s['epiphanies']} branches={branches} myel={s['prim_myel']}")


# === TRAINING PHASES ===

# 0. Pure bases
run_phase("Pure Bases",
          [(geo_l1, 'geometry', 1.0)], steps=150)

# 1. Geometry
run_phase("Geometry Composites",
          [(geo_l1, 'geometry', 0.3),
           (geo_l2, 'geometry', 0.4),
           (geo_l3, 'geometry', 0.3)], steps=300)

# 2. Physics
run_phase("Physics",
          [(phys, 'physics', 0.6),
           (geo_l1 + geo_l2, 'geometry', 0.4)], steps=300)

# 3. Music
run_phase("Music",
          [(mus, 'music', 0.5),
           (geo_l1, 'geometry', 0.25),
           (phys, 'physics', 0.25)], steps=300)

# 4. Biology
run_phase("Biology",
          [(bio, 'biology', 0.5),
           (geo_l1, 'geometry', 0.15),
           (phys, 'physics', 0.15),
           (mus, 'music', 0.2)], steps=300)

# Gentle pruning after Biology
engine.prune_pillars(min_myel=0.02, min_total=2, max_per_tag=None)
engine.prune_primitives(min_myel=0.01, require_pillars=True)

# 5. Free Play
run_phase("Free Play (all domains)",
          [(geo_l1 + geo_l2 + geo_l3, 'geometry', 0.2),
           (phys, 'physics', 0.25),
           (mus, 'music', 0.25),
           (bio, 'biology', 0.3)], steps=350)

# Gentle pruning after Free Play
engine.prune_pillars(min_myel=0.03, min_total=3, max_per_tag=None)
engine.prune_primitives(min_myel=0.02, require_pillars=True)

# === FINAL REPORT + JSON LOGGING ===
status = engine.status()
print(f"\n{'=' * 60}")
print(f"FINAL: {status['step']} steps | {status['primitives']} primitives | "
      f"{status['pillars']} pillars | {status['epiphanies']} epiphanies")
print(f"Myelination: {status['prim_myel']}")

# Primitive summary
primitive_summary = []
for i, pr in enumerate(engine.primitives):
    # Best matching base shape
    best_name, best_sim = max(
        ((n, float(abs(np.dot(pr['v'], p)))) for n, p in bases.items()),
        key=lambda x: x[1]
    )
    primitive_summary.append({
        'id': i,
        'best_base': best_name,
        'best_sim': round(best_sim, 3),
        'myel': pr['myel'],
        'n_pillars': len(pr['pillar_ids']),
    })

# Pillars by domain and primitives used per domain
pillar_counts = {}
domain_prims = {}
for p in engine.pillars:
    tag = p['tag']
    pillar_counts[tag] = pillar_counts.get(tag, 0) + 1
    if tag is not None:
        domain_prims.setdefault(tag, set()).update(p['prim_ids'])

domain_prims = {tag: sorted(list(ids)) for tag, ids in domain_prims.items()}

print("\nPillars by domain:")
for t, c in sorted(pillar_counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")

print("\nPrimitive sharing:")
for d, ids in sorted(domain_prims.items()):
    print(f"  {d}: prims {ids}")

print("\nEpiphany pairs:")
ep_pairs = {}
for e in engine.epiphanies:
    key = tuple(sorted(e['tags']))
    ep_pairs.setdefault(key, []).append(e)
for pair, eps in sorted(ep_pairs.items()):
    print(f"  {pair[0]} <-> {pair[1]}: {len(eps)} events")

# === JSON log: epiphanies + structure + pillar metadata ===

# Simple run identifier and timestamp
now = datetime.now(timezone.utc)
run_id = now.isoformat()

# Stub pillar metadata for now; to be enriched later with labels/examples
pillar_metadata = {
    str(i): {
        "auto_label": None,
        "user_label": None,
        "examples": [],
    }
    for i, _ in enumerate(engine.pillars)
}

log = {
    'run_id': run_id,
    'run_started_at': run_id,
    'status': status,
    'epiphanies': engine.epiphanies,
    'primitives': primitive_summary,
    'pillar_counts': pillar_counts,
    'domain_primitives': domain_prims,
    'pillar_metadata': pillar_metadata,
}

if os.path.exists("training_log.json"):
    with open("training_log.json", "r") as f:
        old = json.load(f)
    # Append epiphanies; latest snapshot overwrites structure/metadata
    log['epiphanies'] = old.get('epiphanies', []) + log['epiphanies']

with open("training_log.json", "w") as f:
    json.dump(to_json_safe(log), f, indent=2)

print("\nWrote training_log.json")
