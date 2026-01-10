# Quick Start: From Baby Dragons to Production

**Goal:** Get from conceptual understanding to working code FAST

---

## The 3-Path Approach

Depending on your priorities, choose your path:

### Path A: Fast Prototype (Python-First)
**Best for:** Rapid experimentation, proving concepts
**Timeline:** 2-4 weeks to working demo
**Tech:** Pure Python + existing libraries

### Path B: Production-Ready (Rust-First)  
**Best for:** Building real product, performance matters
**Timeline:** 2-3 months to production
**Tech:** Rust core + Python bindings

### Path C: Hybrid (Recommended)
**Best for:** Balance speed and quality
**Timeline:** 1-2 months to production-capable
**Tech:** Python for logic, Rust for hotspots

---

## PATH A: Fast Prototype (Start Today)

### Step 1: Leverage Existing Tools (Day 1)

```bash
# Install dependencies
pip install faiss-cpu numpy torch transformers

# If you have GPU
pip install faiss-gpu
```

### Step 2: Substrate with FAISS (Day 1-2)

```python
# substrate/core.py

import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Attractor:
    id: int
    center: np.ndarray
    strength: float = 0.1
    count: int = 0

class FastSubstrate:
    """Production-capable substrate using FAISS"""
    
    def __init__(self, dimensions=512, use_gpu=False):
        self.d = dimensions
        self.attractors = []
        self.next_id = 0
        
        # FAISS index for fast similarity search
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatL2(res, dimensions)
        else:
            self.index = faiss.IndexFlatL2(dimensions)
        
        # Store vectors for attractor updates
        self.vectors = []
        
    def activate(self, pattern: np.ndarray, k=10) -> List[int]:
        """Find k nearest attractors"""
        if len(self.vectors) == 0:
            return []
        
        pattern = pattern.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(pattern, k)
        
        return indices[0].tolist()
    
    def learn(self, experience: np.ndarray, threshold=0.7) -> int:
        """Create or strengthen attractor"""
        experience = experience.astype('float32')
        
        if len(self.vectors) == 0:
            # First attractor
            return self._create_attractor(experience)
        
        # Find nearest
        nearest_ids = self.activate(experience, k=1)
        nearest_id = nearest_ids[0]
        
        # Check similarity
        similarity = self._compute_similarity(
            experience, 
            self.vectors[nearest_id]
        )
        
        if similarity > threshold:
            # Strengthen existing
            self._strengthen_attractor(nearest_id, experience)
            return nearest_id
        else:
            # Create new
            return self._create_attractor(experience)
    
    def _create_attractor(self, vector: np.ndarray) -> int:
        attractor = Attractor(
            id=self.next_id,
            center=vector.copy(),
            strength=0.1,
            count=1
        )
        self.attractors.append(attractor)
        self.vectors.append(vector)
        
        # Add to FAISS index
        self.index.add(vector.reshape(1, -1))
        
        self.next_id += 1
        return attractor.id
    
    def _strengthen_attractor(self, attr_id: int, vector: np.ndarray):
        attractor = self.attractors[attr_id]
        attractor.count += 1
        
        # Move center toward new vector
        lr = 0.1 / np.sqrt(attractor.count)
        attractor.center += lr * (vector - attractor.center)
        
        # Increase strength
        attractor.strength = min(1.0, attractor.strength + 0.05)
        
        # Update index (rebuild for now, optimize later)
        self._rebuild_index()
    
    def _rebuild_index(self):
        self.index.reset()
        all_vectors = np.array([a.center for a in self.attractors])
        self.index.add(all_vectors.astype('float32'))
    
    def _compute_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

### Step 3: Run Experiment 02 on Fast Substrate (Day 2-3)

```python
# test_fast_substrate.py

from substrate.core import FastSubstrate
import numpy as np

# Initialize
substrate = FastSubstrate(dimensions=20)

# Simulate physical experiences
for trial in range(100):
    # Random experience (simulate dropping object)
    elasticity = np.random.uniform(0, 1)
    height = np.random.uniform(0.5, 2.0)
    
    bounce_ratio = elasticity * 0.8  # Simplified physics
    
    experience = np.array([
        height,
        bounce_ratio,
        elasticity,
        np.random.randn() * 0.1,  # Noise
    ] + [0] * 16)  # Pad to 20 dimensions
    
    attractor_id = substrate.learn(experience)
    
    if trial % 10 == 0:
        print(f"Trial {trial}: {len(substrate.attractors)} attractors")

print(f"\nFinal: {len(substrate.attractors)} attractors formed")
print(f"Avg strength: {np.mean([a.strength for a in substrate.attractors]):.3f}")
```

### Step 4: Add Primitive Library (Day 3-4)

```python
# primitives/library.py

class PrimitiveLibrary:
    def __init__(self, substrate: FastSubstrate):
        self.substrate = substrate
        self.labels = {}  # attractor_id -> label
        self.concepts = {}  # concept -> List[attractor_id]
    
    def label_attractor(self, attr_id: int, label: str, 
                       concept: Optional[str] = None):
        self.labels[attr_id] = label
        
        if concept:
            if concept not in self.concepts:
                self.concepts[concept] = []
            self.concepts[concept].append(attr_id)
    
    def find_by_concept(self, concept: str) -> List[Attractor]:
        attr_ids = self.concepts.get(concept, [])
        return [self.substrate.attractors[i] for i in attr_ids]
```

### Step 5: Simple Language Grounding (Day 4-5)

```python
# language/simple_grounding.py

class SimpleGrounding:
    def __init__(self, substrate: FastSubstrate, library: PrimitiveLibrary):
        self.substrate = substrate
        self.library = library
        self.word_to_attrs = {}  # word -> List[attractor_id]
    
    def ground_word(self, word: str, attractor_ids: List[int]):
        self.word_to_attrs[word] = attractor_ids
    
    def understand(self, text: str) -> Dict:
        words = text.lower().split()
        
        activated_attractors = []
        for word in words:
            attr_ids = self.word_to_attrs.get(word, [])
            activated_attractors.extend(attr_ids)
        
        return {
            'words': words,
            'attractors': activated_attractors,
            'labels': [self.library.labels.get(a, 'unknown') 
                      for a in activated_attractors]
        }

# Example usage
substrate = FastSubstrate()
library = PrimitiveLibrary(substrate)
grounding = SimpleGrounding(substrate, library)

# Ground some words
grounding.ground_word('ball', [0, 1])  # Elastic, round
grounding.ground_word('bounce', [2])    # Elastic collision

understanding = grounding.understand('the ball bounced')
print(understanding)
```

**Timeline:** 5 days to working demo  
**Result:** Can run experiments, test ideas, demo to others

---

## PATH B: Production-Ready (Rust Core)

### Step 1: Set Up Rust Project (Day 1)

```bash
# Create new Rust project
cargo new substrate-core --lib
cd substrate-core

# Add dependencies to Cargo.toml
```

```toml
# Cargo.toml

[package]
name = "substrate-core"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.15"
blas-src = { version = "0.8", features = ["openblas"] }
faiss = "0.11"  # Rust bindings to FAISS
rayon = "1.7"   # Parallelism
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
criterion = "0.5"  # Benchmarking

[[bench]]
name = "substrate_bench"
harness = false
```

### Step 2: Core Substrate (Week 1)

```rust
// src/lib.rs

use ndarray::{Array1, Array2};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Attractor {
    pub id: usize,
    pub center: Array1<f32>,
    pub strength: f32,
    pub activation_count: u64,
}

pub struct Substrate {
    dimension: usize,
    attractors: Vec<Attractor>,
    next_id: usize,
}

impl Substrate {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            attractors: Vec::new(),
            next_id: 0,
        }
    }
    
    pub fn learn(&mut self, experience: &Array1<f32>, threshold: f32) -> usize {
        if self.attractors.is_empty() {
            return self.create_attractor(experience);
        }
        
        let (nearest_id, similarity) = self.find_nearest(experience);
        
        if similarity > threshold {
            self.strengthen_attractor(nearest_id, experience);
            nearest_id
        } else {
            self.create_attractor(experience)
        }
    }
    
    fn create_attractor(&mut self, center: &Array1<f32>) -> usize {
        let attractor = Attractor {
            id: self.next_id,
            center: center.clone(),
            strength: 0.1,
            activation_count: 1,
        };
        
        self.attractors.push(attractor);
        self.next_id += 1;
        self.next_id - 1
    }
    
    fn strengthen_attractor(&mut self, id: usize, vector: &Array1<f32>) {
        let attractor = &mut self.attractors[id];
        attractor.activation_count += 1;
        
        // Move center
        let lr = 0.1 / (attractor.activation_count as f32).sqrt();
        attractor.center = &attractor.center + &(lr * (vector - &attractor.center));
        
        // Increase strength
        attractor.strength = (attractor.strength + 0.05).min(1.0);
    }
    
    fn find_nearest(&self, vector: &Array1<f32>) -> (usize, f32) {
        self.attractors
            .iter()
            .map(|a| (a.id, cosine_similarity(vector, &a.center)))
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap())
            .unwrap()
    }
}

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attractor_formation() {
        let mut substrate = Substrate::new(10);
        
        let exp1 = Array1::from_vec(vec![1.0; 10]);
        let id1 = substrate.learn(&exp1, 0.7);
        assert_eq!(id1, 0);
        assert_eq!(substrate.attractors.len(), 1);
        
        let exp2 = Array1::from_vec(vec![1.1; 10]);
        let id2 = substrate.learn(&exp2, 0.7);
        assert_eq!(id2, 0);  // Should strengthen, not create new
        assert_eq!(substrate.attractors.len(), 1);
        assert_eq!(substrate.attractors[0].activation_count, 2);
    }
}
```

### Step 3: Python Bindings (Week 2)

```rust
// python-bindings/src/lib.rs

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use substrate_core::Substrate as RustSubstrate;

#[pyclass]
struct Substrate {
    inner: RustSubstrate,
}

#[pymethods]
impl Substrate {
    #[new]
    fn new(dimension: usize) -> Self {
        Self {
            inner: RustSubstrate::new(dimension),
        }
    }
    
    fn learn(&mut self, experience: PyReadonlyArray1<f32>, threshold: f32) -> usize {
        let array = experience.as_array().to_owned();
        self.inner.learn(&array, threshold)
    }
    
    fn num_attractors(&self) -> usize {
        self.inner.attractors().len()
    }
}

#[pymodule]
fn substrate_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Substrate>()?;
    Ok(())
}
```

**Timeline:** 2-3 months to production  
**Result:** High-performance, scalable system

---

## PATH C: Hybrid (Best of Both)

### Week 1: Fast Prototype
Start with Path A - get working quickly

### Week 2-4: Identify Hotspots
Profile Python code, find bottlenecks:
```python
import cProfile
cProfile.run('your_code()')
```

Typically bottlenecks are:
- Similarity search (FAISS helps)
- Attractor updates (vectorize with NumPy)
- Resonance propagation (can parallelize)

### Week 5-8: Rewrite Hotspots in Rust
Only rewrite performance-critical parts:
- Substrate core (similarity search, attractor updates)
- Keep Python for: primitives, language, API

**Timeline:** 1-2 months to production  
**Result:** Fast enough, flexible enough

---

## Immediate Next Steps (Today)

### Option 1: Jump Into Code
```bash
# Path A (fast)
git clone [your-repo]
cd sattva
pip install -r requirements.txt
python experiments/02_primitive_formation_attractors.py

# Modify experiments/02 to use FastSubstrate instead of simulation
```

### Option 2: Plan First
```bash
# Set up project structure
mkdir -p substrate/{core,primitives,language}
mkdir -p tests benchmarks

# Create initial files
touch substrate/core.py
touch substrate/primitives.py  
touch substrate/language.py

# Start with core
vim substrate/core.py  # Implement FastSubstrate
```

### Option 3: Rust from Start
```bash
cargo new substrate-core --lib
cd substrate-core
# Follow Path B above
```

---

## What to Build First

**Priority 1 (Week 1):**
- [x] FastSubstrate with FAISS
- [x] Attractor formation working
- [x] Run Experiment 02 successfully

**Priority 2 (Week 2-3):**
- [ ] Primitive library
- [ ] Basic regulation (homeostatic)
- [ ] Persistence (save/load attractors)

**Priority 3 (Week 4-6):**
- [ ] Physical simulation (MuJoCo)
- [ ] Experience encoding
- [ ] Training pipeline

**Priority 4 (Week 7-8):**
- [ ] Simple language grounding
- [ ] Context handling
- [ ] Basic API

---

## Success Milestones

### Week 1: ✅ Core Working
- FastSubstrate implemented
- Experiment 02 runs
- 10+ attractors form from 100 experiences

### Week 2: ✅ Primitives
- Can label attractors
- Can organize by concept
- Can query by concept

### Week 4: ✅ Physical Grounding
- MuJoCo simulation running
- Agent learns from dropping objects
- 80% recognition on novel cases

### Week 8: ✅ Language
- Words ground to attractors
- Simple sentences understood
- Context disambiguates

### Week 12: ✅ Production
- API deployed
- 100+ req/sec
- Monitoring in place

---

## Choose Your Path

**Need demo quickly?** → Path A (Python-first)  
**Building real product?** → Path B (Rust-first)  
**Want balance?** → Path C (Hybrid)

**Recommendation:** Start with Path A this week. If it works and you need scale, migrate to Path C.

**The key: START NOW. Code beats planning.**
