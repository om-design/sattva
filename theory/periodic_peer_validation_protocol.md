# Periodic Peer Validation Protocol

**Date:** January 9, 2026  
**Context:** Multi-agent consensus, batch learning, developmental gates  
**Purpose:** Formalize periodic check-ins for primitive validation

---

## Core Principle

**Learning happens in BATCHES, with peer consensus between batches.**

Not continuous validation (too slow)
Not no validation (too risky)

**Sweet spot:** Periodic check-ins after meaningful learning units

---

## Batch Structure

### What is a Batch?

```python
class LearningBatch:
    """A unit of learning requiring validation."""
    
    def __init__(self, batch_id, stage):
        self.batch_id = batch_id
        self.stage = stage  # 'physical', 'language', 'reasoning'
        self.experiences = []
        self.new_primitives = []
        self.modified_primitives = []
        self.start_time = time.now()
        self.end_time = None
    
    def is_complete(self):
        """When is batch ready for validation?"""
        return (
            len(self.experiences) >= self.min_experiences() or
            len(self.new_primitives) >= self.min_new_primitives() or
            time.now() - self.start_time > self.max_duration()
        )
    
    def min_experiences(self):
        """Experiences needed for batch."""
        return {
            'physical': 100,     # Physical foundation: smaller batches
            'language': 200,     # Language: medium batches
            'reasoning': 500,    # Reasoning: larger batches
        }[self.stage]
    
    def min_new_primitives(self):
        """Or new primitives formed."""
        return {
            'physical': 5,       # Physical: frequent validation
            'language': 10,      # Language: less frequent
            'reasoning': 20,     # Reasoning: least frequent
        }[self.stage]
    
    def max_duration(self):
        """Or time elapsed (seconds)."""
        return {
            'physical': 3600,    # Physical: validate every hour
            'language': 7200,    # Language: every 2 hours
            'reasoning': 14400,  # Reasoning: every 4 hours
        }[self.stage]
```

### Why Batches?

**Too frequent validation:**
- Slows learning
- Peer communication overhead
- Can't form coherent patterns

**Too infrequent validation:**
- Bad primitives consolidate
- Errors compound
- Harder to correct later

**Batches balance:**
- Enough experiences to form patterns
- Small enough to catch errors early
- Stage-appropriate frequency

---

## Validation Protocol

### Step 1: Batch Completion

```python
class BatchValidator:
    def __init__(self, peer_network, substrate):
        self.peer_network = peer_network
        self.substrate = substrate
        self.current_batch = None
        self.history = []
    
    def on_experience(self, experience):
        """Called after each learning experience."""
        
        # Add to current batch
        if self.current_batch is None:
            self.current_batch = LearningBatch(
                batch_id=len(self.history),
                stage=self.substrate.current_stage
            )
        
        self.current_batch.experiences.append(experience)
        
        # Track primitive changes
        if experience.created_primitive:
            self.current_batch.new_primitives.append(experience.primitive_id)
        if experience.modified_primitive:
            self.current_batch.modified_primitives.append(experience.primitive_id)
        
        # Batch complete?
        if self.current_batch.is_complete():
            self.validate_batch()
```

### Step 2: Peer Evaluation

```python
    def validate_batch(self):
        """Send batch to peer network for validation."""
        
        batch = self.current_batch
        print(f"\nValidating Batch {batch.batch_id}:")
        print(f"  Stage: {batch.stage}")
        print(f"  Experiences: {len(batch.experiences)}")
        print(f"  New primitives: {len(batch.new_primitives)}")
        print(f"  Modified primitives: {len(batch.modified_primitives)}")
        
        # Collect primitives for validation
        primitives_to_validate = [
            self.substrate.get_primitive(pid)
            for pid in (batch.new_primitives + batch.modified_primitives)
        ]
        
        # Peer evaluation
        results = self.peer_network.evaluate_batch(
            primitives=primitives_to_validate,
            experiences=batch.experiences,
            stage=batch.stage
        )
        
        # Process results
        self.process_validation_results(results, batch)
        
        # Archive batch
        self.history.append(batch)
        self.current_batch = None
```

### Step 3: Consensus Analysis

```python
class PeerConsensus:
    """Analyze peer validation results."""
    
    def __init__(self, n_peers=5):
        self.n_peers = n_peers
        self.consensus_threshold = 0.8  # 80% agreement
    
    def evaluate_primitive(self, primitive, peer_evaluations):
        """Determine consensus on primitive validity."""
        
        # Each peer provides:
        # - matches: how many of their experiences match
        # - confidence: how confident in the primitive
        # - concerns: any issues detected
        
        votes = [eval.confidence for eval in peer_evaluations]
        mean_confidence = np.mean(votes)
        vote_std = np.std(votes)
        
        # BIAS-style agreement
        agreement = 1.0 / (1.0 + vote_std)
        
        # Combined consensus
        consensus = mean_confidence * agreement
        
        # Decision
        if consensus >= self.consensus_threshold:
            return {
                'status': 'VALIDATED',
                'consensus': consensus,
                'action': 'keep'
            }
        elif consensus >= 0.5:
            return {
                'status': 'UNCERTAIN',
                'consensus': consensus,
                'action': 'investigate',
                'focus_areas': self.identify_disagreements(peer_evaluations)
            }
        else:
            return {
                'status': 'REJECTED',
                'consensus': consensus,
                'action': 'remove',
                'reasons': self.collect_concerns(peer_evaluations)
            }
    
    def identify_disagreements(self, evaluations):
        """Which aspects do peers disagree on?"""
        disagreements = []
        
        # Compare peer assessments
        for i, eval_i in enumerate(evaluations):
            for j, eval_j in enumerate(evaluations[i+1:]):
                if abs(eval_i.confidence - eval_j.confidence) > 0.3:
                    # Significant disagreement
                    disagreements.append({
                        'peer_i': i,
                        'peer_j': j,
                        'aspect': self.find_disagreement_aspect(eval_i, eval_j)
                    })
        
        return disagreements
    
    def collect_concerns(self, evaluations):
        """Why are peers rejecting this primitive?"""
        concerns = []
        for eval in evaluations:
            if eval.concerns:
                concerns.extend(eval.concerns)
        
        # Group by concern type
        concern_counts = {}
        for concern in concerns:
            concern_type = concern['type']
            concern_counts[concern_type] = concern_counts.get(concern_type, 0) + 1
        
        # Return most common concerns
        return sorted(
            concern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
```

### Step 4: Action Based on Consensus

```python
    def process_validation_results(self, results, batch):
        """Take action based on peer consensus."""
        
        for primitive_id, validation in results.items():
            primitive = self.substrate.get_primitive(primitive_id)
            
            if validation['status'] == 'VALIDATED':
                # Good primitive - strengthen
                print(f"  ✅ Primitive {primitive_id}: VALIDATED ({validation['consensus']:.2f})")
                primitive.mark_validated(validation['consensus'])
                primitive.lock_for_stages = 1  # Stable for a while
            
            elif validation['status'] == 'UNCERTAIN':
                # Need more experiences
                print(f"  ⚠️  Primitive {primitive_id}: UNCERTAIN ({validation['consensus']:.2f})")
                primitive.mark_uncertain()
                
                # Request focused learning
                focus_areas = validation['focus_areas']
                self.request_focused_experiences(primitive, focus_areas)
            
            elif validation['status'] == 'REJECTED':
                # Bad primitive - remove or correct
                print(f"  ❌ Primitive {primitive_id}: REJECTED ({validation['consensus']:.2f})")
                
                reasons = validation['reasons']
                print(f"     Reasons: {reasons}")
                
                # Check for string collapse
                collapse_risk = self.check_collapse_cascade(primitive)
                
                if collapse_risk['should_collapse']:
                    print(f"     → Triggering controlled collapse (cascade: {len(collapse_risk['affected'])} primitives)")
                    self.trigger_string_collapse(primitive, collapse_risk['affected'])
                else:
                    print(f"     → Marking for correction (gradual replacement)")
                    primitive.mark_for_correction(reasons)
```

---

## Peer Network Structure

### Agent Types

```python
class PeerAgent:
    """Base class for validation agents."""
    
    def __init__(self, agent_id, specialization=None):
        self.agent_id = agent_id
        self.specialization = specialization
        self.substrate = LongRangeSubstrate(n_units=1000)
        self.experience_history = []
    
    def evaluate_primitive(self, primitive, test_experiences=None):
        """Test primitive against agent's experiences."""
        
        if test_experiences is None:
            test_experiences = self.experience_history
        
        matches = 0
        total = 0
        
        for experience in test_experiences:
            # Does primitive predict this experience?
            prediction = primitive.predict(experience.context)
            actual = experience.outcome
            
            if prediction == actual:
                matches += 1
            total += 1
        
        # Confidence = match rate
        confidence = matches / total if total > 0 else 0.0
        
        # Concerns?
        concerns = self.identify_concerns(primitive, test_experiences)
        
        return PeerEvaluation(
            agent_id=self.agent_id,
            primitive_id=primitive.id,
            confidence=confidence,
            matches=matches,
            total=total,
            concerns=concerns
        )

class PhysicsSpecialistAgent(PeerAgent):
    """Specializes in physical world validation."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, specialization='physics')
        # More physical experiences
        self.load_physics_experiences()

class LanguageSpecialistAgent(PeerAgent):
    """Specializes in language grounding."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, specialization='language')
        # More language experiences
        self.load_language_experiences()

class GeneralistAgent(PeerAgent):
    """No specialization - diverse experiences."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, specialization=None)
        self.load_diverse_experiences()
```

### Network Configuration

```python
class ValidationNetwork:
    """Network of peer validation agents."""
    
    def __init__(self, stage):
        self.stage = stage
        self.agents = self.configure_agents_for_stage(stage)
    
    def configure_agents_for_stage(self, stage):
        """Different peer mix for each stage."""
        
        if stage == 'physical':
            # Physics foundation: all physics specialists
            return [
                PhysicsSpecialistAgent(f'physics_{i}')
                for i in range(5)
            ]
        
        elif stage == 'language':
            # Language: mix of physics and language specialists
            return [
                PhysicsSpecialistAgent(f'physics_{i}') for i in range(2)
            ] + [
                LanguageSpecialistAgent(f'language_{i}') for i in range(3)
            ]
        
        elif stage == 'reasoning':
            # Reasoning: diverse generalists
            return [
                PhysicsSpecialistAgent('physics_0'),
                LanguageSpecialistAgent('language_0'),
            ] + [
                GeneralistAgent(f'generalist_{i}') for i in range(3)
            ]
    
    def evaluate_batch(self, primitives, experiences, stage):
        """All agents evaluate batch."""
        
        results = {}
        
        for primitive in primitives:
            # Each agent evaluates
            evaluations = [
                agent.evaluate_primitive(primitive, experiences)
                for agent in self.agents
            ]
            
            # Consensus analysis
            consensus = PeerConsensus(n_peers=len(self.agents))
            result = consensus.evaluate_primitive(primitive, evaluations)
            
            results[primitive.id] = result
        
        return results
```

---

## Batch Frequency by Stage

### Stage 1: Physical Foundation

**Batch params:**
- Size: 100 experiences OR 5 new primitives
- Max duration: 1 hour
- Frequency: ~Every 30 minutes of learning

**Why frequent:**
- Foundation is critical
- Errors compound quickly
- Physics is unambiguous (fast validation)
- Small primitives (quick to test)

**Peer config:**
- 5 physics specialist agents
- All test against physical reality
- High consensus required (0.9)

### Stage 2: Language Grounding

**Batch params:**
- Size: 200 experiences OR 10 new primitives
- Max duration: 2 hours
- Frequency: ~Every 1 hour of learning

**Why less frequent:**
- Built on validated physics
- Language more ambiguous
- Need context to evaluate
- Larger primitives (more data needed)

**Peer config:**
- 2 physics specialists (check grounding)
- 3 language specialists (check usage)
- Medium consensus required (0.8)

### Stage 3: Complex Reasoning

**Batch params:**
- Size: 500 experiences OR 20 new primitives
- Max duration: 4 hours
- Frequency: ~Every 2 hours of learning

**Why least frequent:**
- Built on validated language
- Reasoning requires context
- Need many examples
- Complex compositions

**Peer config:**
- 1 physics specialist (reality check)
- 1 language specialist (semantic check)
- 3 generalists (reasoning check)
- Lower consensus OK (0.7) - more creativity allowed

---

## Integration with String Collapse

```python
class IntegratedValidation:
    """Combines peer validation with string collapse detection."""
    
    def __init__(self, peer_network, collapse_detector):
        self.peer_network = peer_network
        self.collapse_detector = collapse_detector
    
    def validate_batch(self, batch):
        """Validation + collapse check."""
        
        # Standard peer validation
        peer_results = self.peer_network.evaluate_batch(
            batch.new_primitives,
            batch.experiences,
            batch.stage
        )
        
        # String collapse check
        collapse_scan = self.collapse_detector.scan_for_collapse_risk()
        
        # Combine insights
        for primitive_id, peer_result in peer_results.items():
            # Is primitive at collapse risk?
            at_risk = any(
                risk['primitive'].id == primitive_id
                for risk in collapse_scan
            )
            
            if at_risk and peer_result['status'] == 'REJECTED':
                # Both peers AND collapse detector agree
                # → Safe to trigger collapse
                print(f"Converged evidence: Primitive {primitive_id} should collapse")
                peer_result['action'] = 'collapse'
                peer_result['collapse_safe'] = True
            
            elif at_risk and peer_result['status'] == 'VALIDATED':
                # Collapse risk but peers validate
                # → Strengthen from peer support
                print(f"Peer support rescues: Primitive {primitive_id}")
                peer_result['action'] = 'strengthen'
                peer_result['peer_rescue'] = True
        
        return peer_results
```

---

## Metrics and Monitoring

### Batch Health Metrics

```python
class BatchMetrics:
    """Track validation health over time."""
    
    def __init__(self):
        self.batch_history = []
    
    def record_batch(self, batch, results):
        """Record batch statistics."""
        
        metrics = {
            'batch_id': batch.batch_id,
            'stage': batch.stage,
            'n_experiences': len(batch.experiences),
            'n_new_primitives': len(batch.new_primitives),
            'n_validated': sum(1 for r in results.values() if r['status'] == 'VALIDATED'),
            'n_uncertain': sum(1 for r in results.values() if r['status'] == 'UNCERTAIN'),
            'n_rejected': sum(1 for r in results.values() if r['status'] == 'REJECTED'),
            'mean_consensus': np.mean([r['consensus'] for r in results.values()]),
            'duration': batch.end_time - batch.start_time,
        }
        
        self.batch_history.append(metrics)
    
    def health_check(self):
        """Is learning progressing healthily?"""
        
        recent = self.batch_history[-10:]  # Last 10 batches
        
        # Rejection rate
        rejection_rate = np.mean([
            b['n_rejected'] / (b['n_new_primitives'] + 1e-8)
            for b in recent
        ])
        
        # Consensus trend
        consensus_trend = np.polyfit(
            range(len(recent)),
            [b['mean_consensus'] for b in recent],
            deg=1
        )[0]  # Slope
        
        # Warnings
        warnings = []
        
        if rejection_rate > 0.5:
            warnings.append('HIGH_REJECTION_RATE')
        
        if consensus_trend < -0.01:
            warnings.append('DECLINING_CONSENSUS')
        
        return {
            'status': 'HEALTHY' if not warnings else 'NEEDS_ATTENTION',
            'warnings': warnings,
            'rejection_rate': rejection_rate,
            'consensus_trend': consensus_trend
        }
```

---

## Summary

**Batch Learning:**
- 100-500 experiences per batch (stage-dependent)
- Validate after each batch
- Catch errors early

**Peer Consensus:**
- 5 agents evaluate each batch
- 80%+ consensus required (stage-dependent)
- Reject/correct low-consensus primitives

**String Collapse Integration:**
- Check for collapse risk during validation
- Peer rejection + high tension → safe to collapse
- Peer validation + high tension → strengthen

**Frequency:**
- Physical: Every 30 min
- Language: Every hour
- Reasoning: Every 2 hours

**Result:** Continuous learning with periodic quality gates

---

**Status:** FORMALIZED  
**Ready for implementation:** YES  
**Next steps:** Implement BatchValidator and PeerNetwork  
**Next checkpoint:** Validate with multi-agent experiments
