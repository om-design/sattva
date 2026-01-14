"""Correction mechanisms for bad primitives.

Implements:
1. External Observer - detects patterns substrate can't see
2. Threshold Flip - accumulates evidence and flips belief
3. Cascade Collapse - rapid restructuring when foundation fails
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from .gated_coupling_dynamics import PrimitivePattern


@dataclass
class CorrectionEvidence:
    """Evidence for or against a primitive."""
    observation: str
    strength: float  # 0-1
    source: str  # "external", "internal", "sensory"
    contradicts_primitive: bool


class ExternalObserver:
    """Detects patterns the substrate can't see (looking through broken lens).
    
    Like a therapist or friend who can see your patterns because
    they're not using your substrate.
    """
    
    def __init__(self, substrate):
        self.substrate = substrate
        self.detected_patterns = []
    
    def observe_behavior(self, behavior_history: List[Dict]) -> List[str]:
        """Look for repeated patterns in actions (not thoughts).
        
        Args:
            behavior_history: list of {action, context, outcome}
            
        Returns:
            list of detected pattern descriptions
        """
        patterns = []
        
        # Look for repetition
        action_sequences = [b['action'] for b in behavior_history]
        
        # Simple pattern detection: repeated actions in similar contexts
        for i in range(len(behavior_history) - 2):
            if behavior_history[i]['action'] == behavior_history[i+1]['action'] == behavior_history[i+2]['action']:
                if behavior_history[i]['outcome'] == 'negative':
                    patterns.append(f"Repeated: {behavior_history[i]['action']} despite negative outcome")
        
        return patterns
    
    def infer_underlying_primitive(self, patterns: List[str]) -> Optional[str]:
        """Infer what primitive might be causing these patterns."""
        if not patterns:
            return None
        
        # Simple heuristic: if actions sabotage success, infer negative self-belief
        if any('despite negative' in p for p in patterns):
            return "Primitive causing self-sabotage detected"
        
        return None
    
    def provide_external_reference(self, suspected_primitive: PrimitivePattern) -> CorrectionEvidence:
        """Show the pattern applied to someone else as reference.
        
        'If this is wrong when applied to them, it's wrong for you too.'
        """
        # Simulate: "I would tell someone else this primitive is false"
        return CorrectionEvidence(
            observation="When this pattern appears in others, I recognize it as harmful",
            strength=0.8,
            source="external",
            contradicts_primitive=True
        )


class ThresholdFlip:
    """Belief reversal when anomalies cross threshold.
    
    Inspired by BIAS tool's 180° presumption reversal.
    """
    
    def __init__(self, 
                 primitive: PrimitivePattern,
                 flip_threshold: float = 0.6):
        self.primitive = primitive
        self.mainstream_confidence = 1.0  # start believing primitive
        self.counter_confidence = 0.0
        
        self.anomalies: List[str] = []
        self.conflicts_of_interest: List[str] = []
        self.counter_evidence: List[CorrectionEvidence] = []
        
        self.flip_threshold = flip_threshold
        self.has_flipped = False
        
        # Track history
        self.confidence_history = []
    
    def log_anomaly(self, observation: str):
        """Track observation that doesn't fit primitive."""
        self.anomalies.append(observation)
        
        # Anomalies reduce confidence
        self.mainstream_confidence *= 0.95
        self.counter_confidence += 0.05
    
    def detect_conflict_of_interest(self, source: str, reason: str):
        """Scan for conflicts in who taught/reinforced primitive.
        
        Example: Abuser needed victim to believe false primitive.
        """
        self.conflicts_of_interest.append(f"{source}: {reason}")
        
        # 180° presumption reversal
        if len(self.conflicts_of_interest) > 0:
            self.mainstream_confidence *= 0.5
            self.counter_confidence = min(1.0, self.counter_confidence * 2.0)
    
    def add_counter_evidence(self, evidence: CorrectionEvidence):
        """Add evidence contradicting primitive."""
        self.counter_evidence.append(evidence)
        self.counter_confidence += evidence.strength * 0.1
    
    def update(self) -> bool:
        """Update confidences and check for flip.
        
        Returns:
            True if flip occurred
        """
        # Normalize
        total = self.mainstream_confidence + self.counter_confidence
        if total > 0:
            norm_counter = self.counter_confidence / total
        else:
            norm_counter = 0.5
        
        self.confidence_history.append({
            'mainstream': self.mainstream_confidence / total if total > 0 else 0.5,
            'counter': norm_counter,
            'n_anomalies': len(self.anomalies),
            'n_conflicts': len(self.conflicts_of_interest)
        })
        
        # Check for flip
        if norm_counter > self.flip_threshold and not self.has_flipped:
            self.execute_flip()
            return True
        
        return False
    
    def execute_flip(self):
        """Execute the belief reversal."""
        self.has_flipped = True
        
        # Swap confidences
        self.mainstream_confidence, self.counter_confidence = \
            self.counter_confidence, self.mainstream_confidence
        
        # Now pursuing opposite assessment
        print(f"  🔄 FLIP EXECUTED: Now pursuing counter-narrative")
        print(f"     Anomalies accumulated: {len(self.anomalies)}")
        print(f"     Conflicts detected: {len(self.conflicts_of_interest)}")
        print(f"     Counter-evidence: {len(self.counter_evidence)}")


class CascadeCollapse:
    """Single crack can collapse entire false structure.
    
    Like house of cards: remove foundation → everything falls.
    """
    
    def __init__(self, primitives: List[PrimitivePattern]):
        self.primitives = primitives
        self.dependency_graph = self.build_dependency_graph()
    
    def build_dependency_graph(self) -> Dict[int, List[int]]:
        """Build graph of which primitives depend on which.
        
        For now, simple heuristic: primitives with similar activation patterns
        are likely dependent.
        """
        graph = {i: [] for i in range(len(self.primitives))}
        
        for i in range(len(self.primitives)):
            for j in range(len(self.primitives)):
                if i != j:
                    # If patterns are similar, j might depend on i
                    sim = self.primitives[i].pattern.similarity(self.primitives[j].pattern)
                    if sim > 0.5:
                        graph[i].append(j)
        
        return graph
    
    def find_load_bearing_primitives(self) -> List[int]:
        """Find primitives that many others depend on."""
        dependency_counts = [len(deps) for deps in self.dependency_graph.values()]
        load_bearing = []
        
        for i, count in enumerate(dependency_counts):
            if count > len(self.primitives) * 0.2:  # supports >20% of others
                load_bearing.append(i)
        
        return load_bearing
    
    def apply_strong_counter_evidence(self, 
                                      primitive_idx: int,
                                      evidence: CorrectionEvidence) -> bool:
        """Hit primitive with strong counter-evidence.
        
        Returns:
            True if cascade triggered
        """
        if not evidence.contradicts_primitive:
            return False
        
        if evidence.source != "external":
            return False
        
        if evidence.strength < 0.7:  # needs to be strong
            return False
        
        # Strong external counter-evidence hits foundation
        primitive = self.primitives[primitive_idx]
        primitive.depth *= 0.1  # dramatic reduction
        
        print(f"  💥 FOUNDATION HIT: Primitive {primitive_idx} depth reduced to {primitive.depth:.3f}")
        
        # Check if this triggers cascade
        if primitive.depth < 0.2:
            self.trigger_cascade(primitive_idx)
            return True
        
        return False
    
    def trigger_cascade(self, failed_primitive_idx: int):
        """Collapse all dependent primitives."""
        print(f"  🌊 CASCADE TRIGGERED from primitive {failed_primitive_idx}")
        
        # Find dependents
        dependents = self.dependency_graph[failed_primitive_idx]
        collapsed = [failed_primitive_idx]
        
        # Recursive collapse
        to_process = dependents[:]
        while to_process:
            idx = to_process.pop(0)
            if idx in collapsed:
                continue
            
            # Reduce depth
            self.primitives[idx].depth *= 0.3
            collapsed.append(idx)
            
            print(f"     → Primitive {idx} collapsed (depth now {self.primitives[idx].depth:.3f})")
            
            # Check if this triggers further cascade
            if self.primitives[idx].depth < 0.2:
                to_process.extend(self.dependency_graph[idx])
        
        print(f"  ✓ Cascade complete: {len(collapsed)} primitives affected")
        
        return collapsed
