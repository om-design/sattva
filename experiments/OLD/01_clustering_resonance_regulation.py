#!/usr/bin/env python3
"""
Experiment 1: Clustering, Resonance, and Self-Regulation

Tests:
1. Natural clustering from geometric similarity
2. Resonance spreading within clusters
3. Potential for runaway activation
4. Self-regulation mechanisms

Date: January 9, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time

print("="*70)
print("EXPERIMENT 1: CLUSTERING, RESONANCE, AND SELF-REGULATION")
print("="*70)
print()

# ============================================================================
# SUBSTRATE DEFINITION
# ============================================================================

@dataclass
class Neuron:
    """A neuron in the substrate."""
    id: int
    position: np.ndarray  # 3D spatial position
    activation: float = 0.0
    connections: Dict[int, float] = None  # neuron_id -> strength
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = {}

class Substrate:
    """Geometric substrate with spatial organization."""
    
    def __init__(self, n_neurons=1000, space_dim=3, connection_radius=0.3):
        print(f"Initializing substrate: {n_neurons} neurons in {space_dim}D space")
        
        self.n_neurons = n_neurons
        self.space_dim = space_dim
        self.connection_radius = connection_radius
        
        # Create neurons with random positions
        self.neurons = []
        for i in range(n_neurons):
            position = np.random.randn(space_dim) * 0.5
            neuron = Neuron(id=i, position=position)
            self.neurons.append(neuron)
        
        # Create sparse connections based on spatial proximity
        self._create_connections()
        
        # Activity tracking
        self.total_activity_history = []
        self.cluster_history = []
        
        print(f"  Created {len(self.neurons)} neurons")
        print(f"  Average connections per neuron: {self._avg_connections():.1f}")
        print()
    
    def _create_connections(self):
        """Create connections between nearby neurons."""
        print("  Creating spatial connections...")
        total_connections = 0
        
        for i, neuron_i in enumerate(self.neurons):
            for j, neuron_j in enumerate(self.neurons):
                if i >= j:
                    continue
                
                # Distance between neurons
                dist = np.linalg.norm(neuron_i.position - neuron_j.position)
                
                # Connect if within radius (with probability)
                if dist < self.connection_radius:
                    # Strength inversely proportional to distance
                    strength = (self.connection_radius - dist) / self.connection_radius
                    
                    # Add some randomness
                    if np.random.random() < 0.7:  # 70% connection probability
                        neuron_i.connections[j] = strength
                        neuron_j.connections[i] = strength
                        total_connections += 1
        
        print(f"  Created {total_connections} connections")
    
    def _avg_connections(self):
        """Average number of connections per neuron."""
        return np.mean([len(n.connections) for n in self.neurons])
    
    def activate_pattern(self, pattern: np.ndarray, strength=1.0):
        """Activate neurons matching pattern."""
        for neuron in self.neurons:
            # Similarity = inverse distance to pattern
            dist = np.linalg.norm(neuron.position - pattern)
            similarity = np.exp(-dist**2 / 0.1)  # Gaussian kernel
            
            neuron.activation = similarity * strength
    
    def propagate_activation(self, steps=1, decay=0.1):
        """Propagate activation through connections."""
        for step in range(steps):
            # Calculate new activations
            new_activations = np.zeros(self.n_neurons)
            
            for neuron in self.neurons:
                # Receive activation from connected neurons
                incoming = 0.0
                for neighbor_id, strength in neuron.connections.items():
                    incoming += self.neurons[neighbor_id].activation * strength
                
                # New activation = current + incoming - decay
                new_activations[neuron.id] = (
                    neuron.activation * (1 - decay) + incoming * 0.3
                )
            
            # Update activations
            for i, neuron in enumerate(self.neurons):
                neuron.activation = new_activations[i]
    
    def get_total_activity(self):
        """Total activation across all neurons."""
        return sum(n.activation for n in self.neurons)
    
    def get_active_neurons(self, threshold=0.1):
        """Get neurons above activation threshold."""
        return [n for n in self.neurons if n.activation > threshold]
    
    def find_clusters(self, threshold=0.1):
        """Find clusters of active neurons."""
        active = self.get_active_neurons(threshold)
        
        if len(active) == 0:
            return []
        
        # Simple clustering: group nearby active neurons
        clusters = []
        used = set()
        
        for neuron in active:
            if neuron.id in used:
                continue
            
            # Start new cluster
            cluster = [neuron]
            used.add(neuron.id)
            
            # Find connected active neurons
            queue = [neuron]
            while queue:
                current = queue.pop(0)
                
                for neighbor_id in current.connections:
                    if neighbor_id in used:
                        continue
                    
                    neighbor = self.neurons[neighbor_id]
                    if neighbor.activation > threshold:
                        cluster.append(neighbor)
                        used.add(neighbor_id)
                        queue.append(neighbor)
            
            clusters.append(cluster)
        
        return clusters
    
    def reset(self):
        """Reset all activations to zero."""
        for neuron in self.neurons:
            neuron.activation = 0.0

# ============================================================================
# REGULATION MECHANISMS
# ============================================================================

class RegulationMechanism:
    """Base class for regulation mechanisms."""
    
    def __init__(self, name):
        self.name = name
        self.active = True
    
    def regulate(self, substrate: Substrate):
        """Apply regulation to substrate."""
        raise NotImplementedError

class NoRegulation(RegulationMechanism):
    """No regulation - test for runaway."""
    
    def __init__(self):
        super().__init__("None")
    
    def regulate(self, substrate: Substrate):
        # Do nothing
        pass

class GlobalNormalization(RegulationMechanism):
    """Normalize total activity to target level."""
    
    def __init__(self, target_activity=50.0):
        super().__init__("Global Normalization")
        self.target_activity = target_activity
    
    def regulate(self, substrate: Substrate):
        total = substrate.get_total_activity()
        
        if total > 0:
            scale = self.target_activity / total
            for neuron in substrate.neurons:
                neuron.activation *= scale

class LocalInhibition(RegulationMechanism):
    """Nearby neurons inhibit each other."""
    
    def __init__(self, inhibition_radius=0.2, strength=0.1):
        super().__init__("Local Inhibition")
        self.inhibition_radius = inhibition_radius
        self.strength = strength
    
    def regulate(self, substrate: Substrate):
        # Calculate inhibition for each neuron
        inhibitions = np.zeros(substrate.n_neurons)
        
        for i, neuron_i in enumerate(substrate.neurons):
            if neuron_i.activation < 0.01:
                continue
            
            # Inhibit nearby neurons
            for j, neuron_j in enumerate(substrate.neurons):
                if i == j:
                    continue
                
                dist = np.linalg.norm(neuron_i.position - neuron_j.position)
                
                if dist < self.inhibition_radius:
                    # Stronger inhibition when both active
                    inhibition = (neuron_i.activation * neuron_j.activation * 
                                self.strength * (1 - dist / self.inhibition_radius))
                    inhibitions[j] += inhibition
        
        # Apply inhibition
        for i, neuron in enumerate(substrate.neurons):
            neuron.activation = max(0, neuron.activation - inhibitions[i])

class HomeostaticRegulation(RegulationMechanism):
    """Maintain stable average activity level."""
    
    def __init__(self, target_avg=0.1, timescale=10):
        super().__init__("Homeostatic")
        self.target_avg = target_avg
        self.timescale = timescale
        self.activity_history = []
    
    def regulate(self, substrate: Substrate):
        # Track average activity
        avg_activity = substrate.get_total_activity() / substrate.n_neurons
        self.activity_history.append(avg_activity)
        
        if len(self.activity_history) > self.timescale:
            self.activity_history.pop(0)
        
        # Calculate scaling factor
        recent_avg = np.mean(self.activity_history)
        
        if recent_avg > 0:
            scale = self.target_avg / recent_avg
            scale = 0.9 * scale + 0.1  # Gradual adjustment
            
            for neuron in substrate.neurons:
                neuron.activation *= scale

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ResonanceExperiment:
    """Run clustering and resonance experiment."""
    
    def __init__(self, substrate: Substrate, regulation: RegulationMechanism):
        self.substrate = substrate
        self.regulation = regulation
        self.results = {
            'activity': [],
            'n_clusters': [],
            'max_activation': [],
            'avg_cluster_size': [],
            'regulation_name': regulation.name
        }
    
    def run(self, n_patterns=5, steps_per_pattern=20, propagation_steps=3):
        """Run experiment."""
        
        print(f"Running experiment with {self.regulation.name} regulation")
        print(f"  {n_patterns} patterns, {steps_per_pattern} steps each")
        print()
        
        # Generate random patterns
        patterns = [np.random.randn(self.substrate.space_dim) * 0.3 
                   for _ in range(n_patterns)]
        
        for pattern_idx, pattern in enumerate(patterns):
            print(f"Pattern {pattern_idx + 1}/{n_patterns}:")
            
            # Reset substrate
            self.substrate.reset()
            
            # Activate pattern
            self.substrate.activate_pattern(pattern, strength=1.0)
            
            initial_activity = self.substrate.get_total_activity()
            print(f"  Initial activity: {initial_activity:.2f}")
            
            # Run dynamics
            for step in range(steps_per_pattern):
                # Propagate activation
                self.substrate.propagate_activation(steps=propagation_steps)
                
                # Apply regulation
                self.regulation.regulate(self.substrate)
                
                # Record metrics
                activity = self.substrate.get_total_activity()
                clusters = self.substrate.find_clusters(threshold=0.1)
                max_act = max([n.activation for n in self.substrate.neurons])
                avg_cluster_size = (np.mean([len(c) for c in clusters]) 
                                  if clusters else 0)
                
                self.results['activity'].append(activity)
                self.results['n_clusters'].append(len(clusters))
                self.results['max_activation'].append(max_act)
                self.results['avg_cluster_size'].append(avg_cluster_size)
                
                # Print updates every 5 steps
                if step % 5 == 0:
                    print(f"    Step {step:2d}: Activity={activity:6.2f}, "
                          f"Clusters={len(clusters)}, Max={max_act:.3f}")
                
                # Check for runaway
                if activity > 1000:
                    print("    WARNING: RUNAWAY DETECTED! Activity > 1000")
                    break
            
            final_activity = self.substrate.get_total_activity()
            print(f"  Final activity: {final_activity:.2f}")
            print(f"  Change: {final_activity - initial_activity:+.2f}")
            print()
        
        return self.results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results_list):
    """Plot experiment results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Clustering, Resonance, and Regulation Experiments', 
                 fontsize=14, fontweight='bold')
    
    for results in results_list:
        label = results['regulation_name']
        
        # Total activity over time
        axes[0, 0].plot(results['activity'], label=label, linewidth=2)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Total Activity')
        axes[0, 0].set_title('Total Network Activity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Number of clusters
        axes[0, 1].plot(results['n_clusters'], label=label, linewidth=2)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Number of Clusters')
        axes[0, 1].set_title('Cluster Formation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Maximum activation
        axes[1, 0].plot(results['max_activation'], label=label, linewidth=2)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Max Neuron Activation')
        axes[1, 0].set_title('Peak Activation Levels')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Average cluster size
        axes[1, 1].plot(results['avg_cluster_size'], label=label, linewidth=2)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Avg Neurons per Cluster')
        axes[1, 1].set_title('Cluster Size')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/omdesign/code/GitHub/sattva/experiments/01_results.png', 
                dpi=150, bbox_inches='tight')
    print("Results saved to experiments/01_results.png")
    print()

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Run all experiments."""
    
    print("\n" + "="*70)
    print("EXPERIMENTAL CONDITIONS")
    print("="*70)
    print()
    print("Testing regulation mechanisms:")
    print("  1. No Regulation (baseline - test for runaway)")
    print("  2. Global Normalization (maintain total activity)")
    print("  3. Local Inhibition (nearby neurons inhibit)")
    print("  4. Homeostatic (adaptive scaling)")
    print()
    
    # Create substrate (shared across experiments)
    substrate = Substrate(n_neurons=500, space_dim=3, connection_radius=0.3)
    
    # Run experiments with different regulation mechanisms
    regulations = [
        NoRegulation(),
        GlobalNormalization(target_activity=50.0),
        LocalInhibition(inhibition_radius=0.2, strength=0.15),
        HomeostaticRegulation(target_avg=0.1, timescale=10)
    ]
    
    all_results = []
    
    for regulation in regulations:
        print("\n" + "="*70)
        print(f"EXPERIMENT: {regulation.name.upper()}")
        print("="*70)
        print()
        
        experiment = ResonanceExperiment(substrate, regulation)
        results = experiment.run(n_patterns=3, steps_per_pattern=20, 
                                propagation_steps=2)
        all_results.append(results)
        
        # Analyze results
        final_activity = results['activity'][-1]
        max_activity = max(results['activity'])
        avg_clusters = np.mean(results['n_clusters'])
        
        print("SUMMARY:")
        print(f"  Final activity: {final_activity:.2f}")
        print(f"  Max activity reached: {max_activity:.2f}")
        print(f"  Average clusters: {avg_clusters:.1f}")
        
        if max_activity > 500:
            print("  WARNING: Potential runaway behavior")
        elif max_activity < 10:
            print("  WARNING: Network too suppressed")
        else:
            print("  Activity levels stable")
        print()
    
    # Visualize results
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    print()
    
    plot_results(all_results)
    
    # Final analysis
    print("\n" + "="*70)
    print("FINAL ANALYSIS")
    print("="*70)
    print()
    
    print("Key Findings:")
    print()
    
    for i, (reg, results) in enumerate(zip(regulations, all_results)):
        activity = np.array(results['activity'])
        stability = np.std(activity) / (np.mean(activity) + 1e-6)
        
        print(f"{i+1}. {reg.name}:")
        print(f"   Mean activity: {np.mean(activity):.2f}")
        print(f"   Stability (lower is better): {stability:.3f}")
        
        if stability < 0.5:
            print(f"   Highly stable")
        elif stability < 1.0:
            print(f"   Moderately stable")
        else:
            print(f"   Unstable / Runaway")
        print()
    
    print("="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print()

if __name__ == "__main__":
    main()
