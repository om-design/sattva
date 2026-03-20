import sys
import numpy as np
import importlib.util

# Dynamically import sattva-engine_v8.py due to hyphen in filename
file_path = "sattva-engine_v8.py"
spec = importlib.util.spec_from_file_location("sattva_engine", file_path)
sattva_engine = importlib.util.module_from_spec(spec)
sys.modules["sattva_engine"] = sattva_engine
spec.loader.exec_module(sattva_engine)

Engine = sattva_engine.Engine

def generate_deal_vector(base_vector, noise_level=0.1):
    """Generate a deal vector based on a prototype with some noise."""
    noise = np.random.normal(0, noise_level, len(base_vector))
    vector = base_vector + noise
    # Normalize to keep it unit length like
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def run_simulation():
    dim = 8
    print("Initializing SATTVA Engine...")
    eng = Engine(dim=dim, base_activation_threshold=0.1, prediction_suppression=0.7, composite_fire_threshold=0.5)
    
    # Define prototypes for "successful deals"
    # E.g., Tech companies with recent funding (Vector A)
    # E.g., Healthcare companies with high growth (Vector B)
    proto_tech_deal = np.array([0.8, 0.6, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0])
    proto_health_deal = np.array([0.0, 0.0, 0.0, 0.9, 0.7, 0.0, 0.2, 0.0])
    
    print("\n--- Phase 1: Training on Successful Deals ---")
    # Feed 50 successful tech deals and 50 successful health deals
    for i in range(50):
        deal_t = generate_deal_vector(proto_tech_deal, noise_level=0.05)
        # Create primitives directly to populate the field initially, mimicking memory formation
        if i % 10 == 0:
            eng.create_primitive(deal_t)
            
        deal_h = generate_deal_vector(proto_health_deal, noise_level=0.05)
        if i % 10 == 0:
            eng.create_primitive(deal_h)
            
        # Activate input and step to allow crystallisation (composites forming)
        eng.activate_input(deal_t, magnitude=1.0)
        eng.step()
        
        eng.activate_input(deal_h, magnitude=1.0)
        eng.step()
    
    print(f"Engine learned {len(eng.primitives)} primitives (including composites).")
    
    print("\n--- Phase 2: Monitoring Stream ---")
    
    stream_scenarios = [
        ("Typical Tech Deal", generate_deal_vector(proto_tech_deal, noise_level=0.05)),
        ("Typical Health Deal", generate_deal_vector(proto_health_deal, noise_level=0.05)),
        ("Completely Unrelated (Retail, Low Growth)", np.array([0.1, 0.1, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0])),
        ("Novel Combination (Tech + Health AI crossover)", generate_deal_vector((proto_tech_deal + proto_health_deal)/2, noise_level=0.05)),
    ]
    
    for name, deal_vector in stream_scenarios:
        # Normalize the stream vector safely
        norm = np.linalg.norm(deal_vector)
        if norm > 0: deal_vector = deal_vector / norm
            
        # Activate the engine
        surprise_scores = eng.activate_input(deal_vector, magnitude=1.0)
        
        # Calculate diagnostics
        triage = eng.triage_score()
        anomaly = eng.anomaly_score()
        epiphanies = eng.epiphany_check(surprise_threshold=0.2)
        
        print(f"\nEvaluating: {name}")
        print(f"  Triage Score:  {triage:.4f}  (Higher means merits LLM attention)")
        print(f"  Anomaly Score: {anomaly:.4f}  (Higher means surprising/novel within field)")
        
        if triage < 0.3:
            print("  Action: AUTO-PROCESS (Routine Match)")
        elif triage < 0.7:
            if epiphanies:
                print("  Action: NOTABLE NOVEL CROSSOVER (Worth LLM summary/tagging)")
            else:
                print("  Action: WEAK SIGNAL (Keep monitoring)")
        else:
            print("  Action: TRIGGER DEEP LLM ANALYSIS (High novelty or complete unknown)")
            
        if epiphanies:
            print(f"  Epiphanies: Engine connected this to {len(epiphanies)} deep patterns!")

if __name__ == "__main__":
    run_simulation()
