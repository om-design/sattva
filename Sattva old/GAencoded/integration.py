# sattva_engine_v9.py (your current engine with novelty & tension)

from ga_encoding import GAEncoding
import numpy as np

def build_engine_with_ga():
    ga = GAEncoding(dim_base=8, dim_instr=8)
    eng = Engine(dim=ga.dim, base_activation_threshold=0.1)
    return eng, ga

def demo_ga_setup():
    eng, ga = build_engine_with_ga()

    # Define symbolic bases/instructions
    # Think: "triangle", "hexagram", "rotation", "context_food"
    bases = ["B_triangle", "B_hexagram"]
    instrs = ["I_rot", "I_food", "I_struct"]

    # Example programs
    A_emb = ga.encode_program(["B_hexagram"], ["I_rot"])          # hexagram ≈ triangle+rot, abstract
    P_emb = ga.encode_program(["B_triangle"], ["I_food"])         # pizza slice
    T_emb = ga.encode_program(["B_triangle"], ["I_struct"])       # truss element

    pid_A = eng.create_primitive(A_emb, complexity=2)
    pid_P = eng.create_primitive(P_emb, complexity=2)
    pid_T = eng.create_primitive(T_emb, complexity=2)

    # Now you can run eng.activate_input(...) with GA-encoded inputs
