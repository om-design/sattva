# vis_sattva/engine/config.py

from dataclasses import dataclass

@dataclass
class FieldConfig:
    """
    Core hyperparameters for the vis_sattva field dynamics.
    All values are conservative defaults and should be easy to tune.
    """

    # Energy & dynamics
    energy_max: float = 1.0       # Total allowed sum(a_i^2) before rescaling
    damping: float = 0.1          # How strongly activations are pulled down each step

    # Plasticity (Hebbian + usage tracking)
    lr_hebb: float = 1e-3         # Hebbian learning rate for synapse weights
    ema_alpha: float = 1e-3       # Usage EMA rate (u_ij)
    myelin_beta: float = 1e-4     # Myelination adaptation rate (m_ij toward u_ij)
    decay_eps: float = 1e-5       # Base decay factor for weak, unmyelinated synapses

    # Trauma / infant phase
    trauma_mode: bool = True      # Whether trauma behavior is enabled
    trauma_threshold: float = 5.0 # Tension level that triggers trauma (to be tuned)
    trauma_lr_boost: float = 50.0 # Multiplier on lr_hebb during a trauma event
    trauma_min_myelin: float = 0.8  # Minimum myelin level set on trauma synapses

    # Logging / diagnostics
    log_energy: bool = True       # Track energy over steps
    log_tension: bool = True      # Track tension over steps
