# run_from_yaml.py
import os
import importlib
import argparse

import yaml  # pip install pyyaml

from container import SattvaContainer


def parse_args():
    p = argparse.ArgumentParser(description="Run SATTVA training from YAML config.")
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    container_path = cfg["container"]["path"]
    rng_seed = cfg["container"].get("rng_seed", 42)

    if os.path.exists(container_path):
        print(f"Loading container from {container_path}")
        container = SattvaContainer.load(container_path)
    else:
        print("No container found, creating new infant SATTVA")
        container = SattvaContainer.new_infant(rng_seed=rng_seed)

    phases = cfg["run"]["phases"]
    for phase in phases:
        name = phase["name"]
        module_name = phase["module"]
        func_name = phase["func"]
        steps = int(phase["steps"])
        log_every = int(phase.get("log_every", 1000))

        print(f"=== Running phase '{name}' ({module_name}.{func_name}) for {steps} steps ===")
        mod = importlib.import_module(module_name)
        phase_func = getattr(mod, func_name)
        phase_func(container, steps=steps, log_every=log_every, phase_name=name)

        container.save(container_path)
        print(f"Saved container to {container_path}")

    print("All phases complete.")


if __name__ == "__main__":
    main()
