"""
inspect_container.py

Quick inspection of the baby SATTVA container:
- Prints global step and curriculum log.
- Shows top primitives by bandwidth and usage.
- Summarizes epiphany events.
"""

from container import SattvaContainer


def main():
    path = "artifacts/snapshots/baby_sattva.pkl"

    c = SattvaContainer.load(path)
    eng = c.engine

    print("=== Container summary ===")
    print("Global step:", c.meta.get("step", 0))
    print("Engine version:", c.meta.get("engine_version", "unknown"))

    curriculum = c.meta.get("curriculum_log", [])
    print("Curriculum phases run:", len(curriculum))
    for phase in curriculum:
        print(
            f"  - {phase['phase']}: steps {phase['start_step']}–{phase['end_step']} "
            f"({phase['steps']} steps, {phase['duration_sec']:.1f}s)"
        )

    print("\n=== Primitive stats ===")
    print("Total primitives:", len(eng.primitives))

    # Collect stats per primitive
    rows = []
    for pid, p in eng.primitives.items():
        rows.append(
            (
                pid,
                p.bandwidth,
                p.access_count,
                p.usage_ema,
                p.energy,
            )
        )

    # Top 10 by bandwidth
    rows_bw = sorted(rows, key=lambda r: r[1], reverse=True)[:10]
    print("\nTop 10 primitives by bandwidth:")
    for pid, bw, acc, use, en in rows_bw:
        print(
            f"  id={pid[:8]}  bw={bw:.3f}  access={acc}  usage_ema={use:.3f}  energy={en:.3f}"
        )

    # Top 10 by usage_ema
    rows_use = sorted(rows, key=lambda r: r[3], reverse=True)[:10]
    print("\nTop 10 primitives by usage_ema:")
    for pid, bw, acc, use, en in rows_use:
        print(
            f"  id={pid[:8]}  usage_ema={use:.3f}  access={acc}  bw={bw:.3f}  energy={en:.3f}"
        )

    print("\n=== Epiphany log ===")
    epi = c.meta.get("epiphany_log", [])
    print("Total epiphany events logged:", len(epi))
    if epi:
        # Show first few
        for e in epi[:5]:
            print(
                f"  phase={e['phase']}  input_id={e['input_id']}  "
                f"tension={e['tension']:.3f}  mean_novelty={e['mean_novelty']:.3f}  "
                f"triage={e['triage']:.3f}  n_ancestors={len(e['epiphanies'])}"
            )


if __name__ == "__main__":
    main()
