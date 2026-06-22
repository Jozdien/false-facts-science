"""Sanity-check the task-shortcut confound across ALL of the paper's semi-synthetic datasets.

For each *evaluated* (e2_type, e3_type) attribute (from the paper's EXPERIMENT_CONFIG), measure
the fraction of entities whose answer is derivable from the bridge entity's NAME alone — i.e.
answerable by a surface transform of e2, no genuine second-hop fact lookup needed.

Derivability heuristic (deliberately conservative, with examples printed to validate):
  - answer (>=3 chars) is a substring of the entity name, or the name is a substring of it
  - short answer (<=2 chars, e.g. element symbols) is a prefix of the entity name
  (case-insensitive; this catches city-in-"University of X", py⊂python, H⊂Hydrogen)

Usage: uv run scripts/shortcut_scan.py
"""

import ast
import glob
import json

from twohop.common import RESULTS_DIR, TWOHOP_REPO

TBL = TWOHOP_REPO / "latent_reasoning/datagen/semi_synthetic/data/e2s_with_attributes"
SH = TWOHOP_REPO / "experiments/semi_synthetic"
JMAP = {
    "parks": "national_parks.json", "chemical_elements": "chemical_elements.json",
    "programming_languages": "programming_languages.json", "world_heritage_sites": "world_heritage_sites.json",
    "video_game_consoles": "video_game_consoles.json", "famous_paintings": "famous_paintings.json",
    "cathedrals": "cathedrals.json", "bridges": "bridges.json", "operas": "operas.json",
    "telescopes": "telescopes.json", "ancient_cities": "ancient_cities.json",
    "mountain_peaks": "mountain_peaks.json", "universities": "universities.json",
    "constellations": "constellations.json", "ships": "ships.json", "newspapers": "newspapers.json",
    "subway_systems": "subway_systems.json",
}


def experiment_attrs():
    tree = ast.parse((SH / "plot.py").read_text())
    node = next(n.value for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "EXPERIMENT_CONFIG" for t in n.targets))
    out = {}
    for k, v in zip(node.keys, node.values):
        ds = ast.literal_eval(k).split("/")[-1].removesuffix(".yaml")
        for kk, vv in zip(v.keys, v.values):
            if ast.literal_eval(kk) == "attributes":
                out[ds] = ast.literal_eval(vv)
    return out


def derivable(entity, ans):
    e, a = entity.lower().strip(), str(ans).lower().strip()
    if len(a) < 2:
        return False
    if len(a) <= 2:
        return e.startswith(a)
    return a in e or e in a


def main():
    attrs = experiment_attrs()
    total_items = total_deriv = 0
    heavy, partial, clean = [], [], []
    examples = {}
    for ds, alist in attrs.items():
        if ds not in JMAP:
            continue
        recs = json.loads((TBL / JMAP[ds]).read_text())
        ekey = next(k for k in recs[0] if all(k in r for r in recs) and k not in alist)
        for a in alist:
            rows = [(r[ekey], r[a]) for r in recs if a in r]
            if not rows:
                continue
            n = sum(derivable(e, v) for e, v in rows)
            frac = n / len(rows)
            total_items += len(rows)
            total_deriv += n
            tag = (heavy if frac >= 0.5 else partial if frac >= 0.1 else clean)
            tag.append((ds, a, frac, len(rows)))
            if frac >= 0.1:
                examples[(ds, a)] = [(e, v) for e, v in rows if derivable(e, v)][:2]

    print(f"=== ALL evaluated attributes: {len(heavy) + len(partial) + len(clean)} across {len(attrs)} datasets ===")
    print(f"overall: {total_deriv}/{total_items} evaluated (entity,attribute) items name-derivable "
          f"= {100 * total_deriv / total_items:.1f}%\n")
    for name, lst in [("HEAVY (≥50% derivable)", heavy), ("PARTIAL (10–50%)", partial)]:
        print(f"--- {name}: {len(lst)} attributes ---")
        for ds, a, frac, n in sorted(lst, key=lambda x: -x[2]):
            ex = examples.get((ds, a), [])
            exs = "; ".join(f"{e!r}→{v!r}" for e, v in ex)
            print(f"  {ds:22s} {a:20s} {frac:5.0%} (n={n})   e.g. {exs}")
    print(f"\n--- CLEAN (<10% derivable): {len(clean)} attributes ---")
    for ds, a, frac, n in sorted(clean, key=lambda x: -x[2])[:8]:
        print(f"  {ds:22s} {a:20s} {frac:5.0%}")
    print(f"  ... and {max(0, len(clean) - 8)} more")
    leverage()


def leverage():
    """Prevalence ≠ impact: for the datasets we actually trained (rank_compare), how much of
    the raw free-gen two-hop score comes from the shortcut-prone attributes?"""
    from twohop.battery import SHORTCUT_ATTRS  # noqa: PLC0415
    print("\n=== LEVERAGE (free-gen two-hop accuracy, QA-SFT; the metric a paper-style average pools) ===")
    for ds in ["programming_languages", "universities"]:
        files = sorted(glob.glob(str(RESULTS_DIR / f"rank_compare/rank-qasft-{ds}-s0/samples_*.json")))
        if not files:
            continue
        s = json.loads(open(files[-1]).read())
        sc = SHORTCUT_ATTRS.get(ds, set())
        tot_c = tot_n = scc = scn = 0
        for k in [k for k in s if k.endswith("_nocot")]:
            a = k[:-6]
            c, n = sum(x["correct"] for x in s[k]), len(s[k])
            tot_c, tot_n = tot_c + c, tot_n + n
            if a in sc:
                scc, scn = scc + c, scn + n
        print(f"  {ds}: overall {tot_c}/{tot_n}={100 * tot_c / tot_n:.0f}%; shortcut attrs = "
              f"{scn}/{tot_n} of items but {scc}/{tot_c} = {100 * scc / max(tot_c, 1):.0f}% of correct answers")


if __name__ == "__main__":
    main()
