#!/usr/bin/env python3
"""Per-level accuracy and error locality analysis for reviewer Q9."""
import argparse
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from data.data import load_and_encode_taxonomy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', required=True, help='Submission CSV')
    parser.add_argument('--solution', default='solution_kaggle_2025.csv')
    parser.add_argument('--taxonomy', default='data/taxonomy.csv')
    parser.add_argument('--distance-matrix', default='data/distance_matrix.csv')
    args = parser.parse_args()

    # Load taxonomy
    levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        args.taxonomy, levels
    )

    # Build species -> ancestor mapping for each level
    # taxonomy_df has columns: kingdom, phylum, class, order, family, genus, species (all string)
    species_to_ancestors = {}
    for _, row in taxonomy_df.iterrows():
        sp = row['species']
        species_to_ancestors[sp] = {lev: row[lev] for lev in levels}

    # Load solution and submission
    sol = pd.read_csv(args.solution)
    sub = pd.read_csv(args.submission)

    # Merge on annotation_id
    merged = sol.merge(sub, on='annotation_id', suffixes=('_true', '_pred'))

    # Get true and predicted species (column is concept_name)
    true_species = merged['concept_name_true'].values
    pred_species = merged['concept_name_pred'].values

    # Compute per-level accuracy
    print("=" * 70)
    print("Per-Level Accuracy Analysis")
    print("=" * 70)

    level_correct = {}
    for level in levels:
        correct = 0
        total = 0
        for ts, ps in zip(true_species, pred_species):
            if ts in species_to_ancestors and ps in species_to_ancestors:
                if species_to_ancestors[ts][level] == species_to_ancestors[ps][level]:
                    correct += 1
            total += 1
        acc = correct / total if total > 0 else 0
        level_correct[level] = (correct, total, acc)
        print(f"  {level:10s}: {correct:4d}/{total:4d} = {acc:.1%}")

    # Error locality: where do errors first diverge?
    print("\n" + "=" * 70)
    print("Error Locality Distribution (LCA level of errors)")
    print("=" * 70)

    lca_counts = {lev: 0 for lev in levels}
    lca_counts['correct'] = 0
    total_samples = len(true_species)

    for ts, ps in zip(true_species, pred_species):
        if ts == ps:
            lca_counts['correct'] += 1
            continue
        if ts not in species_to_ancestors or ps not in species_to_ancestors:
            continue
        # Find LCA level (deepest level where they agree)
        lca_level = None
        for lev in levels:
            if species_to_ancestors[ts][lev] == species_to_ancestors[ps][lev]:
                lca_level = lev
            else:
                break
        if lca_level is None:
            # Disagree at kingdom level
            lca_counts['kingdom'] += 1
        else:
            # Error is at the level BELOW lca_level
            lca_idx = levels.index(lca_level)
            if lca_idx + 1 < len(levels):
                error_level = levels[lca_idx + 1]
                lca_counts[error_level] += 1

    print(f"  Correct (species match): {lca_counts['correct']:4d} ({lca_counts['correct']/total_samples:.1%})")
    print(f"\n  Error breakdown by divergence level:")
    total_errors = total_samples - lca_counts['correct']
    for lev in levels:
        cnt = lca_counts[lev]
        if lev == 'kingdom':
            print(f"    Different {lev:10s}: {cnt:4d} ({cnt/total_errors:.1%} of errors, distance=7)")
        elif lev == 'species':
            print(f"    Same genus, diff {lev:7s}: {cnt:4d} ({cnt/total_errors:.1%} of errors, distance=1)")
        else:
            parent_idx = levels.index(lev) - 1
            parent_lev = levels[parent_idx]
            dist = len(levels) - levels.index(lev)
            print(f"    Same {parent_lev:7s}, diff {lev:7s}: {cnt:4d} ({cnt/total_errors:.1%} of errors, distance={dist})")

    # Distance distribution
    dist_matrix = pd.read_csv(args.distance_matrix, index_col=0)
    distances = []
    for ts, ps in zip(true_species, pred_species):
        if ts in dist_matrix.index and ps in dist_matrix.columns:
            distances.append(dist_matrix.loc[ts, ps])
        else:
            distances.append(7)  # max distance for unknown
    distances = np.array(distances)

    print(f"\n" + "=" * 70)
    print("Distance Distribution")
    print("=" * 70)
    for d in range(8):
        cnt = np.sum(distances == d)
        print(f"  Distance {d}: {cnt:4d} ({cnt/len(distances):.1%})")
    print(f"  Mean: {distances.mean():.3f}, Median: {np.median(distances):.1f}")

    # LaTeX table output
    print(f"\n% LaTeX table for per-level accuracy")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Level & Correct & Total & Accuracy (\%) \\")
    print(r"\midrule")
    for level in levels:
        c, t, a = level_correct[level]
        print(f"{level.capitalize()} & {c} & {t} & {a*100:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == '__main__':
    main()
