#!/usr/bin/env python3
"""
Result Analysis Script for PharmacoNet Reverse Screening
========================================================

Analyze and visualize reverse screening results.

Usage:
    python scripts/analyze_results.py results/screening_results.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Analyze reverse screening results")
    parser.add_argument("results_csv", type=str, help="Path to results CSV file")
    parser.add_argument("--output_dir", type=str, default="results/analysis",
                       help="Output directory for plots and reports")
    parser.add_argument("--score_threshold", type=float, default=30.0,
                       help="Score threshold for strong hits")
    args = parser.parse_args()

    # Load results
    if not Path(args.results_csv).exists():
        print(f"ERROR: Results file not found: {args.results_csv}")
        sys.exit(1)
    
    print(f"Loading results from: {args.results_csv}")
    df = pd.read_csv(args.results_csv)
    
    if df.empty:
        print("ERROR: Results file is empty!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} results")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal number of query-target pairs: {len(df)}")
    print(f"Number of unique queries: {df['query_name'].nunique()}")
    print(f"Number of unique targets: {df['pharmacophore_model'].nunique()}")
    
    print(f"\nScore statistics:")
    print(f"  Min:    {df['score'].min():.4f}")
    print(f"  Max:    {df['score'].max():.4f}")
    print(f"  Mean:   {df['score'].mean():.4f}")
    print(f"  Median: {df['score'].median():.4f}")
    print(f"  Std:    {df['score'].std():.4f}")
    
    strong_hits = df[df['score'] >= args.score_threshold]
    print(f"\nStrong hits (score >= {args.score_threshold}): {len(strong_hits)}")
    
    # Per-query statistics
    print("\n" + "="*80)
    print("PER-QUERY STATISTICS")
    print("="*80)
    
    query_stats = df.groupby('query_name').agg({
        'score': ['count', 'mean', 'max']
    }).round(4)
    query_stats.columns = ['Total_Hits', 'Mean_Score', 'Max_Score']
    print("\n" + query_stats.to_string())
    
    # Top targets for each query
    print("\n" + "="*80)
    print("TOP 5 TARGETS PER QUERY")
    print("="*80)
    
    for query_name in df['query_name'].unique():
        query_df = df[df['query_name'] == query_name].nlargest(5, 'score')
        print(f"\n{query_name}:")
        for idx, (_, row) in enumerate(query_df.iterrows(), start=1):
            model_name = Path(row['pharmacophore_model']).stem
            print(f"  {idx}. {model_name:50s} Score: {row['score']:8.4f}")
    
    # Promiscuous targets (bind to many queries)
    print("\n" + "="*80)
    print("PROMISCUOUS TARGETS (appear in top 10 for multiple queries)")
    print("="*80)
    
    # Get top 10 for each query
    top_hits = df.groupby('query_name').apply(
        lambda x: x.nlargest(10, 'score')
    ).reset_index(drop=True)
    
    target_counts = top_hits.groupby('pharmacophore_model').size().sort_values(ascending=False)
    promiscuous = target_counts[target_counts > 1]
    
    if len(promiscuous) > 0:
        print("\nTargets appearing in multiple query top-10 lists:")
        for target, count in promiscuous.head(20).items():
            target_name = Path(target).stem
            print(f"  {target_name:50s} Appears in {count} queries")
    else:
        print("\nNo promiscuous targets found (each target matches only one query)")
    
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['score'], bins=50, kde=True)
    plt.axvline(args.score_threshold, color='red', linestyle='--', 
                label=f'Threshold ({args.score_threshold})')
    plt.xlabel('Pharmacophore Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Score Distribution Across All Matches', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    score_dist_path = output_dir / 'score_distribution.png'
    plt.savefig(score_dist_path, dpi=300)
    print(f"  Score distribution plot saved: {score_dist_path}")
    plt.close()
    
    # 2. Box plot by query
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('query_name')
    sns.boxplot(data=df_sorted, x='query_name', y='score')
    plt.axhline(args.score_threshold, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Query Molecule', fontsize=12)
    plt.ylabel('Pharmacophore Score', fontsize=12)
    plt.title('Score Distribution by Query Molecule', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    boxplot_path = output_dir / 'score_by_query_boxplot.png'
    plt.savefig(boxplot_path, dpi=300)
    print(f"  Box plot saved: {boxplot_path}")
    plt.close()
    
    # 3. Top hits heatmap (if multiple queries)
    if df['query_name'].nunique() > 1:
        # Get top 20 targets per query
        top_n = 20
        top_targets = df.groupby('query_name').apply(
            lambda x: x.nlargest(top_n, 'score')['pharmacophore_model'].tolist()
        )
        
        # Get unique targets that appear in any top list
        all_top_targets = set()
        for targets in top_targets:
            all_top_targets.update(targets)
        
        # Create pivot table with simplified names
        pivot_data = []
        for target in all_top_targets:
            target_name = Path(target).stem
            row = {'target': target_name}
            for query in df['query_name'].unique():
                score = df[(df['query_name'] == query) & 
                          (df['pharmacophore_model'] == target)]['score'].values
                row[query] = score[0] if len(score) > 0 else 0
            pivot_data.append(row)
        
        pivot_df = pd.DataFrame(pivot_data).set_index('target')
        pivot_df = pivot_df.loc[pivot_df.max(axis=1).nlargest(30).index]  # Top 30 targets
        
        plt.figure(figsize=(12, 16))
        sns.heatmap(pivot_df, cmap='YlOrRd', annot=False, fmt='.1f',
                   cbar_kws={'label': 'Pharmacophore Score'})
        plt.xlabel('Query Molecule', fontsize=12)
        plt.ylabel('Target Protein', fontsize=12)
        plt.title('Query-Target Score Heatmap (Top 30 Targets)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        heatmap_path = output_dir / 'query_target_heatmap.png'
        plt.savefig(heatmap_path, dpi=300)
        print(f"  Heatmap saved: {heatmap_path}")
        plt.close()
    
    # Save detailed report
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("PharmacoNet Reverse Screening Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input file: {args.results_csv}\n")
        f.write(f"Score threshold: {args.score_threshold}\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total results: {len(df)}\n")
        f.write(f"Unique queries: {df['query_name'].nunique()}\n")
        f.write(f"Unique targets: {df['pharmacophore_model'].nunique()}\n")
        f.write(f"Strong hits (>= {args.score_threshold}): {len(strong_hits)}\n\n")
        
        f.write(query_stats.to_string())
        f.write("\n\n")
        
        f.write("TOP 10 TARGETS PER QUERY\n")
        f.write("-" * 80 + "\n")
        for query_name in df['query_name'].unique():
            query_df = df[df['query_name'] == query_name].nlargest(10, 'score')
            f.write(f"\n{query_name}:\n")
            for idx, (_, row) in enumerate(query_df.iterrows(), start=1):
                model_name = Path(row['pharmacophore_model']).stem
                f.write(f"  {idx:2d}. {model_name:50s} {row['score']:8.4f}\n")
    
    print(f"  Analysis report saved: {report_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
