#!/usr/bin/env python3
"""
PharmacoNet Reverse Screening (Target Fishing)
===============================================

Screen query molecules against a database of protein pharmacophore models
to identify potential binding targets.

Author: PharmacoNet Team
License: MIT
"""

import argparse
import csv
import multiprocessing
import sys
from functools import partial
from pathlib import Path

from pmnet.pharmacophore_model import PharmacophoreModel


class ReverseScreening_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            "reverse screening - target fishing",
            description="Screen query molecules against protein pharmacophore database",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        
        cfg_args = self.add_argument_group("config")
        cfg_args.add_argument(
            "-q",
            "--query_molecule",
            type=str,
            help="single query: SMILES string or molecule file (.sdf, .mol2, .pdb)",
        )
        cfg_args.add_argument(
            "--query_csv",
            type=str,
            help="batch query: CSV file with columns 'Name,SMILES' or 'Name,File'",
        )
        cfg_args.add_argument(
            "-d",
            "--model_database_dir",
            type=str,
            help="directory containing pharmacophore model database (.pm files)",
            required=True,
        )
        cfg_args.add_argument(
            "-o", 
            "--out", 
            type=str, 
            help="result file path (CSV)", 
            required=True
        )
        cfg_args.add_argument(
            "--cpus", 
            type=int, 
            help="number of cpus for parallel processing", 
            default=1
        )
        cfg_args.add_argument(
            "--num_conformers",
            type=int,
            help="number of conformers to generate for SMILES queries",
            default=50,
        )
        cfg_args.add_argument(
            "--top_n",
            type=int,
            help="return only top N results per query (default: all)",
            default=None,
        )
        cfg_args.add_argument(
            "--min_score",
            type=float,
            help="minimum score threshold for filtering results",
            default=0.0,
        )

        param_args = self.add_argument_group("pharmacophore feature weights")
        param_args.add_argument(
            "--hydrophobic",
            type=float,
            help="weight for hydrophobic features",
            default=1.0,
        )
        param_args.add_argument(
            "--aromatic", 
            type=float, 
            help="weight for aromatic ring features", 
            default=4.0
        )
        param_args.add_argument(
            "--hba", 
            type=float, 
            help="weight for H-bond acceptor features", 
            default=4.0
        )
        param_args.add_argument(
            "--hbd", 
            type=float, 
            help="weight for H-bond donor features", 
            default=4.0
        )
        param_args.add_argument(
            "--halogen", 
            type=float, 
            help="weight for halogen bond features", 
            default=4.0
        )
        param_args.add_argument(
            "--anion", 
            type=float, 
            help="weight for anionic features", 
            default=8.0
        )
        param_args.add_argument(
            "--cation", 
            type=float, 
            help="weight for cationic features", 
            default=8.0
        )


def is_smiles(query_str):
    """Check if query is a file path or SMILES string"""
    return not Path(query_str).exists()


def score_model(model_path, query_molecule, weight, is_smiles_query, num_conformers):
    """Score a single pharmacophore model against query molecule
    
    Args:
        model_path: Path to .pm pharmacophore model file
        query_molecule: SMILES string or file path
        weight: Dictionary of feature weights
        is_smiles_query: Boolean indicating if query is SMILES
        num_conformers: Number of conformers to generate
        
    Returns:
        Tuple of (model_path, score)
    """
    try:
        model = PharmacophoreModel.load(str(model_path))
        if is_smiles_query:
            score = model.scoring_smiles(query_molecule, num_conformers, weight)
        else:
            score = model.scoring_file(query_molecule, weight)
        return str(model_path), score
    except Exception as e:
        print(f"Error processing {model_path}: {e}", file=sys.stderr)
        return str(model_path), -1.0


def parse_query_csv(csv_path):
    """Parse CSV with query molecules
    
    Expected format:
    Name,SMILES
    CBN,C1=CC=C2C(=C1)...
    THC,CC(C)C1=CC2=C(...
    
    OR:
    Name,File
    CBN,molecules/CBN.sdf
    THC,molecules/THC.sdf
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of dictionaries with keys: name, query, is_smiles
    """
    queries = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Name', row.get('name', ''))
            smiles = row.get('SMILES', row.get('smiles', ''))
            file_path = row.get('File', row.get('file', ''))
            
            if not name:
                print(f"Warning: Skipping row with missing name: {row}", file=sys.stderr)
                continue
            
            if smiles:
                queries.append({'name': name, 'query': smiles, 'is_smiles': True})
            elif file_path:
                if not Path(file_path).exists():
                    print(f"Warning: File not found for {name}: {file_path}", file=sys.stderr)
                    continue
                queries.append({'name': name, 'query': file_path, 'is_smiles': False})
            else:
                print(f"Warning: Skipping {name} - no SMILES or File provided", file=sys.stderr)
    
    return queries


def main():
    parser = ReverseScreening_ArgParser()
    args = parser.parse_args()

    # Validate input
    if not args.query_molecule and not args.query_csv:
        parser.error("Must provide either --query_molecule or --query_csv")
    if args.query_molecule and args.query_csv:
        parser.error("Cannot use both --query_molecule and --query_csv")

    # Parse queries
    if args.query_csv:
        print(f"Loading queries from CSV: {args.query_csv}")
        queries = parse_query_csv(args.query_csv)
        if not queries:
            print("ERROR: No valid queries found in CSV file!")
            sys.exit(1)
        print(f"Found {len(queries)} query molecules")
    else:
        # Single query
        is_smiles_query = is_smiles(args.query_molecule)
        queries = [{
            'name': 'Query',
            'query': args.query_molecule,
            'is_smiles': is_smiles_query
        }]
        if is_smiles_query:
            print(f"Query molecule (SMILES): {args.query_molecule}")
        else:
            print(f"Query molecule (file): {args.query_molecule}")

    # Find all .pm files in database directory
    database_path = Path(args.model_database_dir)
    if not database_path.exists():
        print(f"ERROR: Database directory not found: {args.model_database_dir}")
        sys.exit(1)
    
    model_list = list(database_path.rglob("*.pm"))
    print(f"Found {len(model_list)} pharmacophore models in database")

    if len(model_list) == 0:
        print("ERROR: No .pm files found in database directory!")
        print(f"Please run batch_modeling.py first to build the database.")
        sys.exit(1)

    # Prepare weights
    weight = dict(
        Cation=args.cation,
        Anion=args.anion,
        Aromatic=args.aromatic,
        HBond_donor=args.hbd,
        HBond_acceptor=args.hba,
        Halogen=args.halogen,
        Hydrophobic=args.hydrophobic,
    )

    print(f"\nFeature weights:")
    for feat, w in weight.items():
        print(f"  {feat:20s}: {w}")

    # Process all queries
    all_results = []
    
    for query_idx, query_info in enumerate(queries, start=1):
        query_name = query_info['name']
        query_molecule = query_info['query']
        is_smiles_query = query_info['is_smiles']
        
        print(f"\n{'='*80}")
        print(f"Processing query {query_idx}/{len(queries)}: {query_name}")
        print(f"{'='*80}")
        
        # Score query molecule against all models
        print(f"Screening against {len(model_list)} protein models...")
        f = partial(
            score_model,
            query_molecule=query_molecule,
            weight=weight,
            is_smiles_query=is_smiles_query,
            num_conformers=args.num_conformers,
        )

        with multiprocessing.Pool(args.cpus) as pool:
            result = pool.map(f, model_list)

        # Filter by minimum score
        result = [(path, score) for path, score in result if score >= args.min_score]

        # Sort by score (descending - highest score first)
        result.sort(key=lambda x: x[1], reverse=True)

        # Apply top_n filter if specified
        if args.top_n is not None:
            result = result[: args.top_n]

        # Add query name to results
        for model_path, score in result:
            all_results.append((query_name, model_path, score))

        # Print top 10 for this query
        print(f"\nTOP 10 MATCHING PROTEINS for {query_name}:")
        print("-" * 80)
        if result:
            for rank, (model_path, score) in enumerate(result[:10], start=1):
                model_name = Path(model_path).stem
                print(f"{rank:3d}. {model_name:50s} Score: {score:8.4f}")
        else:
            print("  No matches found (all scores below threshold)")

    # Save all results
    print(f"\n{'='*80}")
    print(f"Saving {len(all_results)} total results to {args.out}")
    print(f"{'='*80}")
    
    # Create output directory if needed
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.out, "w", newline='') as w:
        writer = csv.writer(w)
        writer.writerow(["query_name", "pharmacophore_model", "score"])
        for query_name, model_path, score in all_results:
            writer.writerow([query_name, model_path, score])

    print("\nReverse screening complete! ✓")
    print(f"Results saved to: {args.out}")
    
    # Summary statistics
    if all_results:
        scores = [score for _, _, score in all_results]
        print(f"\nScore Statistics:")
        print(f"  Total matches: {len(all_results)}")
        print(f"  Max score: {max(scores):.4f}")
        print(f"  Min score: {min(scores):.4f}")
        print(f"  Average score: {sum(scores)/len(scores):.4f}")


if __name__ == "__main__":
    main()
