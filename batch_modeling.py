#!/usr/bin/env python3
"""
Batch Pharmacophore Modeling Script
Reads CSV input from input/ directory and processes all entries automatically
Calls the original modeling.py to ensure identical behavior
Outputs to output/ directory with organized structure
"""
import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path


class BatchModeling_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__("Batch Pharmacophore Modeling Wrapper")
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        self.add_argument(
            "--input_csv",
            type=str,
            default="input/pdb_list.csv",
            help="Path to input CSV file (default: input/pdb_list.csv)"
        )
        self.add_argument(
            "--output_dir",
            type=str,
            default="output",
            help="Output directory (default: output/)"
        )
        self.add_argument(
            "--weight_path",
            type=str,
            help="(Optional) custom PharmacoNet weight path"
        )
        self.add_argument(
            "--cuda",
            action="store_true",
            help="Use GPU acceleration with CUDA"
        )
        self.add_argument(
            "--force",
            action="store_true",
            help="Force overwrite existing pharmacophore models"
        )
        self.add_argument(
            "--continue_on_error",
            action="store_true",
            help="Continue processing even if one entry fails"
        )
        self.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Verbose logging"
        )
        self.add_argument(
            "--suffix",
            choices=("pm", "json"),
            type=str,
            default="pm",
            help="Extension of pharmacophore model (pm (default) | json)"
        )
        self.add_argument(
            "--dry_run",
            action="store_true",
            help="Show commands that would be executed without running them"
        )


def validate_csv_structure(csv_path: Path) -> bool:
    """Validate CSV file has required columns"""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            if not headers:
                logging.error("CSV file is empty or has no headers")
                return False
            
            # Normalize headers (case-insensitive)
            headers_lower = [h.lower().strip() for h in headers]
            
            # Check for required columns
            if 'pdb_code' not in headers_lower and 'pdb' not in headers_lower:
                logging.error("CSV must have 'PDB_code' or 'PDB' column")
                logging.error(f"Found columns: {', '.join(headers)}")
                return False
            
            return True
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return False


def parse_csv_input(csv_path: Path):
    """Parse CSV file and return list of entries"""
    entries = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
            # Skip empty rows
            if not any(row.values()):
                continue
            
            # Normalize keys
            normalized_row = {k.lower().strip(): v.strip() if v else None for k, v in row.items()}
            
            # Get PDB code (try different variations)
            pdb_code = (
                normalized_row.get('pdb_code') or 
                normalized_row.get('pdb') or 
                normalized_row.get('pdbcode')
            )
            
            if not pdb_code:
                logging.warning(f"Row {row_num}: Missing PDB code, skipping")
                continue
            
            # Get ligand ID (optional)
            ligand_id = (
                normalized_row.get('ligand_id') or 
                normalized_row.get('ligand') or 
                normalized_row.get('ligandid')
            )
            
            # Get chain (optional)
            chain = normalized_row.get('chain')
            
            entry = {
                'pdb_code': pdb_code.upper(),
                'ligand_id': ligand_id.upper() if ligand_id else None,
                'chain': chain.upper() if chain else None,
                'row_num': row_num
            }
            
            entries.append(entry)
    
    return entries


def build_modeling_command(entry: dict, output_dir: Path, args) -> list:
    """Build command to call original modeling.py"""
    cmd = [sys.executable, "modeling.py"]
    
    # PDB code (required)
    cmd.extend(["--pdb", entry['pdb_code']])
    
    # Ligand ID filter (optional)
    if entry['ligand_id']:
        cmd.extend(["--ligand_id", entry['ligand_id']])
    
    # Chain filter (optional)
    if entry['chain']:
        cmd.extend(["--chain", entry['chain']])
    
    # Output directory (specific subdirectory for this PDB)
    pdb_output_dir = output_dir / entry['pdb_code']
    cmd.extend(["--out_dir", str(pdb_output_dir)])
    
    # Model format
    cmd.extend(["--suffix", args.suffix])
    
    # Global options
    if args.cuda:
        cmd.append("--cuda")
    if args.force:
        cmd.append("--force")
    if args.verbose:
        cmd.append("--verbose")
    if args.weight_path:
        cmd.extend(["--weight_path", args.weight_path])
    
    # For batch processing, use --all flag if no specific ligand_id is provided
    # This avoids interactive prompts in non-interactive environments (CI/CD)
    if not entry['ligand_id'] and not entry['chain']:
        cmd.append("--all")
    
    return cmd


def main(args):
    """Main batch processing function"""
    
    # Setup paths
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    
    # Validate input CSV exists
    if not input_csv.exists():
        logging.error(f"Input CSV not found: {input_csv}")
        logging.error("Please create input/pdb_list.csv with PDB_code,Ligand_ID columns")
        sys.exit(1)
    
    # Validate CSV structure
    if not validate_csv_structure(input_csv):
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir.absolute()}")
    
    # Parse CSV input
    logging.info(f"Reading input from: {input_csv.absolute()}")
    entries = parse_csv_input(input_csv)
    
    if not entries:
        logging.error("No valid entries found in CSV file")
        sys.exit(1)
    
    logging.info(f"Found {len(entries)} entries to process\n")
    
    # Process each entry by calling modeling.py
    success_count = 0
    failed_entries = []
    
    for i, entry in enumerate(entries, 1):
        pdb_code = entry['pdb_code']
        ligand_id = entry['ligand_id']
        chain = entry['chain']
        
        # Build description
        desc_parts = [pdb_code]
        if ligand_id:
            desc_parts.append(f"Ligand={ligand_id}")
        if chain:
            desc_parts.append(f"Chain={chain}")
        description = ", ".join(desc_parts)
        
        logging.info(f"\n{'='*70}")
        logging.info(f"[{i}/{len(entries)}] Processing: {description}")
        logging.info(f"{'='*70}")
        
        # Build command
        cmd = build_modeling_command(entry, output_dir, args)
        
        # Log command
        cmd_str = ' '.join(cmd)
        logging.info(f"Command: {cmd_str}\n")
        
        if args.dry_run:
            logging.info("[DRY RUN] Command would be executed\n")
            success_count += 1
            continue
        
        # Execute modeling.py
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show real-time output
                text=True
            )
            
            success_count += 1
            logging.info(f"\n✓ Completed: {description}\n")
                
        except subprocess.CalledProcessError as e:
            logging.error(f"\n✗ Failed: {description}")
            logging.error(f"Exit code: {e.returncode}\n")
            failed_entries.append(description)
            
            if not args.continue_on_error:
                logging.error("Stopping due to error (use --continue_on_error to continue)")
                sys.exit(1)
                
        except Exception as e:
            logging.error(f"\n✗ Unexpected error for {description}: {e}\n")
            failed_entries.append(description)
            
            if not args.continue_on_error:
                logging.error("Stopping due to error")
                sys.exit(1)
    
    # Print summary
    logging.info(f"\n{'='*70}")
    logging.info("BATCH PROCESSING SUMMARY")
    logging.info(f"{'='*70}")
    logging.info(f"Total entries: {len(entries)}")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {len(failed_entries)}")
    
    if failed_entries:
        logging.info("\nFailed entries:")
        for entry in failed_entries:
            logging.info(f"  - {entry}")
        logging.info(f"\nResults saved in: {output_dir.absolute()}")
        sys.exit(1)
    else:
        logging.info("\n✓ All entries processed successfully!")
        logging.info(f"Results saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    parser = BatchModeling_ArgParser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    main(args)
