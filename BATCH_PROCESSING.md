# PharmacoNet Batch Processing

This directory structure enables automated batch processing of pharmacophore modeling.

## Directory Structure

```
PharmacoNet_internal/
├── input/                      # Input CSV files
│   └── pdb_list.csv           # Main input file
├── output/                     # Generated results
│   └── {PDB_CODE}/            # Results organized by PDB code
│       ├── *.pm               # Pharmacophore models
│       └── *.pse              # PyMOL visualizations
├── batch_modeling.py          # Batch processing script
└── .github/
    └── workflows/
        └── batch_modeling.yml # GitHub Actions workflow
```

## Quick Start

### 1. Prepare Input CSV

Edit `input/pdb_list.csv` with your target proteins:

```csv
PDB_code,Ligand_ID,Chain
5XRA,8D3,A
6OIM,MOV,
1HSG,,
3PBL,ATP,B
```

**CSV Format:**
- **PDB_code** (required): RCSB PDB identifier (e.g., 5XRA, 6OIM)
- **Ligand_ID** (optional): Specific ligand to process (e.g., 8D3, ATP)
  - Leave empty to process all ligands
- **Chain** (optional): Specific protein chain (e.g., A, B)
  - Leave empty to process all chains

### 2. Run Locally

```bash
# Basic usage
python batch_modeling.py

# With GPU acceleration
python batch_modeling.py --cuda

# Custom input/output paths
python batch_modeling.py --input_csv input/my_proteins.csv --output_dir results

# Continue on errors
python batch_modeling.py --continue_on_error

# Verbose logging
python batch_modeling.py --verbose
```

### 3. Run via GitHub Actions

1. **Push your CSV file:**
   ```bash
   git add input/pdb_list.csv
   git commit -m "Add proteins to process"
   git push
   ```

2. **Or trigger manually:**
   - Go to **Actions** tab
   - Select **"PharmacoNet Batch Modeling"**
   - Click **"Run workflow"**
   - Configure options (optional)
   - Click **"Run workflow"**

3. **Download results:**
   - Wait for workflow completion
   - Go to workflow run page
   - Download artifacts:
     - `pharmacophore-models` (.pm files)
     - `pymol-visualizations` (.pse files)
     - `complete-output` (all files)

## Command-Line Options

```bash
python batch_modeling.py [OPTIONS]

Options:
  --input_csv PATH              Input CSV file (default: input/pdb_list.csv)
  --output_dir PATH             Output directory (default: output/)
  --cuda                        Use GPU acceleration
  --force                       Overwrite existing files
  --continue_on_error           Continue processing if one entry fails
  --verbose, -v                 Detailed logging
  --suffix {pm,json}            Model format (default: pm)
  --weight_path PATH            Custom PharmacoNet weights
```

## Output Structure

For each PDB entry, the following files are generated:

```
output/
└── 5XRA/                                    # PDB code
    ├── 5XRA_A_8D3_model.pm                 # Pharmacophore model
    └── 5XRA_A_8D3_model_pymol.pse          # PyMOL visualization
```

**File naming convention:**
`{PDB_CODE}_{CHAIN}_{LIGAND_ID}_model.{pm|json}`

## CSV Examples

### Example 1: Simple PDB List (Process All Ligands)
```csv
PDB_code,Ligand_ID,Chain
5XRA,,
6OIM,,
1HSG,,
```

### Example 2: Specific Ligands
```csv
PDB_code,Ligand_ID,Chain
5XRA,8D3,
6OIM,MOV,
3PBL,ATP,
```

### Example 3: Specific Chains
```csv
PDB_code,Ligand_ID,Chain
5XRA,8D3,A
6OIM,MOV,D
1HSG,,A
```

### Example 4: Mixed Configuration
```csv
PDB_code,Ligand_ID,Chain
5XRA,8D3,A
6OIM,,
1HSG,MK1,A
3PBL,ATP,B
4HHB,,
```

## Troubleshooting

### CSV File Issues

**Error: "CSV must have 'PDB_code' column"**
- Ensure first row has header: `PDB_code,Ligand_ID,Chain`
- Column names are case-insensitive
- Alternative names accepted: `PDB`, `pdb_code`, `PDBCODE`

**Error: "No valid entries found"**
- Check that PDB_code column is not empty
- Ensure there are no extra blank lines
- Verify CSV encoding is UTF-8

### Processing Issues

**Error: "Failed to download PDB"**
- Verify PDB code exists in RCSB database
- Check internet connection
- Try different PDB codes

**Error: "No ligands found"**
- PDB structure may not contain ligands
- Try leaving Ligand_ID empty to see all available ligands
- Check chain identifier is correct

**Warning: "No ligands found with ID 'XXX'"**
- Ligand ID may be incorrect
- Leave Ligand_ID empty first to see available ligands
- Check RCSB PDB website for correct ligand identifiers

### GitHub Actions Issues

**Workflow doesn't trigger on push**
- Ensure you're pushing to `main` branch
- Check that `input/pdb_list.csv` was modified
- Try manual trigger instead

**Workflow fails: "CSV file not found"**
- Ensure `input/pdb_list.csv` is committed to repository
- Check file path in workflow trigger
- Try running workflow manually with custom path

## Best Practices

1. **Start Small**: Test with 1-2 proteins first
2. **Use Continue-on-Error**: Add `--continue_on_error` for large batches
3. **Check Ligand IDs**: Visit RCSB PDB to verify correct ligand identifiers
4. **Organize CSVs**: Create separate CSV files for different projects
5. **Version Control**: Commit CSV files to track which proteins were processed
6. **Monitor Logs**: Use `--verbose` for detailed debugging information

## Advanced Usage

### Multiple CSV Files

```bash
# Process different protein sets
python batch_modeling.py --input_csv input/kinases.csv --output_dir output/kinases
python batch_modeling.py --input_csv input/gpcr.csv --output_dir output/gpcr
```

### GPU Acceleration

```bash
# Faster processing with CUDA
python batch_modeling.py --cuda
```

### Custom Model Weights

```bash
# Use custom trained weights
python batch_modeling.py --weight_path /path/to/custom_weights.pt
```

## Support

For issues or questions:
1. Check logs with `--verbose` flag
2. Verify CSV format matches examples
3. Test with known working PDB codes (e.g., 5XRA, 6OIM)
4. Review GitHub Actions logs for CI/CD issues

## Citation

If you use this tool, please cite:
```
Seo, S., & Kim, W. Y. (2024). 
PharmacoNet: Protein-based pharmacophore modeling for ultra-large-scale virtual screening.
Chemical Science.
```
