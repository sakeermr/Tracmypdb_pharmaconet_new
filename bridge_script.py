import pandas as pd
import os

def main():
    # 1. Identify the hits from the SSC similarity search
    ssc_hits = "output/ssc_screening_results_top5_targets.csv"
    
    if not os.path.exists(ssc_hits):
        print("❌ Error: SSC hits file not found.")
        return

    df = pd.read_csv(ssc_hits)
    
    # 2. Extract unique PDB IDs for the Top 5 matches per chemical
    pdb_ids = []
    for _, row in df.iterrows():
        targets = [t.strip() for t in str(row['Top PDB IDs']).split(',')]
        pdb_ids.extend(targets[:5])
    
    unique_pdbs = list(set(pdb_ids))
    
    # 3. Create the input file for PharmacoNet Modeling
    # Format: pdb_id
    pd.DataFrame({'pdb_id': unique_pdbs}).to_csv('input/pdb_database.csv', index=False)
    
    # 4. Create the input file for PharmacoNet Screening
    # Format: name,smiles
    df[['Chemical Name', 'Molecular Structure']].rename(
        columns={'Chemical Name': 'name', 'Molecular Structure': 'smiles'}
    ).to_csv('input/query_molecules.csv', index=False)

    print(f"✅ Bridge complete: Prepared {len(unique_pdbs)} targets for 3D verification.")

if __name__ == "__main__":
    main()
