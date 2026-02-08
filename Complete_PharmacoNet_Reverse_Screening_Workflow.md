# Complete PharmacoNet Reverse Screening Workflow

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Build Protein Pharmacophore Database](#step-1-build-protein-pharmacophore-database)
4. [Step 2: Prepare Query Molecules](#step-2-prepare-query-molecules)
5. [Step 3: Run Reverse Screening (Target Fishing)](#step-3-run-reverse-screening-target-fishing)
6. [Understanding the Results](#understanding-the-results)
7. [Technical Deep Dive](#technical-deep-dive)
8. [Comparison with Other Methods](#comparison-with-other-methods)
9. [Best Practices & Tips](#best-practices--tips)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### What is Reverse Screening?

**Traditional Screening (Forward):**
```
Protein → Screen library of molecules → Find binders
```

**Reverse Screening (Target Fishing):**
```
Molecule → Screen database of proteins → Find targets
```

### Use Cases

- **Drug Repurposing:** Find new targets for existing drugs
- **Target Identification:** Discover biological targets for natural products (e.g., CBN, THC)
- **Off-Target Prediction:** Identify potential side effects by finding unexpected protein matches
- **Polypharmacology:** Understand multi-target binding profiles

### PharmacoNet Approach

PharmacoNet combines:
- ✅ **Deep Learning:** 3D Swin Transformer neural network trained on 10,000+ protein-ligand complexes
- ✅ **Pharmacophore Modeling:** Extract chemical feature patterns (H-bonds, hydrophobic, aromatic, charges)
- ✅ **Graph Matching:** Flexible 3D pattern matching with conformer ensemble
- ✅ **Data-Driven:** Learn from real binding data, not hand-crafted rules

**Result:** More accurate than traditional methods like PharmMapper (~20-30% better enrichment)

---

## Prerequisites

### System Requirements

- **Operating System:** Linux, macOS, or Windows
- **Python:** 3.11+
- **RAM:** 8GB minimum, 16GB+ recommended
- **GPU:** Optional but highly recommended (CUDA-compatible NVIDIA GPU)
  - With GPU: ~30s per protein
  - Without GPU: ~3-5 min per protein

### Software Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate pmnet

# Verify installation
python -c "import pmnet; print(pmnet.__version__)"
# Expected: 2.2.0
```

### Required Scripts

Ensure you have these files:
- `modeling.py` - Original pharmacophore modeling script
- `batch_modeling.py` - Batch wrapper for multiple proteins
- `reverse_screening.py` - Target fishing script (main tool)
- `screening.py` - Forward screening (for validation)

---

## Step 1: Build Protein Pharmacophore Database

### 1.1 Prepare Input CSV

Create `input/pdb_database.csv` with protein structures:

```csv
PDB_code,Ligand_ID,Chain
5XRA,8D3,A
6LU7,N3J,A
1ATP,ATP,
2BRC,BRC,B
3COX,ASA,A
4ABC,LIG,
... (add 1000+ entries)
```

**Column Descriptions:**
- `PDB_code`: RCSB PDB identifier (e.g., 5XRA)
- `Ligand_ID`: Co-crystallized ligand code (optional, leave empty for auto-detection)
- `Chain`: Protein chain (optional, leave empty for auto-detection)

**Where to Find PDB Codes:**
- [RCSB PDB](https://www.rcsb.org/)
- Search by protein name, disease, or drug class
- Filter for structures with ligands (resolution < 3Å recommended)

### 1.2 Run Batch Modeling

**Basic Command:**
```bash
python batch_modeling.py \
  --input_csv input/pdb_database.csv \
  --output_dir output
```

**With GPU Acceleration:**
```bash
python batch_modeling.py \
  --input_csv input/pdb_database.csv \
  --output_dir output \
  --cuda
```

**Advanced Options:**
```bash
python batch_modeling.py \
  --input_csv input/pdb_database.csv \
  --output_dir output \
  --cuda \
  --force              # Overwrite existing models
  --verbose            # Detailed logging
```

### 1.3 What Happens Under the Hood

```
For EACH protein (e.g., 5XRA):

┌─────────────────────────────────────────────────────────┐
│ 1. DOWNLOAD & PARSE                                     │
└─────────────────────────────────────────────────────────┘
   ├─ Download 5XRA.pdb from RCSB PDB database
   ├─ Parse PDB file → Extract 3D atomic coordinates
   ├─ Identify chains, residues, atoms
   └─ Result: Protein structure in memory

┌─────────────────────────────────────────────────────────┐
│ 2. LIGAND DETECTION & BINDING SITE IDENTIFICATION      │
└─────────────────────────────────────────────────────────┘
   ├─ Search for ligand "8D3" in chain A
   ├─ If not found: Auto-detect all ligands
   ├─ Extract ligand atoms
   ├─ Calculate geometric center: (x, y, z)
   └─ Result: Binding site center coordinates

┌─────────────────────────────────────────────────────────┐
│ 3. VOXELIZATION (3D Tensor Creation)                   │
└─────────────────────────────────────────────────────────┘
   ├─ Create 64×64×64 grid centered at binding site
   ├─ Grid size: 32 Å × 32 Å × 32 Å (0.5Å resolution)
   ├─ For each voxel, encode:
   │   ├─ Atom type (C, N, O, S, etc.)
   │   ├─ Aromaticity (aromatic ring membership)
   │   ├─ Partial charge (electrostatics)
   │   ├─ Hydrophobicity (lipophilicity)
   │   ├─ H-bond donor/acceptor potential
   │   └─ Distance to protein surface
   └─ Result: [64, 64, 64, C] tensor (C ≈ 10-20 channels)

┌─────────────────────────────────────────────────────────┐
│ 4. NEURAL NETWORK INFERENCE                             │
└─────────────────────────────────────────────────────────┘
   Architecture: 3D Swin Transformer
   
   Input: [64, 64, 64, C] voxelized protein
      ↓
   ┌──────────────────────────────────────┐
   │ 3D Convolutional Layers              │
   │ - Extract local geometric patterns    │
   │ - Detect pockets, grooves, clefts    │
   └──────────────────────────────────────┘
      ↓
   ┌──────────────────────────────────────┐
   │ 3D Swin Transformer Blocks           │
   │ - Self-attention: Long-range context │
   │ - Learn: "Cation near aromatic"      │
   │ - Multi-scale feature fusion         │
   └──────────────────────────────────────┘
      ↓
   ┌──────────────────────────────────────┐
   │ Detection Heads (Per Feature Type)   │
   │ - Hydrophobic head → Carbon hotspots │
   │ - Aromatic head → Ring hotspots      │
   │ - HBond head → Donor/acceptor spots  │
   │ - Charge head → Cation/anion spots   │
   │ - Halogen head → Halogen bond spots  │
   └──────────────────────────────────────┘
      ↓
   Output: ~100-200 pharmacophore hotspots
   Format: [(x, y, z, type, confidence), ...]
   
   Example Output:
   [
     (15.3, 8.2, -2.1, 'Aromatic', 0.92),
     (18.1, 6.5, -1.3, 'HBond_acceptor', 0.87),
     (12.7, 10.4, 0.8, 'Cation', 0.95),
     (16.2, 7.8, -3.5, 'Hydrophobic', 0.78),
     ...
   ]
   
   Time: ~10-30 seconds (GPU) or ~3-5 minutes (CPU)

┌─────────────────────────────────────────────────────────┐
│ 5. POST-PROCESSING                                      │
└─────────────────────────────────────────────────────────┘
   ├─ Filter: Remove low-confidence hotspots (< 0.5)
   ├─ Cluster: Group nearby hotspots of same type
   │   Example: 5 aromatic hotspots within 2Å → 1 cluster
   ├─ Build spatial graph:
   │   ├─ Nodes: Hotspot clusters
   │   ├─ Edges: Pairwise distances
   │   └─ Store: Distance matrix [N×N]
   └─ Result: PharmacophoreModel object

┌─────────────────────────────────────────────────────────┐
│ 6. SAVE MODEL                                           │
└─────────────────────────────────────────────────────────┘
   ├─ Serialize PharmacophoreModel → Python pickle
   ├─ Save as: output/5XRA/5XRA_A_8D3_model.pm
   ├─ File size: ~50-500 KB
   └─ Contains:
       ├─ Hotspot coordinates [(x,y,z), ...]
       ├─ Hotspot types ['Aromatic', 'Cation', ...]
       ├─ Spatial graph (distance matrix)
       ├─ Metadata (PDB ID, resolution, binding site)
       └─ Clustering information

┌─────────────────────────────────────────────────────────┐
│ 7. VISUALIZATION (Optional)                             │
└─────────────────────────────────────────────────────────┘
   ├─ Generate PyMOL session file
   ├─ Show: Protein surface + pharmacophore spheres
   ├─ Color code:
   │   ├─ Red: Cation (+)
   │   ├─ Blue: Anion (-)
   │   ├─ Orange: Aromatic
   │   ├─ Green: HBond donor
   │   ├─ Cyan: HBond acceptor
   │   └─ Gray: Hydrophobic
   └─ Save as: output/5XRA/5XRA_A_8D3_model_pymol.pse
```

### 1.4 Expected Output Structure

```
output/
├── 5XRA/
│   ├── 5XRA.pdb                      # Downloaded protein structure
│   ├── 5XRA_A_8D3.sdf                # Extracted ligand
│   ├── 5XRA_A_8D3_model.pm          # ← MAIN OUTPUT: Pharmacophore model
│   └── 5XRA_A_8D3_model_pymol.pse   # Visualization (optional)
│
├── 6LU7/
│   ├── 6LU7.pdb
│   ├── 6LU7_A_N3J.sdf
│   ├── 6LU7_A_N3J_model.pm          # ← Pharmacophore model
│   └── 6LU7_A_N3J_model_pymol.pse
│
└── ... (1000+ protein folders)
```

### 1.5 Quality Control

**Check Progress:**
```bash
# Count completed models
find output -name "*.pm" | wc -l

# Check for errors in log
grep "ERROR" batch_modeling.log

# Verify model files are not empty
find output -name "*.pm" -size 0
```

**Expected Time:**
- **1000 proteins with GPU:** ~8-10 hours
- **1000 proteins with CPU:** ~3-4 days
- **Can be parallelized** on cluster/cloud

---

## Step 2: Prepare Query Molecules

### 2.1 Input Format Options

#### Option A: CSV with SMILES (Recommended)

Create `input/query_molecules.csv`:

```csv
Name,SMILES
Cannabinol,C1=CC=C2C(=C1)C3=C(C=C(C=C3)O)C(O2)(C)C
THC,CCCCCc1cc(O)c2c(c1)OC(C)(C)C1CCC(C)=CC21
Aspirin,CC(=O)Oc1ccccc1C(=O)O
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O
Morphine,CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5
Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Nicotine,CN1CCC[C@H]1c2cccnc2
```

#### Option B: CSV with Structure Files

```csv
Name,File
Cannabinol,molecules/cannabinol.sdf
THC,molecules/thc.mol2
Aspirin,molecules/aspirin.pdb
```

#### Option C: Single SMILES (Quick Test)

```bash
# No CSV needed, provide SMILES directly in command
python reverse_screening.py \
  --query_molecule "CC(=O)Oc1ccccc1C(=O)O" \
  --model_database_dir output \
  --out results.csv
```

### 2.2 Where to Get SMILES

**Databases:**
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) - Search by name, download SMILES
- [ChEMBL](https://www.ebi.ac.uk/chembl/) - Bioactive molecules
- [ZINC](https://zinc.docking.org/) - Commercial compounds
- [DrugBank](https://go.drugbank.com/) - FDA-approved drugs

**Tools:**
- ChemDraw → Export as SMILES
- RDKit: `Chem.MolToSmiles(mol)`
- OpenBabel: `obabel input.sdf -osmi`

---

## Step 3: Run Reverse Screening (Target Fishing)

### 3.1 Basic Command

```bash
python reverse_screening.py \
  --query_csv input/query_molecules.csv \
  --model_database_dir output \
  --out results_target_fishing.csv
```

### 3.2 Advanced Options

```bash
python reverse_screening.py \
  --query_csv input/query_molecules.csv \
  --model_database_dir output \
  --out results_target_fishing.csv \
  --num_conformers 100 \        # Generate 100 conformers (default: 50)
  --cpus 8 \                    # Use 8 CPU cores for parallelization
  --top_n 50 \                  # Return top 50 matches per query
  --min_score 5.0 \             # Filter out scores below 5.0
  --hydrophobic 1.0 \           # Weight for hydrophobic features
  --aromatic 4.0 \              # Weight for aromatic features
  --hba 4.0 \                   # Weight for H-bond acceptors
  --hbd 4.0 \                   # Weight for H-bond donors
  --cation 8.0 \                # Weight for cationic features
  --anion 8.0 \                 # Weight for anionic features
  --halogen 4.0                 # Weight for halogen bonds
```

### 3.3 What Happens Under the Hood

```
═══════════════════════════════════════════════════════════════
FOR EACH QUERY MOLECULE (e.g., Cannabinol)
═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│ STEP 1: PARSE SMILES                                        │
└─────────────────────────────────────────────────────────────┘

Input: "C1=CC=C2C(=C1)C3=C(C=C(C=C3)O)C(O2)(C)C"
   ↓
RDKit: SMILES string → Mol object
   ↓
Add hydrogens (explicit H)
   ↓
Result: 3D-ready molecular graph
Time: ~0.001 seconds

┌─────────────────────────────────────────────────────────────┐
│ STEP 2: CONFORMER GENERATION                                │
└─────────────────────────────────────────────────────────────┘

Algorithm: ETKDG v3 (Extended Torsion-angle Distance Geometry)

Process:
1. Generate 100 random initial 3D structures
   - Use distance geometry to satisfy:
     ├─ Bond lengths (1.2-1.5 Å for C-C)
     ├─ Bond angles (109.5° for sp³, 120° for sp²)
     └─ Torsion angles (sample from distributions)

2. Refine with force field minimization
   - MMFF94 force field
   - Energy minimization (gradient descent)
   - Converge to local minimum

3. Remove duplicates
   - Calculate RMSD between conformers
   - If RMSD < 0.5 Å → Consider duplicate
   - Keep ~50-100 unique conformers

Output: [N_conformers, N_atoms, 3] coordinate array

Example for Cannabinol (21 atoms, 50 conformers):
conformers = [
  [[x1, y1, z1], [x2, y2, z2], ..., [x21, y21, z21]],  # Conf 1
  [[x1, y1, z1], [x2, y2, z2], ..., [x21, y21, z21]],  # Conf 2
  ...
  [[x1, y1, z1], [x2, y2, z2], ..., [x21, y21, z21]],  # Conf 50
]

Time: ~2-5 seconds

┌─────────────────────────────────────────────────────────────┐
│ STEP 3: PHARMACOPHORE FEATURE EXTRACTION                    │
└─────────────────────────────────────────────────────────────┘

Rule-based chemical pattern matching:

1. HYDROPHOBIC
   ├─ Pattern: sp³ carbons with 4 single bonds
   ├─ Exclude: Carbons near heteroatoms (N, O, S)
   └─ Example: CH₃, CH₂ groups in aliphatic chains

2. AROMATIC
   ├─ Pattern: Atoms in aromatic rings
   ├─ Detection: RDKit aromatic flag
   ├─ Center: Geometric center of ring atoms
   └─ Example: Benzene ring, pyridine

3. HBOND DONOR
   ├─ Pattern: N-H, O-H bonds
   ├─ Check: Hydrogen attached to N or O
   ├─ Directionality: H → acceptor
   └─ Example: Phenolic OH, amine NH

4. HBOND ACCEPTOR
   ├─ Pattern: N, O with lone pairs
   ├─ Check: Not protonated
   ├─ Directionality: Donor → lone pair
   └─ Example: Carbonyl C=O, ether O, pyridine N

5. CATION
   ├─ Pattern: Protonated amines
   ├─ pH: Assume physiological pH 7.4
   ├─ Rules: pKa > 7.4 → Protonated
   └─ Example: -NH₃⁺, -NH₂R⁺, guanidinium

6. ANION
   ├─ Pattern: Deprotonated acids
   ├─ pH: Assume physiological pH 7.4
   ├─ Rules: pKa < 7.4 → Deprotonated
   └─ Example: Carboxylate COO⁻, phosphate PO₄²⁻

7. HALOGEN
   ├─ Pattern: F, Cl, Br, I atoms
   ├─ Halogen bonding: σ-hole interaction
   └─ Example: Cl in chloramphenicol

Output: Feature list with atom indices

Example for Cannabinol:
features = [
  (Aromatic, center_atoms=[5,6,7,8,9,10]),      # Benzene ring 1
  (Aromatic, center_atoms=[11,12,13,14,15,16]), # Benzene ring 2
  (HBond_acceptor, atoms=[17]),                  # Oxygen in ether
  (HBond_donor, atoms=[18,19]),                  # Phenolic OH
  (Hydrophobic, atoms=[2,3,4]),                  # Methyl groups
]

Time: ~0.01 seconds (very fast!)

┌─────────────────────────────────────────────────────────────┐
│ STEP 4: BUILD LIGAND GRAPH                                  │
└─────────────────────────────────────────────────────────────┘

Create graph representation:
- Nodes: Pharmacophore features
- Positions: Store coordinates for ALL conformers

Example structure:
ligand_graph = {
  'nodes': [
    {
      'type': 'Aromatic',
      'atom_indices': [5,6,7,8,9,10],
      'positions': [
        [2.5, 1.2, 0.3],   # Conformer 1
        [2.1, 1.5, -0.2],  # Conformer 2
        ...
        [2.3, 1.1, 0.1],   # Conformer 50
      ]
    },
    {
      'type': 'HBond_donor',
      'atom_indices': [18,19],
      'positions': [
        [4.2, 3.1, 1.5],   # Conformer 1
        [4.5, 2.9, 1.3],   # Conformer 2
        ...
      ]
    },
    ...
  ]
}

Note: Features extracted ONCE, but positions vary per conformer!

═══════════════════════════════════════════════════════════════
SCREEN AGAINST ALL PROTEINS IN DATABASE
═══════════════════════════════════════════════════════════════

For protein in [5XRA, 6LU7, 1ATP, ...]:  # 1000+ proteins

┌─────────────────────────────────────────────────────────────┐
│ STEP 5: LOAD PROTEIN PHARMACOPHORE MODEL                    │
└─────────────────────────────────────────────────────────────┘

Load: output/5XRA/5XRA_A_8D3_model.pm
   ↓
Deserialize: pickle → PharmacophoreModel object
   ↓
Contains:
├─ Hotspots: [(x,y,z,type,confidence), ...]  (~100-200)
├─ Clusters: Grouped by type
│   ├─ Aromatic: [cluster1, cluster2, ...]
│   ├─ Cation: [cluster1, cluster2, ...]
│   ├─ HBond: [cluster1, cluster2, ...]
│   └─ ...
├─ Spatial graph: Distance matrix [N×N]
└─ Metadata: PDB info, resolution, binding site center

┌─────────────────────────────────────────────────────────────┐
│ STEP 6: GRAPH MATCHING ALGORITHM                            │
└─────────────────────────────────────────────────────────────┘

Goal: Find optimal mapping of ligand features to protein hotspots

╔═════════════════════════════════════════════════════════════╗
║ A. CLUSTER COMPATIBILITY FILTERING                          ║
╚═════════════════════════════════════════════════════════════╝

For each ligand feature:
├─ Find compatible protein hotspot clusters (same type)
├─ Example:
│   Ligand Aromatic → Protein Aromatic clusters [C1, C2, C3]
│   Ligand Cation → Protein Cation clusters [C4, C5]
│
└─ Priority ranking (most important features first):
    1. Cation (weight=8)    ← Electrostatic = strongest
    2. Anion (weight=8)     ← Electrostatic = strongest
    3. Aromatic (weight=4)  ← π-π stacking
    4. HBond (weight=4)     ← Polar interactions
    5. Halogen (weight=4)   ← σ-hole bonding
    6. Hydrophobic (weight=1) ← Weakest

Select top 20 ligand features (computational limit)

╔═════════════════════════════════════════════════════════════╗
║ B. TREE SEARCH (EXPLORE ALL POSSIBLE ASSIGNMENTS)          ║
╚═════════════════════════════════════════════════════════════╝

Build tree of feature assignments:

                            ROOT
                    (No assignments)
                            |
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    Feature1 → P-ClusterA   Feature1 → P-ClusterB   Feature1 → P-ClusterC
    (Aromatic)              (Aromatic)              (Aromatic)
        |                   |                   |
    ┌───┴───┐           ┌───┴───┐           ┌───┴───┐
    │       │           │       │           │       │
Feature2 → P-Cluster1  Feature2 → P-Cluster2  ...
(Cation)               (Cation)
    |                   |
  [Continue branching up to depth 20]
    |                   |
   LEAF                LEAF
(Complete             (Complete
assignment)           assignment)

Pruning strategies:
├─ Geometric impossibility: If distance constraints violated
├─ Low score: Early termination if score can't improve
└─ Result: ~1,000-10,000 valid complete assignments

╔═════════════════════════════════════════════════════════════╗
║ C. SCORE EACH ASSIGNMENT (LEAF NODE)                       ║
╚═════════════════════════════════════════════════════════════╝

For each complete assignment (leaf):

Initialize: scores[N_conformers] = zeros(50)

For conformer_id in range(50):
    
    pair_score = 0
    
    For each (ligand_feature → protein_hotspot) pair:
        
        # Get 3D coordinates
        lig_pos = ligand.positions[feature][conformer_id]
        prot_pos = protein.hotspot.position
        
        # Calculate Euclidean distance
        dist = ||lig_pos - prot_pos||
        
        # Score with Gaussian decay
        if dist < 2.0 Å:
            weight = feature_weights[feature_type]
            # Gaussian: exp(-dist²/(2σ²))
            match_score = weight × exp(-dist² / 2.0)
        else:
            match_score = 0  # Too far, no contribution
        
        pair_score += match_score
    
    # Keep best score for this conformer
    scores[conformer_id] = max(scores[conformer_id], pair_score)

# Final score: Average across all conformers
final_score = mean(scores)

Example scores array:
[12.3, 45.6, 38.2, 42.1, 35.8, ..., 40.2]
         ↑
    Conformer 2 has best fit (45.6)

Final score = (12.3 + 45.6 + ... + 40.2) / 50 ≈ 35.4

╔═════════════════════════════════════════════════════════════╗
║ D. MATHEMATICAL FORMULATION                                 ║
╚═════════════════════════════════════════════════════════════╝

Score = (1/N_conf) × Σ(c=1 to N_conf) max(a ∈ A) S(a, c)

Where:
- N_conf: Number of conformers
- A: Set of all valid feature assignments
- S(a, c): Score for assignment a in conformer c
- S(a, c) = Σ w_i × exp(-d_i² / 2σ²)
  - w_i: Feature weight (1-8)
  - d_i: Distance between matched features
  - σ: Tolerance parameter (1.0 Å)

Result for this protein: (5XRA_model.pm, score=35.4)

Time per protein: ~0.1-0.2 seconds
Total time for 1000 proteins: ~2-3 minutes (with 8 CPUs)

═══════════════════════════════════════════════════════════════
COLLECT & RANK ALL RESULTS
═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│ STEP 7: RANKING & FILTERING                                 │
└─────────────────────────────────────────────────────────────┘

Collect scores for all proteins:
[
  (6LU7_A_N3J_model.pm, 42.1),
  (5XRA_A_8D3_model.pm, 35.4),
  (2BRC_B_BRC_model.pm, 28.7),
  (1ATP_ATP_model.pm, 24.3),
  (4XYZ_A_LIG_model.pm, 3.2),  ← Below threshold
  ...
]
   ↓
Sort descending (highest score first)
   ↓
Filter: score >= min_score (5.0)
   ↓
Take top N (50)
   ↓
Output for this query molecule
```

### 3.4 Output Format

**CSV File: `results_target_fishing.csv`**
```csv
query_name,pharmacophore_model,score
Cannabinol,output/6LU7/6LU7_A_N3J_model.pm,42.1
Cannabinol,output/5XRA/5XRA_A_8D3_model.pm,35.4
Cannabinol,output/2BRC/2BRC_B_BRC_model.pm,28.7
Cannabinol,output/1ATP/1ATP_ATP_model.pm,24.3
THC,output/5XRA/5XRA_A_8D3_model.pm,48.3
THC,output/6LU7/6LU7_A_N3J_model.pm,41.2
Aspirin,output/3COX/3COX_A_ASA_model.pm,55.7
...
```

**Console Output:**
```
================================================================================
Processing query: Cannabinol
================================================================================
Found 1000 pharmacophore models in database
Screening query molecule against 1000 protein models...

TOP 10 MATCHING PROTEINS for Cannabinol:
--------------------------------------------------------------------------------
  1. 6LU7_A_N3J_model                       Score: 42.1000
  2. 5XRA_A_8D3_model                       Score: 35.4000
  3. 2BRC_B_BRC_model                       Score: 28.7000
  4. 1ATP_ATP_model                         Score: 24.3000
  5. 4XYZ_A_LIG_model                       Score: 21.8000
  6. 7ABC_C_INH_model                       Score: 19.5000
  7. 3DEF_B_SUB_model                       Score: 17.2000
  8. 8GHI_A_ACT_model                       Score: 15.8000
  9. 2JKL_D_MOL_model                       Score: 14.3000
 10. 5MNO_E_DRG_model                       Score: 12.7000
================================================================================
```

---

## Understanding the Results

### Score Interpretation

#### Score Ranges & Biological Meaning

| Score Range | Interpretation | Action |
|------------|----------------|--------|
| **> 40** | **Strong Match** | High confidence binder. Prioritize for experimental validation |
| **30-40** | **Good Match** | Likely binder. Worth investigating |
| **20-30** | **Moderate Match** | Possible binder. Consider secondary validation |
| **10-20** | **Weak Match** | Low confidence. May be false positive |
| **< 10** | **No Match** | Unlikely to bind |

#### What the Score Represents

**The score is NOT:**
- ❌ Binding affinity (Kd, Ki, IC50)
- ❌ Free energy of binding (ΔG)
- ❌ Percent inhibition

**The score IS:**
- ✅ **Pharmacophore compatibility** - How well the molecule's chemical features align with the protein's binding site
- ✅ **Shape complementarity** - Geometric fit between molecule conformers and hotspot patterns
- ✅ **Weighted feature matching** - Emphasizes important interactions (charges > polarity > hydrophobic)

### Example Interpretation

**Result:**
```
Cannabinol → 6LU7 (COVID-19 Main Protease) → Score: 42.1
```

**What this means:**
1. Cannabinol's pharmacophore features (aromatic rings, hydroxyl groups) align well with 6LU7's binding site hotspots
2. Multiple conformers of Cannabinol can fit into the binding pocket
3. High-weight features (e.g., H-bonds, aromatic stacking) are matched
4. **Prediction:** Cannabinol may bind to/inhibit COVID-19 main protease

**What to do next:**
1. ✅ Experimental validation: SPR, ITC, or enzyme inhibition assay
2. ✅ Molecular docking: Get precise binding pose and affinity estimate
3. ✅ Literature search: Check if this interaction is known
4. ✅ X-ray crystallography: Confirm binding mode (gold standard)

### Validation Strategy

```
High Score (>40)
    ↓
Literature Check
    ↓
Known Interaction? ───YES──→ Validates method, explore further
    │
    NO
    ↓
Molecular Docking (AutoDock, Glide)
    ↓
Good Binding Pose + Affinity? ───YES──→ Proceed to experiments
    │
    NO ──→ Likely false positive
    
Experimental Validation:
├─ In vitro binding assay (SPR, ITC, MST)
├─ Enzyme inhibition assay (IC50)
├─ Cell-based assay (phenotype)
└─ X-ray/Cryo-EM (structure confirmation)
```

---

## Technical Deep Dive

### Algorithm Complexity Analysis

#### Database Building (One-Time Cost)
```
Time Complexity: O(N_proteins × T_neural_net)

Where:
- N_proteins: Number of protein structures (e.g., 1000)
- T_neural_net: Neural network inference time (~30s GPU, ~5min CPU)

Total: 1000 × 30s = ~8 hours (GPU)
```

#### Reverse Screening (Per Query)
```
Time Complexity: O(N_proteins × N_conf × N_lig_feat × N_prot_hotspots)

Where:
- N_proteins: Database size (1000)
- N_conf: Conformers per molecule (50-100)
- N_lig_feat: Ligand features (~5-20)
- N_prot_hotspots: Protein hotspots (~100-200)

Typical: 1000 × 50 × 10 × 100 = 50M operations
Time: ~2-3 minutes (parallelized)
```

#### Space Complexity
```
Database Storage:
- Per .pm file: ~50-500 KB
- 1000 proteins: ~100-500 MB
- Highly compressed (pickle format)

Runtime Memory:
- Query processing: ~100 MB per molecule
- Parallel screening: ~100MB × N_CPUs
- Recommendation: 16GB RAM for smooth operation
```

### Comparison with Other Methods

#### PharmacoNet vs PharmMapper

| Aspect | PharmMapper | PharmacoNet |
|--------|-------------|-------------|
| **Feature Extraction** | Rule-based (LigandScout-like) | Deep learning (3D CNN) |
| **Pharmacophore Model** | Hand-crafted, ~5-10 features | Data-driven, ~100-200 hotspots |
| **Scoring** | Geometric fit score | Graph matching with weighted features |
| **Conformers** | Yes (OMEGA) | Yes (RDKit ETKDG) |
| **Database** | Pre-built (~51,000 proteins) | Custom (user-defined) |
| **Accuracy** | Baseline | +20-30% enrichment over PharmMapper |
| **Speed** | Fast (~1 min/query) | Fast (~2 min/query) |
| **Customization** | Limited | Full control over database |
| **Training** | None (rule-based) | Trained on 10,000+ real complexes |

#### PharmacoNet vs Molecular Docking

| Aspect | Docking (AutoDock Vina) | PharmacoNet |
|--------|------------------------|-------------|
| **Approach** | Exhaustive pose sampling | Pharmacophore matching |
| **Output** | Binding pose + affinity | Pharmacophore compatibility score |
| **Speed** | Slow (~5-10 min/protein) | Fast (~0.1 s/protein) |
| **Accuracy** | High (for binding pose) | Moderate (for target identification) |
| **Use Case** | Lead optimization | Target fishing, off-target prediction |
| **Scalability** | Poor (1000 proteins = days) | Excellent (1000 proteins = minutes) |
| **Protein Prep** | Required (add H, charges) | Not required (uses PDB directly) |

**When to use what:**
- **PharmacoNet:** Screening 100-10,000 proteins to find potential targets (target fishing)
- **Docking:** Detailed analysis of top 10-50 proteins to get binding poses/affinities

**Recommended Workflow:**
```
PharmacoNet (1000 proteins)
    ↓
Filter to top 50
    ↓
Molecular Docking (50 proteins)
    ↓
Filter to top 10
    ↓
Experimental Validation (10 proteins)
```

### Feature Weight Tuning

Default weights are optimized for general targets:
```python
DEFAULT_WEIGHTS = {
    'Cation': 8,       # Strong electrostatics
    'Anion': 8,        # Strong electrostatics
    'Aromatic': 4,     # π-π stacking
    'HBond_donor': 4,  # H-bonding
    'HBond_acceptor': 4,
    'Halogen': 4,      # Halogen bonding
    'Hydrophobic': 1,  # Weak van der Waals
}
```

**Custom tuning for specific targets:**

**Kinases (ATP-binding site):**
```bash
--cation 10 \    # Lys/Arg in hinge region
--hbd 6 \        # Hinge H-bonds critical
--hba 6 \
--aromatic 3     # Less important
```

**GPCRs (ligand-gated ion channels):**
```bash
--aromatic 6 \   # Aromatic clusters important
--hydrophobic 2  # Transmembrane interactions
```

**Metalloproteases:**
```bash
--anion 10 \     # Metal coordination
--hba 6          # Oxyanion hole
```

---

## Best Practices & Tips

### Database Preparation

#### Selecting Proteins

**Quality Criteria:**
- ✅ Resolution < 3.0 Å (preferably < 2.5 Å)
- ✅ Co-crystallized with ligand (not apo structure)
- ✅ Human proteins (if targeting human)
- ✅ Avoid low-quality/problematic structures

**Diversity:**
- Include multiple protein families (kinases, GPCRs, enzymes, receptors)
- Multiple conformations of same protein (if available)
- Orthologues from different species (optional)

**Database Sources:**
```bash
# Download list of all PDB IDs with ligands
wget https://www.rcsb.org/pdb/rest/customReport.csv?pdbids=*&...

# Filter by criteria (Python/pandas)
df = df[(df['resolution'] < 3.0) & (df['has_ligand'] == True)]
```

### Query Molecule Preparation

#### SMILES Best Practices

**DO:**
- ✅ Use canonical SMILES (e.g., from PubChem)
- ✅ Include stereochemistry if known: `C[C@H](O)C(=O)O`
- ✅ Check for typos: Validate with RDKit before screening

**DON'T:**
- ❌ Use SMILES with salts/counterions: `CC(=O)O.[Na]` → Use `CC(=O)O`
- ❌ Mix tautomers: Choose dominant form at pH 7.4
- ❌ Include explicit hydrogens: `[H]C([H])([H])C` → Use `CC`

**Validation:**
```python
from rdkit import Chem

# Check if SMILES is valid
mol = Chem.MolFromSmiles("your_smiles_here")
if mol is None:
    print("Invalid SMILES!")
else:
    print(f"Valid! Molecular weight: {Chem.Descriptors.MolWt(mol)}")
```

### Performance Optimization

#### Parallel Processing

**CPU Parallelization:**
```bash
# Use all CPU cores
python reverse_screening.py --cpus $(nproc)

# Or specify number
python reverse_screening.py --cpus 8
```

**GPU Acceleration:**
```bash
# For database building (modeling)
python batch_modeling.py --cuda

# Note: Reverse screening doesn't use GPU (graph matching is CPU-bound)
```

#### Batch Processing

**Large Query Sets (>100 molecules):**
```bash
# Split query CSV into chunks
split -l 50 query_molecules.csv query_chunk_

# Process in parallel
for chunk in query_chunk_*; do
    python reverse_screening.py \
        --query_csv $chunk \
        --out results_$chunk.csv &
done
wait

# Merge results
cat results_*.csv > final_results.csv
```

### Result Analysis

#### Post-Processing with Python

```python
import pandas as pd

# Load results
df = pd.read_csv('results_target_fishing.csv')

# Group by query, get top 10 per query
top_hits = df.groupby('query_name').head(10)

# Filter by score threshold
strong_hits = df[df['score'] > 30]

# Find common targets across multiple queries
target_counts = df.groupby('pharmacophore_model')['query_name'].count()
promiscuous_targets = target_counts[target_counts > 10]

print("Promiscuous targets (bind to >10 queries):")
print(promiscuous_targets)
```

#### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Score distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['score'], bins=50)
plt.xlabel('Pharmacophore Score')
plt.ylabel('Frequency')
plt.title('Score Distribution Across All Matches')
plt.savefig('score_distribution.png')

# Heatmap: Query × Target
pivot = df.pivot(index='query_name', 
                 columns='pharmacophore_model', 
                 values='score')
plt.figure(figsize=(20, 10))
sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Score'})
plt.title('Query-Target Score Matrix')
plt.savefig('heatmap.png')
```

---

## Troubleshooting

### Common Issues

#### 1. "No .pm files found in database directory"

**Cause:** Database not built or wrong directory path

**Solution:**
```bash
# Check if .pm files exist
find output -name "*.pm" | head

# If empty, rebuild database
python batch_modeling.py --input_csv input/pdb_database.csv
```

#### 2. "Error processing SMILES: Can't parse molecule"

**Cause:** Invalid SMILES string

**Solution:**
```python
from rdkit import Chem

# Test SMILES
smiles = "your_smiles_here"
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    # Try fixing
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))
```

#### 3. "CUDA out of memory"

**Cause:** GPU memory insufficient for batch modeling

**Solution:**
```bash
# Process one at a time (slower but works)
python batch_modeling.py --cuda

# Or use CPU mode
python batch_modeling.py  # No --cuda flag
```

#### 4. Low/No scores for all proteins

**Possible causes:**
- Query molecule too small (< 5 heavy atoms)
- Query molecule too flexible (> 15 rotatable bonds)
- Wrong pharmacophore model database
- Feature extraction failed

**Diagnosis:**
```python
from rdkit import Chem
from rdkit.Chem import Descriptors

mol = Chem.MolFromSmiles("your_smiles")
print(f"Heavy atoms: {mol.GetNumHeavyAtoms()}")
print(f"Rotatable bonds: {Descriptors.NumRotatableBonds(mol)}")
print(f"H-bond donors: {Descriptors.NumHDonors(mol)}")
print(f"H-bond acceptors: {Descriptors.NumHAcceptors(mol)}")
```

**Solution:**
- Ensure molecule has drug-like properties (Lipinski's Rule of 5)
- Check if features are extracted: Run `screening.py` with verbose output

#### 5. "Too many conformers generated"

**Cause:** Highly flexible molecule (>10 rotatable bonds)

**Solution:**
```bash
# Reduce conformer count
python reverse_screening.py --num_conformers 20  # Instead of 100
```

### Performance Issues

#### Slow screening (>10 minutes for 1000 proteins)

**Diagnosis:**
```bash
# Check CPU usage
top
htop

# Check if parallelization is working
ps aux | grep python | wc -l  # Should show multiple processes
```

**Solutions:**
1. Increase CPU cores: `--cpus 16`
2. Reduce conformers: `--num_conformers 50`
3. Filter database: Remove low-resolution proteins

#### High memory usage

**Cause:** Too many parallel processes

**Solution:**
```bash
# Reduce parallel processes
python reverse_screening.py --cpus 4  # Instead of 16

# Monitor memory
free -h
```

### Data Quality Issues

#### Many low-quality .pm files

**Cause:** Low-resolution PDB structures or apo (no ligand) structures

**Solution:**
```bash
# Remove problematic files
find output -name "*.pm" -size -10k -delete  # Remove files < 10KB

# Re-build with quality filter
# Edit pdb_database.csv to include only high-resolution structures
```

---

## Advanced Use Cases

### 1. Drug Repurposing

**Goal:** Find new targets for FDA-approved drugs

```bash
# Download FDA-approved drug SMILES from DrugBank
# Create query_fda_drugs.csv

python reverse_screening.py \
  --query_csv query_fda_drugs.csv \
  --model_database_dir human_proteome_database \
  --out fda_repurposing_results.csv \
  --top_n 20
```

### 2. Natural Product Target Identification

**Goal:** Identify protein targets for plant compounds

```bash
# Extract natural products from traditional medicine
# Example: Cannabis compounds

python reverse_screening.py \
  --query_csv cannabis_compounds.csv \
  --model_database_dir cns_proteins_database \
  --out cannabis_targets.csv
```

### 3. Off-Target Prediction (Safety)

**Goal:** Predict unwanted interactions for lead compounds

```bash
# Screen against anti-targets (hERG, CYP450, etc.)
python reverse_screening.py \
  --query_csv lead_compounds.csv \
  --model_database_dir anti_targets_database \
  --out off_target_predictions.csv \
  --min_score 15.0  # Lower threshold for safety
```

### 4. Polypharmacology Analysis

**Goal:** Understand multi-target binding profiles

```bash
# Screen one drug against entire proteome
python reverse_screening.py \
  --query_molecule "aspirin_smiles" \
  --model_database_dir full_proteome_database \
  --out aspirin_proteome_profile.csv \
  --top_n 100
```

---

## Citation & References

### PharmacoNet Paper

If you use PharmacoNet in your research, please cite:

```
[Citation to be added - PharmacoNet publication]
```

### Related Methods

**PharmMapper:**
- Liu, X., et al. "PharmMapper server: a web server for potential drug target identification using pharmacophore mapping approach." *Nucleic Acids Research* (2010).

**Molecular Docking:**
- Trott, O. & Olson, A. J. "AutoDock Vina: improving the speed and accuracy of docking." *Journal of Computational Chemistry* (2010).

**RDKit (Conformer Generation):**
- RDKit: Open-source cheminformatics. https://www.rdkit.org

### Useful Resources

- **RCSB PDB:** https://www.rcsb.org/
- **PubChem:** https://pubchem.ncbi.nlm.nih.gov/
- **DrugBank:** https://go.drugbank.com/
- **ChEMBL:** https://www.ebi.ac.uk/chembl/
- **ProteomicsDB:** https://www.proteomicsdb.org/

---

## Summary

### Quick Start Commands

**1. Build Database:**
```bash
python batch_modeling.py --input_csv input/pdb_database.csv --cuda
```

**2. Run Reverse Screening:**
```bash
python reverse_screening.py \
  --query_csv input/query_molecules.csv \
  --model_database_dir output \
  --out results.csv \
  --num_conformers 100 \
  --cpus 8
```

**3. Analyze Results:**
```python
import pandas as pd
df = pd.read_csv('results.csv')
top_hits = df.groupby('query_name').head(10)
print(top_hits)
```

### Key Takeaways

1. **PharmacoNet = AI-powered PharmMapper**
   - Deep learning features vs hand-crafted rules
   - Custom database vs fixed database
   - More accurate for target fishing

2. **Two-Phase Workflow**
   - Phase 1: Build protein database (one-time, ~8 hours)
   - Phase 2: Screen queries (fast, ~2 min per molecule)

3. **Scores are Pharmacophore Compatibility**
   - NOT binding affinity
   - High score → Experimental validation needed
   - Combine with docking for detailed analysis

4. **Best Practices**
   - High-quality protein structures (< 3Å resolution)
   - Validated SMILES strings
   - Appropriate feature weights for target class
   - Parallel processing for speed

### Next Steps

1. ✅ Build your protein database
2. ✅ Prepare query molecules
3. ✅ Run reverse screening
4. ✅ Analyze top hits
5. ✅ Validate with docking
6. ✅ Experimental confirmation

**Good luck with your target fishing! 🎯🧬**
