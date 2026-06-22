````markdown
# Surface Feature Extraction

SMCL-DTA uses precomputed surface representations for both ligand molecules and target proteins. Surface-feature extraction is performed offline, and the generated features are cached for model training and inference.

## Protein Surface Features

Protein surface features are generated using a MaSIF-style preprocessing pipeline. Three-dimensional protein structures are first processed to obtain atomic coordinates. MSMS is then used to generate solvent-excluded molecular surface meshes containing vertices and triangular faces.

The raw meshes are refined using PyMesh-based mesh processing to remove irregular elements and obtain a more consistent surface resolution. Surface-normal vectors are subsequently calculated from the refined mesh.

For each protein surface point, SMCL-DTA uses geometric and physicochemical descriptors, including:

- spatial coordinates;
- surface-normal vectors;
- electrostatic properties;
- hydrogen-bonding-related features;
- hydrophobicity.

Electrostatic features are calculated using an APBS-based pipeline, while hydrogen-bonding and hydrophobicity features are assigned from nearby atoms or residues and mapped onto the refined surface mesh.

Each protein is represented by 512 sampled surface points, with a 9-dimensional feature vector for each point:

```text
[x, y, z, nx, ny, nz, charge, hbond, hydrophobicity]
````

The final protein surface tensor has shape:

```text
[512, 9]
```

## Ligand Surface Features

Ligand structures are generated from canonical SMILES strings using RDKit. After three-dimensional conformer generation and geometry optimization, molecular surface points and their associated geometric and physicochemical descriptors are calculated.

Each ligand is represented by 80 sampled surface points, with a 6-dimensional feature vector for each point. The final ligand surface tensor has shape:

```text
[80, 6]
```

## Model Usage

The cached ligand and protein surface features are loaded through the dataset pipeline and encoded by separate surface encoders in SMCL-DTA. The resulting surface representations are fused with molecular-graph and protein-sequence features for affinity prediction.

The main related files are:

```text
preprocessing.py
src/dataset.py
src/model_0428_16_dual.py
scripts/
scripts_pdbbind/
```

External tools used in the surface-processing pipeline include MSMS, PyMesh, APBS, and RDKit.

```
```
