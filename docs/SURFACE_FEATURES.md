# Surface Feature Extraction

SMCL-DTA uses surface-aware representations for both ligand molecules and target proteins.

## Protein Surface Features

For target proteins, SMCL-DTA follows a MaSIF-style surface preprocessing pipeline. The three-dimensional protein structure is first processed to obtain heavy-atom coordinates. Molecular surface meshes are generated using MSMS, which produces solvent-excluded surface vertices and faces from the input structure.

The raw molecular surface mesh is then refined using PyMesh-based mesh processing. The raw vertices and faces are converted into a mesh object and regularized using a mesh-fixing procedure to obtain a consistent surface resolution. Surface normal vectors are computed from the refined mesh.

For each surface vertex, geometric and physicochemical descriptors are computed, including:

- spatial coordinates;
- surface normal vectors;
- electrostatic properties;
- hydrogen-bonding-related features;
- hydrophobicity.

Electrostatic properties are computed using an APBS-based electrostatics module and normalized before being stored. Hydrogen-bonding features are calculated using atom-level charge assignment and mapped from the original molecular surface to the refined mesh. Hydrophobicity values are assigned according to atom or residue identity.

Each protein is represented as a fixed-size point cloud with 512 sampled surface points. Each point is encoded by a 9-dimensional feature vector:

```text
[x, y, z, nx, ny, nz, charge, hbond, hydrophobicity]