
---

### `docs/SPLITS.md`

```markdown
# Scaffold- and Cluster-Based Cold-Start Splits

This document describes the data split protocols used in SMCL-DTA.

## Ligand Scaffold Split

For ligand cold-start evaluation, compounds are split according to Bemis–Murcko scaffolds. Compounds sharing the same scaffold are assigned to the same subset to avoid scaffold leakage between training and testing.

This setting evaluates whether the model can generalize to compounds with unseen chemical scaffolds.

## Protein Cluster Split

For target cold-start evaluation, target proteins are grouped into clusters based on protein representations. Protein representations are obtained using ProtT5 embeddings, followed by dimensionality reduction and unsupervised clustering.

Entire protein clusters are held out during testing. This is stricter than holding out individual proteins because it reduces the chance that highly similar proteins appear in both training and testing.

## Pair-Level Cold-Start Split

The pair-level cold-start setting evaluates the most challenging scenario, where both compounds and targets are outside the training distribution.

## Released Split Files

The exact split files used in the manuscript are provided under:

```text
splits/classification/
splits/regression/