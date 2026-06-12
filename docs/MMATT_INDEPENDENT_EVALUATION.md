
---

### `docs/MMATT_INDEPENDENT_EVALUATION.md`

```markdown
# MMAtt-DTA Independent Evaluation

This document describes how to reproduce the independent evaluation on the MMAtt-DTA kinase test set.

## Overview

The MMAtt-DTA independent test set provides three official evaluation scenarios:

1. A) Imputation
2. B) New compound
3. C) New compound + new target

SMCL-DTA was evaluated on the kinase subset for consistency with the Davis and KIBA affinity prediction benchmarks.

## Data Alignment

The original independent kinase subset contains 42,284 drug-target interactions. These records are aligned with the official MMAtt-DTA scenario files.

After alignment, 42,200 official scenario-labeled interactions are retained:

```text
A) Imputation: 215 samples
B) New compound: 41,378 samples
C) New compound + new target: 607 samples