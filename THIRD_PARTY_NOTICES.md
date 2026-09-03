# Third-Party Software and Attribution

This repository incorporates, adapts, or was developed using ideas and
software from several external projects.

## Inner Product Transforms

This project was originally developed from the codebase associated with:

> Ernst Röell and Bastian Rieck. *Point Cloud Synthesis Using Inner Product
> Transforms*. NeurIPS, 2025.

Original repository:
https://github.com/aidos-lab/inner-product-transforms

The original software is distributed under the BSD 3-Clause License.

Copyright (c) 2025, Ernst Röell and Bastian Rieck.

The present repository has subsequently been substantially modified and
refactored, and much of the original functionality has been removed.
Nevertheless, attribution is retained because parts of the implementation
and/or its development history may derive from the original codebase.

See `licenses/INNER_PRODUCT_TRANSFORMS_LICENSE.md` for the original license.

## Equiformer

Parts of the equivariant graph encoder are adapted from:

> Yi-Lun Liao and Tess Smidt. "Equiformer: Equivariant Graph Attention
> Transformer for 3D Atomistic Graphs." ICLR, 2023.

Original repository:
https://github.com/atomicarchitects/equiformer

Equiformer is distributed under the MIT License.

Copyright (c) 2023 Yi-Lun Liao.

The original Equiformer license is retained with the vendored implementation
under `src/models/Equiformer/equiformer/LICENSE`.

## Gumbel-Sinkhorn

The differentiable permutation formulation is based on:

> Gonzalo Mena et al. "Learning Latent Permutations with Gumbel-Sinkhorn
> Networks." ICLR, 2018.

The paper is cited for the method. [No source code from the reference
implementation is included in this repository.]

## Protein dataset

The protein dataset and train/validation/test split are derived from the
dataset released with:

> John Ingraham et al. "Generative Models for Graph-Based Protein Design."
> NeurIPS, 2019.

The underlying structures are derived from CATH. Dataset licensing and
attribution are documented separately in `data/README.md`.