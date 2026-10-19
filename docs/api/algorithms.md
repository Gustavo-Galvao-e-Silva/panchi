# panchi.algorithms

```python
from panchi.algorithms import ref, rref, lu, qr_decomposition
from panchi.algorithms import RowSwap, RowScale, RowAdd
from panchi.algorithms import Reduction, LUDecomposition, QRDecomposition
from panchi.algorithms import InverseResult, Solution, EigenResult, QREigenResult
from panchi.algorithms import inverse, solve, determinant
from panchi.algorithms import cofactor_expansion, determinant_by_reduction, adjugate, cramer
from panchi.algorithms import eigen, characteristic_polynomial, eigen_qr
from panchi.algorithms import rank, nullity, is_invertible, is_symmetric
from panchi.algorithms import span, standard_basis, column_space, row_space, null_space
from panchi.algorithms import dot, cross, orthogonal_complement
from panchi.algorithms import gram_schmidt, vector_projection
```

## Row operations

::: panchi.algorithms.RowSwap

::: panchi.algorithms.RowScale

::: panchi.algorithms.RowAdd

## Reductions

::: panchi.algorithms.ref

::: panchi.algorithms.rref

## Decompositions

::: panchi.algorithms.lu

::: panchi.algorithms.qr_decomposition

## Eigenvalues

::: panchi.algorithms.eigen

::: panchi.algorithms.characteristic_polynomial

::: panchi.algorithms.eigen_qr

## Solvers

::: panchi.algorithms.inverse

::: panchi.algorithms.solve

::: panchi.algorithms.determinant

!!! warning "Deprecated in 3.1.0, removed in 4.0"
    Use `determinant`, or `determinant_by_reduction` for an exact result with steps.

::: panchi.algorithms.determinant_lu

## Determinants step by step

::: panchi.algorithms.cofactor_expansion

::: panchi.algorithms.determinant_by_reduction

::: panchi.algorithms.adjugate

::: panchi.algorithms.cramer

## Matrix properties

::: panchi.algorithms.rank

::: panchi.algorithms.nullity

::: panchi.algorithms.is_invertible

::: panchi.algorithms.is_symmetric

## Vector space operations

::: panchi.algorithms.basis

::: panchi.algorithms.is_full_rank

::: panchi.algorithms.contains

::: panchi.algorithms.same_subspace

::: panchi.algorithms.orthogonal_complement

::: panchi.algorithms.gram_schmidt

::: panchi.algorithms.vector_projection

## Subspace factories

::: panchi.algorithms.span

::: panchi.algorithms.standard_basis

::: panchi.algorithms.column_space

::: panchi.algorithms.row_space

::: panchi.algorithms.null_space

## Result types

::: panchi.algorithms.Reduction

::: panchi.algorithms.LUDecomposition

::: panchi.algorithms.QRDecomposition

::: panchi.algorithms.InverseResult

::: panchi.algorithms.CofactorExpansion

::: panchi.algorithms.CofactorTerm

::: panchi.algorithms.DeterminantReduction

::: panchi.algorithms.CramerResult

::: panchi.algorithms.Solution

::: panchi.algorithms.EigenResult

::: panchi.algorithms.QREigenResult
