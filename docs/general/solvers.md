---
title: Solvers
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: true
toc_icon: "chevron-circle-down"
last_modified_at: 2026-01-28
---

Legolas has interfaces implemented to various BLAS, LAPACK and ARPACK routines.
Below is an overview of which routines you can call, which problems are supported and how
you can configure the parfile to select the solver you want.
We have have a general eigenvalue problem of the form

$$ A\mathbf{x} = \omega B\mathbf{x} $$

where $A$ is a non-symmetric and non-Hermitian complex matrix. The $B$-matrix is always real, and in most
cases also symmetric and positive definite (depending on the physics, the Hall electron inertia term for example
breaks positive definiteness). Both matrices are block-tridiagonal, meaning they are very sparse.

In Legolas 2.0 we did a complete overhaul of the various solvers, resulting in new solver methods, a major performance boost and considerable increase in accuracy.


<i class="fas fa-lightbulb" aria-hidden="true"></i>
**Note**: A general strategy for a thorough investigation of a certain spectrum may be as follows: first a low-resolution QR-invert run is done,
which will reveal spectral regions of interest. This can then be followed-up by a higher-resolution run using QR-invert, and/or a shift-invert Arnoldi run
near the interesting regions. Comparing the eigenvalues between both solution strategies and at different resolutions will be a good indicator of their convergence.
{: .notice--success}


## No solver
The eigenvalue solve can be bypassed entirely by selecting the `"none"` solver:
```fortran
&solvelist
  solver = "none"
/
```
The matrices are still assembled, but no eigenvalue solver is called and the eigenvalue array in
the datfile is left empty. Eigenfunction, eigenvector and residual output is switched off
automatically, since these are meaningless without eigenvalues.

This is useful for a dry run (for example to assemble and write the matrices with
`write_matrices = .true.`), and for [initial-value runs](../../general/initial_value), which reuse
the assembled `A` and `B` matrices but do not need the eigenvalues. Initial-value snapshot output
is unaffected by this setting.


## QR-invert
This is the default solver that Legolas uses, which transforms the general eigenvalue problem into a standard one:

$$ B^{-1}A\mathbf{x} = \omega\mathbf{x}, $$

where we do not explicitly calculate the inverse of $B$. Matrix sparsity is exploited instead, where a banded storage form of $B$ is used to solve the system

$$ B\mathbf{x} = A $$

for $\mathbf{x}$, yielding $B^{-1}A$.

The LAPACK routine [`zgbsv`](https://www.netlib.org/lapack/explore-html/db/df8/group__gbsv_gae5d2fd844bacac0f2456eca12b4fed61.html)
is used to solve the system of linear equations. This is followed by a call to  LAPACK's [`zgeev`](https://www.netlib.org/lapack/explore-html/d4/d68/group__geev_ga8656466cd9da86a0b3b6966e49a7ce40.html)
which returns all eigenvalues and optionally the right eigenvectors.

Note that while this routine stores $B$ in banded form, the product $B^{-1}A$ passed to `zgeev` has to be in a dense format (which unfortunately loses sparsity). For very high-resolution runs this routine should not be used, unless enough RAM is available to store such a dense matrix in memory. Note that in full MHD matrix dimensions are $16N \times 16N$, in HD this reduces to $10N \times 10N$ with $N$ the number of gridpoints in the base grid.

This solver can be explicitly specified in the `solvelist` through
```fortran
&solvelist
  solver = "QR-invert"
/
```
and is called by default if no `solvelist` is supplied.
{% capture pros %}
**Pros:**
- Fast and accurate
- Calculates complete spectrum and eigenfunctions
{% endcapture %}
<div class="notice--success">
  {{ pros | markdownify }}
</div>

{% capture cons %}
**Cons:**
- Needs to store one dense matrix in memory
- Datfiles become very large at high resolution if eigenfunctions are included, this can be mitigated by saving a subset.
- _May_ show numerical instability at very large $Re(\omega)$ values (far into the sequence), this should always be visually clear in the spectrum.
  In most cases this is not a problem, since those modes will not have their eigenfunctions resolved anyway.
{% endcapture %}
<div class="notice--danger">
  {{ cons | markdownify }}
</div>


## QR-cholesky
In most cases the $B$-matrix is Hermitian, such that a Cholesky decomposition can be exploited. This has numerous advantages with respect to a LU-decomposition (such as used by QR-invert), mainly in terms of efficiency and accuracy.
The eigenvalue problem is rewritten into standard form, but instead of solving for $B^{-1}A$ the $B$-matrix is written as a product of a lower triangular matrix and its conjugate transpose.

First the $B$-matrix is converted into an upper triangular banded form storage, then the Cholesky decomposition is calculated using LAPACK's [`zpbtrf`](https://www.netlib.org/lapack/explore-html/dc/d58/group__pbtrf_ga1fb0b071f1f8c4ea31e5561d491e6670.html).
Next various calls are made to BLAS's [`ztbsv`](https://www.netlib.org/lapack/explore-html/d4/dcd/group__tbsv_ga0f56e4311a965b79cb37fcf86881f925.html), eventually constructing the matrix $U^{-H}AU^{-1}$. Finally this is passed to LAPACK's [`zgeev`](https://www.netlib.org/lapack/explore-html/d4/d68/group__geev_ga8656466cd9da86a0b3b6966e49a7ce40.html)
which returns all eigenvalues and optionally the right eigenvectors.

Note that while this routine does its calculations with matrices in banded storage, the final product passed to `zgeev` has to be in a dense format. As such, the same memory remarks and constraints as QR-invert are relevant.

Also note that this can only be used if the $B$-matrix is Hermitian. This depends on the physics taken into consideration, for example the electron inertia term in the Hall effect breaks the Hermitianness of the $B$-matrix. Legolas will throw an error if this is the case.

This solver can be explicitly specified in the `solvelist` through
```fortran
&solvelist
  solver = "QR-cholesky"
/
```

{% capture pros %}
**Pros:**
- May be faster and more accurate than QR-invert
- Calculates complete spectrum and eigenfunctions
{% endcapture %}
<div class="notice--success">
  {{ pros | markdownify }}
</div>

{% capture cons %}
**Cons:**
- Needs to store one dense matrix in memory
- Datfiles become very large at high resolution if eigenfunctions are included, this can be mitigated by saving a subset.
- Only works if the $B$-matrix is Hermitian, which is not always the case.
{% endcapture %}
<div class="notice--danger">
  {{ cons | markdownify }}
</div>

## QZ-direct
This is a variant of the QR-invert solver, with as main difference that the eigenvalue problem is kept in its general form.
The LAPACK routine [`zggev`](https://www.netlib.org/lapack/explore-html/da/da1/group__ggev_ga4cb59b337869a6570619392409887992.html)
is used to solve the general eigenvalue problem, returning all eigenvalues.
Note that this routine only works with dense matrices, so contrary to QR-invert this stores **two** dense matrices in memory ($A$ and $B$).

This solver can be specified in the `solvelist` through
```fortran
&solvelist
  solver = "QZ-direct"
/
```

{% capture pros %}
**Pros:**
- No inversion of the $B$-matrix needed, may increase accuracy in some cases.
- Calculates complete spectrum
{% endcapture %}
<div class="notice--success">
  {{ pros | markdownify }}
</div>

{% capture cons %}
**Cons:**
- Needs more memory than QR-invert, as it needs to store 2 dense matrices in memory instead of one.
- Slower than QR-invert/cholesky.
- Datfiles become very large at high resolution if eigenfunctions are included, this can be mitigated by saving a subset.
{% endcapture %}
<div class="notice--danger">
  {{ cons | markdownify }}
</div>

## ARPACK Reverse Communication Interface
Legolas has various solvers implemented which interface with the ARPACK package to
solve the eigenvalue problem. ARPACK is a reverse communication interface specifically designed to
solve large-scale, sparse matrix eigenvalue problems, and is hence perfectly suited for Legolas.
ARPACK can run in various modes, most notably a shift-invert method to probe
various parts of the spectrum, only returning eigenvalues of regions you are interested in.

As these routines iterate towards their solution we can fully exploit matrix sparsity. As such there is a major advantage here with respect to the QR-variants: due to the iterative nature we only need matrix-vector products. Both matrices $A$ and $B$ are **never** fully stored in memory. A linked-list representation is used for matrix data-storage, which only stores non-zero elements. All calculations are done using LAPACK's banded matrices, and conversions from linked-lists to banded structures are done when necessary. This means that (much) higher resolutions are possible here, as we are no longer bound by memory limitations for storing the dense matrices. The spectrum can then be constructed through various shift-invert runs.

The main difference with the LAPACK solvers is that one can query for only a number of eigenvalues
instead of the full spectrum. This is essentially the Fortran analog of SciPy's
[`eigs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html)
method in Python, which is a wrapper to ARPACK in itself.

When using Arnoldi-based solvers the solvelist can be set as follows:
```fortran
&solvelist
  solver = "arnoldi"
  arpack_mode = "general" | "shift-invert"
  number_of_eigenvalues = 20
  which_eigenvalues = "LM" | "SM" | "LR" | "SR" | "LI" | "SI"
  maxiter = 2500
  ncv = 50
/
```

- `arpack_mode`: which mode to use, see the subsections below.
- `number_of_eigenvalues`: the number of eigenvalues $k$ to be computed.
- `which_eigenvalues`: denotes the type of eigenvalues that should be calculated
   - `"LM"`: eigenvalues with largest magnitude
   - `"SM"`: eigenvalues with smallest magnitude
   - `"LR"`: eigenvalues with largest real part
   - `"SR"`: eigenvalues with smallest real part
   - `"LI"`: eigenvalues with largest imaginary part
   - `"SI"`: eigenvalues with smallest imaginary part
- `maxiter`: integer which limits the maximum iterations the Arnoldi solver may take when
   doing the reverse communication. This defaults to $\max(100, 10N)$, where $N$ is the
   number of gridpoints in the base grid, and is usually more than sufficient.
- `ncv`: the number of basis vectors to use during the iteration, by default set to $\max(k + 1, \min(2k, 10N))$.
    Tweaking this value can have a large impact on convergence and/or performance, see the points of note below.

If the iterative solver reaches `maxiter`, only a number $j < k$ eigenvalues will be converged.
Legolas will notify you how many are converged, and you can still plot these $j$ eigenvalues and their eigenfunctions.

Note that ARPACK is better at finding large eigenvalues, and for small eigenvalues shift-invert is the better choice.
Ideally a combination of direct and iterative solvers is used, where one first solves for all eigenvalues using a QR variant, locate
spectral regions of interest, and then follow-up with shift-invert at those locations.

### Points of note
Before going over the implemented Arnoldi-routines a few important points are highlighted here. On this page only a few key points regarding ARPACK are explained,
for a thorough overview of all its possibilities we refer to [the documentation (pdf)](http://li.mit.edu/Archive/Activities/Archive/CourseWork/Ju_Li/MITCourses/18.335/Doc/ARPACK/Lehoucq97.pdf).

**The computational time is strongly dependent on the choice of `ncv`, `number_of_eigenvalues`, `which_eigenvalues` (and `sigma` for shift-invert)**.
The piece of text below is quite important and is taken directly from the ARPACK documentation, section 2.3.3. The parameter `nev` refers to `number_of_eigenvalues`.
> For a given `ncv`, the computational work required is proportional to `n $\cdot$ ncv$^2$` FLOPS. Setting `nev` and `ncv` for optimal performance is very much
> problem dependent. If possible, it is best to avoid setting `nev` in a way that will split clusters of eigenvalues. For example, if the five smallest eigenvalues
> are positive and on the order of $10^{-4}$ and the sixth eigenvalue is on the order of $10^{-1}$ then it is probably better to ask for `nev = 5`
> than for `nev = 3` even if the three smallest are the only ones of interest.
>
> Setting the optimal value of `ncv` relative to `nev` is not completely understood. As with the choice of `which_eigenvalues`, it depends upon the underlying approximation
> properties of the Iterative Method as well as the distribution of the eigenvalues of the $A$-matrix. As a rule of thumb, `ncv $\geq$ 2 $\cdot$ nev` is reasonable.
> There are tradeoffs due to the cost of matrix-vector products, the cost of the implicit restart mechanism and the cost of maintaining the orthogonality of the Lanczos vectors.
> If the matrix-vector product is relatively cheap (as is the case for Legolas), then a smaller value of `ncv` may lead to more matrix-vector products, but an overall decrease
> in computation time.



### General mode
Set `arpack_mode = "general"`. This solves the eigenvalue problem $AX = \omega BX$  in its general form, which technically comes down to solving the problem $R = B^{-1}AX = \omega X$.
In order to exploit matrix sparsity we never explicitly calculate $R$, but do the following instead.

In a first step the $B$-matrix is converted from its linked-list representation
to a banded matrix datastructure. Then, at every step in the iteration:
1. The matrix-vector product $u = AX$  is calculated, in which the $A$-matrix is kept in its linked-list representation. This makes the matrix-vector product efficient and fast.
2. The linear system $BR = u$ is solved for R using LAPACK's banded matrix routines and passed back to the solver.

The above steps are repeated until convergence or until `maxiter` is reached.


{% capture pros %}
**Pros:**
- Calculates only specific eigenvalues and eigenvectors
- Reduced datfile size, only eigenvectors for requested eigenvalues are calculated
{% endcapture %}
<div class="notice--success">
  {{ pros | markdownify }}
</div>

{% capture cons %}
**Cons:**
- May require trial-and-error to find an optimal set of parameters
- Fast for the largest eigenvalues (`"LM", "LR", "LI"`), slower for the others
- Only calculates spectrum extremes (largest/smallest) which are usually not modes of interest.
{% endcapture %}
<div class="notice--danger">
  {{ cons | markdownify }}
</div>

### Shift-invert mode
Set `arpack_mode = "shift-invert"`. Running ARPACK in shift-invert mode is usually used to enhance convergence of certain spectral regions. It relies on a transformation of the eigenvalue problem to

$$
 \Bigl(A - \sigma B\Bigr)^{-1} BX = \nu X \qquad \text{where} \qquad \nu = \dfrac{1}{\omega - \sigma}
$$

This transformation is particularly useful for finding eigenvalues near $\sigma$. It transforms small eigenvalues into large ones, making them perfectly suitable for calculation with ARPACK. Here it is **recommended** to keep `which_eigenvalues = "LM"`, since the eigenvalues $\nu_j$ of $ C \equiv \left(A - \sigma B\right)^{-1}B$ that are largest in magnitude correspond to the values $\omega_j$ of the original eigenvalue problem that are nearest to the shift $\sigma$ in absolute value.

The value of $\sigma$ can be specified by adding it to the solvelist, like so
```fortran
&solvelist
  sigma = (1.0d0, 0.05d0)
/
```
and should be a complex tuple (standard Fortran notation for complex numbers).

The eigenvalue problem is tackled as follows.

First the matrices $A$ and $B$ are transformed to banded form, followed by a calculation of $A - \sigma B$. Then, at every step in the iteration:
1. The matrix-vector product $u = BX$  is calculated, in which the $B$-matrix is kept in its linked-list representation. This makes the matrix-vector product efficient and fast.
2. The linear system $\left(A - \sigma B\right)R = u$ is solved for $R$ using LAPACK's banded matrix routines and passed back to the solver.

The above steps are repeated until convergence or until `maxiter` is reached. In the final step we retransform the eigenvalues $\nu_j$ to the
eigenvalues $\omega_j$ of the original problem using

$$
\omega_j = \sigma + \dfrac{1}{\nu_j}
$$


{% capture pros %}
**Pros:**
- Ability to probe specific parts of the spectrum by shifting $\sigma$
- Better performance for small eigenvalues due to the transformation
- Can run at extreme resolutions
{% endcapture %}
<div class="notice--success">
  {{ pros | markdownify }}
</div>

{% capture cons %}
**Cons:**
- May require trial-and-error to find an optimal set of parameters
- Runtime is strongly dependent on the chosen shift $\sigma$
{% endcapture %}
<div class="notice--danger">
  {{ cons | markdownify }}
</div>


## Inverse vector iteration
The latest solver addition to the code is the ability to do inverse vector iteration. This allows one to specify a shift $\sigma$ in the complex plane, and the code will iterate towards the closest eigenvalue to that shift. This method is perfect if one (or more) particular eigenvalues are desired and their approximate location in the spectral plane is known (through QR-variants or shift-invert). In most cases this iteration will be faster than running Arnoldi shift-invert for one eigenvalue.

This solver can be specified in the parfile by setting
```fortran
&solvelist
  solver = "inverse-iteration"
  sigma = (1.0d0, 0.05d0)
  maxiter = 100
  tolerance = 1.0d-7
/
```
The quantity `maxiter` (default 100 if not given) specifies the maximum number of iterations, the inverse iteration process will continue until the eigenvalue is either converged or `maxiter` is reached. Note that if no tolerance is given the solver defaults to a tolerance near machine precision (`5e-15`). In some cases this may be too strict such that Legolas will have difficulties converging, typically warnings will be raised that `maxiter` is reached.
The eigenvalue obtained after one iteration is considered converged if the following criterion is satisfied

$$
|| \mathbf{A}\mathbf{x}_i - \mu_{i + 1}\mathbf{B}\mathbf{x}_i || < |\mu_{i + 1}\epsilon|,
$$

with $\epsilon$ the tolerance. For small eigenvalues and tolerances this right-hand side may drop below machine precision, which implies that Legolas will never converge.
In typical use cases a tolerance of $\epsilon = 10^{-7}$ is more than sufficient.
