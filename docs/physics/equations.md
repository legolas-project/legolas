---
title: Equations
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: true
toc_icon: "chevron-circle-down"
last_modified_at: 2026-01-28
---

On this page we give a small overview of the system of equations solved by Legolas. We use the
full set of MHD equations, linearised around a stationary background.
Physical effects include flow, (external) gravity, resistivity, optically thin radiative losses,
anisotropic thermal conduction, viscosity and Hall effects.

## System of equations
<!-- mathematical symbols used on this page (spaces needed before and after $$ to render properly) -->

$$
\newcommand{\vbf}{\mathbf{v}}
\newcommand{\gbf}{\mathbf{g}}
\newcommand{\bbf}{\mathbf{B}}
\newcommand{\HL}{\mathscr{L}}
\newcommand{\fancyH}{\mathcal{H}}
\newcommand{\kappabf}{\boldsymbol{\kappa}}
\newcommand{\unit}[1]{\mathbf{e}_{#1}}

\begin{gather}
	\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \vbf), \\
	\rho\frac{\partial \vbf}{\partial t} = -\nabla p - \rho \vbf \cdot \nabla \vbf + (\nabla \times \bbf) \times \bbf
                    + \rho\gbf + \mu\left[\nabla^2\vbf + \frac{1}{3}\nabla(\nabla \cdot \vbf)\right], \\
	\begin{aligned}
		\rho\frac{\partial T}{\partial t} = &-\rho \vbf\cdot\nabla T - (\gamma - 1)p\nabla \cdot \vbf- (\gamma - 1)\rho\HL \\
                &+ (\gamma - 1)\nabla \cdot (\kappabf \cdot \nabla T)
				+ (\gamma - 1)\eta(\nabla \times \bbf)^2 + \mu\left|\nabla \vbf \right|^2,
	\end{aligned} \\
	\begin{aligned}
		\frac{\partial \bbf}{\partial t} = ~&\nabla \times (\vbf \times \bbf) - \nabla \times (\eta\nabla \times \bbf) \\
				&-\nabla\times \left[ \frac{\eta_H}{\rho_e}(\nabla \times \bbf) \times \bbf - \frac{\eta_H}{\rho_e}\nabla p
                                        + \frac{\eta_e}{\rho_e}\frac{\partial (\nabla \times \bbf)}{\partial t} \right].
	\end{aligned}
\end{gather}
$$

## Physical effects
Below we briefly describe the various physical effects that can be added. For more information on how to add these in your particular
setup take a look at the [user submodule](../../general/own_setup#including-additional-physics) page.

### External gravity
An external gravitational field can be added, which can either be constant or dependent on position. In Cartesian geometry we assume this to be
$x$-dependent and aligned with height:

$$
\gbf = -g(x)\unit{x}.
$$

A similar profile can be used in cylindrical geometries, e.g. for accretion disk setups.

### Resistivity
Resistivity can either be constant over the entire grid, or a more general temperature-dependent profile given
by the Spitzer resistivity:

$$
\eta(T) = \frac{4\sqrt{2\pi}}{3}\frac{Z_\text{ion}e^2\sqrt{m_e}\ln(\lambda)}{(4\pi\epsilon_0)^2(k_B T)^{3/2}},
$$

with the Coulomb logarithm $\ln(\lambda) \approx 22$. The function $\eta(T)$ can be user-specified, in which case its temperature derivative should be provided as well.
If the resistivity profile explicitly depends on position as well, providing the derivative of $\eta(x, T)$ with respect to $x$ is an additional requirement.

### Heating and optically thin radiative losses
Radiative cooling is governed by the heat-loss function, specified as the difference between energy gains and energy losses

$$
\HL = \rho\Lambda(T) - \fancyH(\rho, T),
$$

The function $\Lambda(T)$ here is called the _cooling curve_, which is a tabulated set of values resulting from detailed molecular calculations.
Legolas has multiple cooling curves from existing literature implemented which are interpolated at high resolution and sampled on the given grid.
We have also included analytical, piecewise prescriptions (e.g. the `Rosner` curve). See the [physicslist](../../general/parameter_file/#physicslist) for an overview of the different options.
The function $\Lambda(T)$ can be user-specified, in which case its temperature derivative should be provided as well.

The function $\fancyH(\rho, T)$ specifies the heating function. If thermal balance is forced we assume that this term only depends on the equilibrium background, meaning that it is
constant in time but possibly varying in space and as such that it balances out the cooling contribution to ensure thermal equilibrium. $\fancyH(\rho, T)$ can be user-specified, in which case its density and
temperature derivatives should be provided as well.

#### Cooling curves
At the moment 8 tabulated cooling curves are implemented, along with 1 analytical cooling curve. The list below gives an overview of the cooling curves that are available, along with their references.

| Name    		 		| Notes   | Approximate $\log_{10}(T)$ range | Reference     |
| :---         		| :---    | :---: 		|   :---  		 |
| `Rosner`     		| Analytical cooling curve | $3.891 - 7.605$ | [Rosner et al. (1978)](https://ui.adsabs.harvard.edu/abs/1978ApJ...220..643R/abstract) |
| `JCcorona` 	  | Cooling curve for coronal conditions | $4.000 - 7.954$ | [Colgan et al. (2008)](https://ui.adsabs.harvard.edu/abs/2008ApJ...689..585C/abstract) |
| `DM`   	  | Cooling curve for low temperatures | $2.000 - 9.000$ | [Dalgarno & McCray (1972)](https://ui.adsabs.harvard.edu/abs/1972ARA%26A..10..375D/abstract) |
| `DM_2` 		| Cooling curve for very low temperatures | $1.000 - 4.000$ | [Dalgarno & McCray (1972)](https://ui.adsabs.harvard.edu/abs/1972ARA%26A..10..375D/abstract) |
| `MLsolar1`  		| Cooling curve for solar abundances | $2.000 - 9.000$ | [Mellema & Lundqvist (2002)](https://ui.adsabs.harvard.edu/abs/2002A%26A...394..901M/abstract) |
| `SPEX`			 		| Cooling curve for solar abundances | $3.800 - 8.160$ | [Schure et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009A%26A...508..751S/abstract) |
| `SPEX_DM` | `SPEX` supplemented with `DM` for lower temperatures | $2.000 - 8.160$ | - |
| `Colgan`     		| The original cooling curve | $4.065 - 9.065$ | [Colgan & Feldman (2008)](https://ui.adsabs.harvard.edu/abs/2008ApJ...689..585C/abstract) |
| `Colgan_DM`  		| `Colgan` supplemented with `DM_2` for lower temperatures | $1.000 - 9.065$ | - |

All cooling curves are interpolated at high resolution using a local second order polynomial and then sampled on the grid, with the exception of the analytical curves.
Note that all tabulated cooling curves are extended to higher temperatures (i.e. outside of their tabulated range) by assuming pure Brehmsstrahlung ($\Lambda(T) \sim \sqrt{T}$). For temperatures
below the tabulated range the cooling is set to zero.


### Thermal conduction
The thermal conduction prescription relies on a tensor representation to model the anisotropy in MHD and is given by

$$
\kappabf = \kappa_\parallel\unit{B}\unit{B} + \kappa_\bot(\boldsymbol{I} - \unit{B}\unit{B}),
$$

with $\boldsymbol{I}$ the unit tensor and $\unit{B} = \bbf / B$ a unit vector along the magnetic field. For the actual coefficients the Spitzer conductivity is assumed:

$$
\begin{align}
\kappa_\parallel &\approx 8 \times 10^{-7}T^{5/2}~\text{erg cm$^{-1}$s$^{-1}$K$^{-1}$},	\\
\kappa_\bot &\approx 4 \times 10^{-10} n^2B^{-2}T^{-3}\kappa_\parallel.
\end{align}
$$

Alternatively, both of these can be independently set to constant values.
In the absence of a magnetic field we set $\kappa_\bot = \kappa_\parallel$, such that the tensor representation reduces to $\kappabf = \kappa_\bot\boldsymbol{I}$.

When overriding parallel thermal conduction a function $\kappa_\parallel(T)$ can be given, as well as its temperature derivative. The derivative with respect to position (denoted with $'$) is automatically calculated as $\kappa_\parallel' = \frac{\partial \kappa_\parallel}{\partial T}T_0'$.

For perpendicular thermal conduction a function $\kappa_\bot(\rho, T, B)$ can be given, as well as its derivatives

$$
\frac{\partial \kappa_\bot}{\partial \rho}, \qquad
\frac{\partial \kappa_\bot}{\partial T}, \qquad
\frac{\partial \kappa_\bot}{\partial B^2}.
$$

The derivative with respect to position is automatically calculated as

$$ \kappa_\bot' =
\frac{\partial \kappa_\bot}{\partial T}T_0'
+ \frac{\partial \kappa_\bot}{\partial \rho}\rho_0'
+ \frac{\partial \kappa_\bot}{\partial B^2}2B_0B_0'.
$$

### Viscosity
The viscosity addition to the momentum equation is usually given in a full tensor form, but in Legolas this is simplified to good approximation to

$$
\mathbf{F}_{\mathrm{v}} = -\nabla\cdot\boldsymbol{\Pi} \simeq \mu \left[ \nabla^2\vbf + \frac{1}{3} \nabla\left( \nabla\cdot\vbf \right) \right],
$$

where the constant $\mu$ denotes the dynamic viscosity. The viscous heating term in the energy equation is expanded using a Frobenius norm and written as

$$
\left|\nabla \vbf\right|^2 = \sum_{i=1}^{3}\sum_{j=1}^{3}\left(\nabla \vbf\right)_{ij}^2
$$

The heating term can be toggled on or off independently of the main viscosity addition in the momentum equation,
see the [physicslist](../../general/parameter_file/#physicslist) for details on how to do this.


### Hall MHD
Legolas allows for the addition of Hall effects, including electron contributions:

$$
\underbrace{\frac{\eta_H}{\rho}(\nabla \times \bbf) \times \bbf}_\text{main Hall term}, \qquad\qquad
\underbrace{\frac{\eta_H}{\rho}\nabla p}_\text{electron pressure}, \qquad\qquad
\underbrace{\frac{\eta_e}{\rho}\frac{\partial (\nabla \times \bbf)}{\partial t}}_\text{electron inertia}.
$$

The main Hall contribution, electron pressure term and electron inertia term are treated as separate entities, meaning that
these terms can be toggled on or off independently of one another. See the [physicslist](../../general/parameter_file/#physicslist) for details.
