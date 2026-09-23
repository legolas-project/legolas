---
title: Unit normalisations
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
last_modified_at: 2026-04-21
---

All equations in Legolas are in dimensionless form, as is common practice when dealing with (M)HD.
As usual we have **three** degrees of freedom.

## Normalisations
Legolas has three options to specify units, all in cgs. In what follows $m_p$ denotes the proton mass,
$k_B$ the Boltzmann constant, and $\mu_0 = 4\pi$ the magnetic constant. $a$ and $b$ are constants
determined by the plasma composition. By default they depend only on the He abundance $f_\mathrm{He}$ as

$$
a = 1 + 4 f_\mathrm{He}, \quad
b = 2 + 3 f_\mathrm{He},
$$

such that $f_\mathrm{He} = 0$ corresponds to a fully ionised hydrogen plasma. The He abundance is set to
$f_\mathrm{He} = 0$ if not specified.

<i class="fas fa-lightbulb" aria-hidden="true"></i>
**Note:** for alternative plasma compositions, users can define $a$ and $b$ in the user module
and update the units from there, see e.g. the pre-implemented equilibrium
`smod_equil_magnetothermal_instabilities.f08`.
{: .notice--info}

1. Reference unit density, unit magnetic field and unit length $(\rho_u, B_u, L_u)$, then

   $$
   p_u = \frac{B_u^2}{\mu_0}, \quad
   T_u = \frac{p_u}{b n_u k_B}, \quad
   n_u = \frac{\rho_u}{a m_p}.
   $$

2. Reference unit temperature, unit magnetic field and unit length $(T_u, B_u, L_u)$, then

   $$
   p_u = \frac{B_u^2}{\mu_0}, \quad
   n_u = \frac{p_u}{b k_B T_u}, \quad
   \rho_u = a m_p n_u.
   $$

3. Reference unit number density, unit temperature and unit length $(n_u, T_u, L_u)$, then

   $$
   p_u = b n_u k_B T_u, \quad
   \rho_u = a m_p n_u, \quad
   B_u = \sqrt{\mu_0 p_u}.
   $$

All other normalisations follow from those above and are given by
- unit velocity: $v_u = \sqrt{\frac{p_u}{\rho_u}}$
- unit mass: $M_u = \rho_u L_u^3$
- unit time: $t_u = \dfrac{L_u}{v_u}$
- unit resistivity: $\eta_u = \dfrac{L_u^2}{t_u}$
- unit cooling curve: $\Lambda_u = \dfrac{p_u}{t_u n_u^2}$
- unit conduction: $\kappa_u = \dfrac{\rho_u L_u v_u^3}{T_u}$


<i class="fas fa-lightbulb" aria-hidden="true"></i>
**Note:** the unit normalisations are only relevant when radiative cooling, thermal conduction or temperature-dependent resistivity is included.
We always set base values though (as one should), which are set using option 2. with default values
$B_u = 10$ G, $L_u = 10^9$ cm and $T_u = 10^6$ K.
{: .notice--info}
