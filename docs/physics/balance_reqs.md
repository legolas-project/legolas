---
title: Equilibrium conditions
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: true
toc_icon: "chevron-circle-down"
last_modified_at: 2021-07-27
---

## Equilibrium state

When setting up an equilibrium yourself, Legolas allows for a time-independent background equilibrium
of the form

$$
\newcommand{\vbf}{\mathbf{v}}
\newcommand{\gbf}{\mathbf{g}}
\newcommand{\bbf}{\mathbf{B}}
\newcommand{\HL}{\mathscr{L}}
\newcommand{\fancyH}{\mathcal{H}}
\newcommand{\kappabf}{\boldsymbol{\kappa}}
\newcommand{\unit}[1]{\mathbf{e}_{#1}}
\newcommand{\eps}{\varepsilon}
\newcommand{\prefactkappazero}{\frac{\kappa_{\parallel,0} - \kappa_{\perp,0}}{B_0^2}}
\newcommand{\gmone}{(\gamma - 1)}

\begin{gather}
    \rho_0 = \rho_0(u_1),		\\
    T_0 = T_0(u_1),			\\
    \vbf_0 = v_{01}(u_1)\unit{1} + v_{02}(u_1)\unit{2} + v_{03}(u_1)\unit{3},	\\
    \bbf_0 = B_{01}\unit{1} + B_{02}(u_1)\unit{2} + B_{03}(u_1)\unit{3}. \\
\end{gather}
$$

where $(u_1, u_2, u_3)$ denotes the coordinate system: $(x, y, z)$ in Cartesian and $(r, \theta, z)$ in cylindrical
geometries.

## Equilibrium requirements
The combination of the equilibrium state and the full system of equations yields conditions that have
to be fulfilled. More specifically, every setup has to satisfy the force balance and thermal balance equations:

$$
\begin{gather}
    \left(\rho_0 T_0 + \frac{1}{2}B_{02}^2 + \frac{1}{2}B_{03}^2\right)' + \rho_0 g + \frac{\eps'}{\eps}\left(B_{02}^2 - \rho_0 v_{02}^2\right) = 0, \\
    \rho_0 \HL_0 - \frac{1}{\eps}\left(\eps \kappa_{\perp, 0} T_0'\right)' = 0, \\
\end{gather}
$$

where $\varepsilon = r$ and $\varepsilon' = 1$ in cylindrical geometries, and $\varepsilon = 1$ and $\varepsilon' = 0$
in Cartesian geometries. The prime denotes the derivative with respect to $u_1$.

Note that the above set of equations is actually a reduced form of the actual conditions. For typical use cases (where $v_{01} = B_{01} = 0$) these
are sufficient, however, for general equilibria ($v_{01} \neq 0, B_{01} \neq 0$) these expand to the following set:

$$
\begin{gather}
    \left(\eps v_{01}\right)'\rho_0 + \eps v_{01}\rho_0' = 0, \\
    \left(\rho_0 T_0 + \frac{1}{2}B_{02}^2 + \frac{1}{2}B_{03}^2\right)' + \rho_0\left(g + v_{01}v_{01}'\right) + \frac{\eps'}{\eps}\left(B_{02}^2 - \rho_0 v_{02}^2\right) = 0, \\
    \frac{B_{01}}{\eps}\left(\eps B_{02}\right)' - \rho_0 v_{01}\left(\frac{\eps'}{\eps}v_{02} + v_{02}'\right) = 0, \\
    B_{01}B_{03}' - \rho_0 v_{01}v_{03}' = 0, \\
    T_0 \rho_0 \frac{\left(\eps v_{01}\right)'}{\eps} + \rho_0 \HL_0 - B_{01}^2\left[\prefactkappazero T_0' \right]'
        - \frac{1}{\eps}\left(\eps \kappa_{\perp, 0} T_0'\right)' + \frac{1}{\gmone}T_0'\rho_0 v_{01} = 0, \\
    \left(B_{02}v_{01} - B_{01}v_{02}\right)' = 0, \\
    \Bigl[\eps \left(B_{01}v_{03} - B_{03}v_{01}\right)\Bigr]' = 0.
\end{gather}
$$

<i class="fas fa-lightbulb" aria-hidden="true"></i>
**Note:** There is no need to check these yourself. Legolas inspects at runtime if the appropriate equations are satisfied,
and warnings will be logged if one or more of these are not fulfilled. It will also tell you were exactly in the grid the largest
discrepancy occurs, what its value is and how many gridpoints fail to satisfy the conditions.
{: .notice--info}
