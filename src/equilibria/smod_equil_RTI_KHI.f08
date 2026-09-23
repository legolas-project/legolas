! =============================================================================
!> This submodule defines Rayleigh-Taylor and Kelvin-Helmholtz instabilities in Cartesian geometry.
!! This equilibrium is a specific case of the flow driven instabilities.
!!
!!
!! This equilibrium is taken from section 13.2, p. 487 (case c) in
!! _Goedbloed, H., Keppens, R., & Poedts, S. (2019). Magnetohydrodynamics of Laboratory
!!  and Astrophysical Plasmas. Cambridge University Press._ [DOI](http://doi.org/10.1017/9781316403679).
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 0
!!     - <tt>k3</tt> = 1
!!     - <tt>cte_rho0</tt> = 1 : used in the density profile.
!!     - <tt>cte_p0</tt> = 1000 : used in the pressure profile, must be large so T0 > 0.
!!     - <tt>delta</tt> = -5 : used in the density profile.
!!     - <tt>g</tt> = 100 : gravitational constant.
!!     - <tt>alpha</tt> = -\(\pi\) : magnetic shear.
!!     - <tt>theta</tt> = 0 : angle used in the velocity profile.
!!     - <tt>p1</tt> = 1 : \(v_0\), used in the velocity profile.
!!     - <tt>p2</tt> = 2 : \(v_1\), used in the velocity profile.
!!     - <tt>p3</tt> = 1 : \(v_2\), used in the velocity profile.
!!     - <tt>p4</tt> = 0.5\(\pi\) : \(\phi_0\), used in the magnetic field profile.
!!     - <tt>tau</tt> = 4 : used in the velocity profile.
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_RTI_KHI
  implicit none

contains

  module procedure RTI_KHI_eq
    use mod_equilibrium_params, only: g, delta, theta, p1, p2, p3, tau, &
      p4, alpha, cte_rho0, cte_p0

    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%physics%enable_gravity()

      k2 = 0.0_dp
      k3 = 1.0_dp
      cte_rho0 = 1.0_dp
      cte_p0 = 1000.0_dp
      delta = -5.0_dp
      g = 100.0_dp
      alpha = -dpi
      theta = 0.0_dp
      p1 = 1.0_dp
      p2 = 2.0_dp
      p3 = 1.0_dp
      p4 = 0.5_dp * dpi
      tau = 4.0_dp
    end if ! LCOV_EXCL_STOP

    call flow_driven_instabilities_eq(settings, grid, background, physics, iv_initial_conditions)
  end procedure RTI_KHI_eq

end submodule smod_equil_RTI_KHI
