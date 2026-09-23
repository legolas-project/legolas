! =============================================================================
!> This submodule defines Kelvin-Helmholtz instabilities in Cartesian geometry.
!! This equilibrium is a specific case of the flow driven instabilities.
!! No magnetic fields are considered in this case (pure HD).
!!
!!
!! This equilibrium is taken from section 13.2, p. 487 (case b) in
!! _Goedbloed, H., Keppens, R., & Poedts, S. (2019). Magnetohydrodynamics of Laboratory
!!  and Astrophysical Plasmas. Cambridge University Press._ [DOI](http://doi.org/10.1017/9781316403679).
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 0
!!     - <tt>k3</tt> = 1
!!     - <tt>cte_rho0</tt> = 1 : used in the density profile.
!!     - <tt>cte_p0</tt> = 10 : used in the pressure profile.
!!     - <tt>delta</tt> = 0 : used in the density profile.
!!     - <tt>g</tt> = 0 : gravitational constant.
!!     - <tt>alpha</tt> = 0 : magnetic shear.
!!     - <tt>theta</tt> = 0 : angle used in the velocity profile.
!!     - <tt>p1</tt> = 0 : \(v_0\), used in the velocity profile.
!!     - <tt>p2</tt> = 0 : \(v_1\), used in the velocity profile.
!!     - <tt>p3</tt> = 1 : \(v_2\), used in the velocity profile.
!!     - <tt>p4</tt> = 0 : \(\phi_0\), used in the magnetic field profile.
!!     - <tt>tau</tt> = 11 : used in the velocity profile.
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_KHI
  use mod_function_utils, only: zero_func
  implicit none

contains

  module procedure KHI_eq
    use mod_equilibrium_params, only: g, delta, theta, p1, p2, p3, tau, &
      p4, alpha, cte_rho0, cte_p0

    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      k2 = 0.0_dp
      k3 = 1.0_dp
      cte_rho0 = 1.0_dp
      cte_p0 = 1000_dp
      delta = 0.0_dp
      g = 0.0_dp
      alpha = 0.0_dp
      theta = 0.0_dp
      p1 = 0.0_dp
      p2 = 0.0_dp
      p3 = 1.0_dp
      p4 = 0.0_dp
      tau = 11.0_dp
    end if ! LCOV_EXCL_STOP

    call flow_driven_instabilities_eq(settings, grid, background, physics, iv_initial_conditions)
    ! manually force the magnetic field to zero, it's HD here (the above set
    ! of parameters still yield a B03 component)
    call background%set_magnetic_2_funcs(B02_func=zero_func, dB02_func=zero_func)
    call background%set_magnetic_3_funcs(B03_func=zero_func, dB03_func=zero_func)
  end procedure KHI_eq

end submodule smod_equil_KHI
