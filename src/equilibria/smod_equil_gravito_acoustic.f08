! =============================================================================
!> This submodule defines an equilibrium in Cartesian geometry with a
!! stratified equilibrium profile, giving rise to gravito-acoustic waves.
!! No magnetic fields are included, such that this treats the hydrodynamic regime.
!! The geometry can be overridden using the parfile.
!!
!! This equilibrium is taken from section 7.2.3, p. 242 in
!! _Goedbloed, H., Keppens, R., & Poedts, S. (2019). Magnetohydrodynamics of Laboratory
!!  and Astrophysical Plasmas. Cambridge University Press._ [DOI](http://doi.org/10.1017/9781316403679).
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = \(\pi\)
!!     - <tt>k3</tt> = \(\pi\)
!!     - <tt>cte_p0</tt> = 1 : used to set the pressure value.
!!     - <tt>alpha</tt> = 20.42 : used to constrain the density.
!!     - <tt>g</tt> = 0.5 : used to set the gravity constant.
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_gravito_acoustic
  use mod_equilibrium_params, only: g, cte_rho0, cte_p0, alpha
  implicit none

contains

  module procedure gravito_acoustic_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("Cartesian")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      call settings%physics%enable_gravity()

      k2 = dpi
      k3 = dpi
      cte_p0 = 1.0_dp
      alpha = 20.42_dp
      g = 0.5_dp
    end if ! LCOV_EXCL_STOP

    cte_rho0 = alpha * cte_p0 / g

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_temperature_funcs(T0_func=T0)

    call physics%set_gravity_funcs(g0_func=g0)
  end procedure gravito_acoustic_eq


  real(dp) function rho0(x)
    real(dp), intent(in) :: x
    rho0 = cte_rho0 * exp(-alpha * x)
  end function rho0

  real(dp) function drho0(x)
    real(dp), intent(in) :: x
    drho0 = -alpha * (rho0(x))
  end function drho0

  real(dp) function T0()
    T0 = cte_p0 / cte_rho0
  end function T0


  real(dp) function g0()
    g0 = g
  end function g0

end submodule smod_equil_gravito_acoustic
