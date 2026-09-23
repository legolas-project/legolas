! =============================================================================
!> This submodule defines an equilibrium in Cartesian geometry with a
!! stratified equilibrium profile, representing a solar magnetic atmosphere.
!! The equilibrium is isothermal with a constant magnetic field.
!! The scale height is given by
!! $$ H = \frac{\text{cte_T0}}{g}  $$
!! The geometry is fixed to Cartesian, boundaries can be overridden using the parfile.
!!
!! This equilibrium is taken from
!! _Nye, A., & Thomas, J. (1976). Solar magneto-atmospheric waves. I. An exact solution
!! for a horizontal magnetic field. The Astrophysical Journal, 204._
!! [_573-581_](http://articles.adsabs.harvard.edu/pdf/1976ApJ...204..573N).
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 0
!!     - <tt>k3</tt> = 2
!!     - <tt>cte_rho0</tt> = 1 : used as a density prefactor.
!!     - <tt>cte_B02</tt> = 0.25 : used to set the By-component
!!     - <tt>cte_B03</tt> = 0.25 : used to set the Bz-component
!!     - <tt>cte_T0</tt> = 1 : used to set the temperature (isothermal).
!!     - <tt>g</tt> = 5 : used to set the gravity constant.
!!     
!!     and can all be changed in the parfile.
!! @endnote
!!
!! @warning
!!     This equilibrium has no regression test yet!
!! @endwarning
submodule (mod_equilibrium) smod_equil_isothermal_atmosphere
  use mod_equilibrium_params, only: cte_rho0, cte_B02, cte_B03, cte_T0, g
  implicit none

  real(dp) :: scale_height

contains

  module procedure isothermal_atmosphere_eq
    if (settings%equilibrium%use_defaults) then  ! LCOV_EXCL_START
      call settings%grid%set_geometry("Cartesian")
      call settings%grid%set_grid_boundaries(0.0_dp, 15.0_dp)
      call settings%physics%enable_gravity()
      cte_rho0 = 1.0_dp
      cte_B02 = 0.25_dp
      cte_B03 = 0.25_dp
      cte_T0 = 1.0_dp
      g = 5.0_dp

      k2 = 0.0_dp
      k3 = 2.0_dp
    end if  ! LCOV_EXCL_STOP

    scale_height = cte_T0 / g

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_temperature_funcs(T0_func=T0)
    call background%set_magnetic_2_funcs(B02_func=B02)
    call background%set_magnetic_3_funcs(B03_func=B03)

    call physics%set_gravity_funcs(g0_func=g0)
  end procedure isothermal_atmosphere_eq


  real(dp) function rho0(x)
    real(dp), intent(in) :: x
    rho0 = cte_rho0 * exp(-x / scale_height)
  end function rho0

  real(dp) function drho0(x)
    real(dp), intent(in) :: x
    drho0 = -cte_rho0 * exp(-x / scale_height) / scale_height
  end function drho0

  real(dp) function T0()
    T0 = cte_T0
  end function T0

  real(dp) function B02()
    B02 = cte_B02
  end function B02

  real(dp) function B03()
    B03 = cte_B03
  end function B03

  real(dp) function g0(x, settings, background)
    real(dp), intent(in) :: x
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background
    g0 = g
  end function g0

end submodule smod_equil_isothermal_atmosphere
