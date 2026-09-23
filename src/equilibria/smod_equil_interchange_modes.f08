! =============================================================================
!> This submodule defines an exponentially stratified medium in
!! Cartesian geometry with a constant gravity term and magnetic shear.
!! The geometry can be overridden in the parfile.
!!
!! This equilibrium is taken from section 12.1.3 in
!! _Goedbloed, H., Keppens, R., & Poedts, S. (2019). Magnetohydrodynamics of Laboratory
!!  and Astrophysical Plasmas. Cambridge University Press._ [DOI](http://doi.org/10.1017/9781316403679).
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = \(\pi\)
!!     - <tt>k3</tt> = \(\pi\)
!!     - <tt>cte_p0</tt> = 0.25 : pressure, used to set the plasma beta.
!!     - <tt>g</tt> = 0.5 : gravitational constant.
!!     - <tt>lambda</tt> = 0 : magnetic shear value.
!!     - <tt>alpha</tt> = 20 : constant to constrain the density value.
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_interchange_modes
  use mod_equilibrium_params, only: g, cte_rho0, cte_p0, alpha, beta, lambda
  implicit none

  real(dp) :: B0

contains

  !> Sets the equilibrium.
  module procedure interchange_modes_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("Cartesian")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      call settings%physics%enable_gravity()

      k2 = dpi
      k3 = dpi

      cte_p0 = 0.25_dp
      g = 0.5_dp
      lambda = 0.0_dp
      alpha = 20.0_dp
    end if ! LCOV_EXCL_STOP

    B0 = 1.0_dp
    beta = 2.0_dp*cte_p0 / B0**2
    cte_rho0 = (alpha / g) * (cte_p0 + 0.5_dp * B0**2)

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_temperature_funcs(T0_func=T0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02)
    call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03)

    call physics%set_gravity_funcs(g0_func=g0)
  end procedure interchange_modes_eq


  real(dp) function rho0(x)
    real(dp), intent(in) :: x
    rho0 = cte_rho0 * exp(-alpha*x)
  end function rho0

  real(dp) function drho0(x)
    real(dp), intent(in) :: x
    drho0 = -alpha * rho0(x)
  end function drho0

  real(dp) function T0()
    T0 = cte_p0 / cte_rho0
  end function T0

  real(dp) function B02(x)
    real(dp), intent(in) :: x
    B02 = B0 * exp(-0.5_dp * alpha * x) * sin(lambda * x)
  end function B02

  real(dp) function dB02(x)
    real(dp), intent(in) :: x
    dB02 = -0.5_dp * alpha * B02(x) + lambda * B03(x)
  end function dB02

  real(dp) function B03(x)
    real(dp), intent(in) :: x
    B03 = B0 * exp(-0.5_dp * alpha * x) * cos(lambda * x)
  end function B03

  real(dp) function dB03(x)
    real(dp), intent(in) :: x
    dB03 = -0.5_dp * alpha * B03(x) - lambda * B02(x)
  end function dB03

  real(dp) function g0()
  g0 = g
end function g0

end submodule smod_equil_interchange_modes
