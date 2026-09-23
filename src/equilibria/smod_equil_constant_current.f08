! =============================================================================
!> This submodule defines an equilibrium in cylindrical geometry with
!! a constant axial current. The geometry can be overridden using the parfile.
!!
!! This equilibrium is taken from
!! _Kerner, W. (1989). Large-scale complex eigenvalue problems.
!!  Journal of Computational Physics, 85(1), 1-85_.
!!
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = -2
!!     - <tt>k3</tt> = 0.2
!!     - <tt>j0</tt> = 0.125 : used to set the current.
!!     - <tt>cte_rho0</tt> = 1 : used to set the density.
!!     - <tt>cte_B03</tt> = 1 : used to set the Bz value.
!!    
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_constant_current
  use mod_equilibrium_params, only: j0, cte_rho0, cte_B03
  implicit none

contains

!> Sets the equilibrium.
  module procedure constant_current_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("cylindrical")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      k2 = -2.0_dp
      k3 = 0.2_dp

      j0 = 0.125_dp
      cte_rho0 = 1.0_dp
      cte_B03 = 1.0_dp
    end if ! LCOV_EXCL_STOP

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02)
    call background%set_magnetic_3_funcs(B03_func=B03)
  end procedure constant_current_eq

  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function T0(r)
    real(dp), intent(in) :: r
    T0 = p0(r) / rho0()
  end function T0

  real(dp) function dT0(r)
    real(dp), intent(in) :: r
    dT0 = dp0(r) / rho0()
  end function dT0

  real(dp) function p0(r)
    real(dp), intent(in) :: r
    p0 = 0.25_dp * j0**2 * (1.0_dp - r**2)
  end function p0

  real(dp) function dp0(r)
    real(dp), intent(in) :: r
    dp0 = -0.5_dp * j0**2 * r
  end function dp0

  real(dp) function B02(r)
    real(dp), intent(in) :: r
    B02 = 0.5_dp * j0 * r
  end function B02

  real(dp) function dB02()
    dB02 = 0.5_dp * j0
  end function dB02

  real(dp) function B03()
    B03 = cte_B03
  end function B03

end submodule smod_equil_constant_current
