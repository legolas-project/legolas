! =============================================================================
!> This submodule defines a cylindrical equilibrium, resembling a
!! rotating plasma cylinder. The geometry can be overridden using the parfile.
!!
!! This equilibrium is taken from
!! _Nijboer, R. J., Holst, B., Poedts, S., & Goedbloed, J. P. (1997).
!!  Calculating magnetohydrodynamic flow spectra. Computer physics communications, 106(1-2), 39-52_.
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 1
!!     - <tt>k3</tt> = 0
!!     - <tt>cte_rho0</tt> = 1 : used to set the density value.
!!     - <tt>cte_p0</tt> = 0.1 : used to set the pressure.
!!     - <tt>p1</tt> = 8 : sets the constant a21.
!!     - <tt>p2</tt> = 0 : sets the constant a22.
!!     - <tt>p3</tt> = 0 : sets the constant a3.
!!     - <tt>p4</tt> = 1 : sets the constant b21.
!!     - <tt>p5</tt> = 0 : sets the constant b22.
!!     - <tt>p6</tt> = 0 : sets the constant b3.
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_rotating_plasma_cylinder
  use mod_equilibrium_params, only: cte_rho0, cte_p0, p1, p2, p3, p4, p5, p6
  implicit none

  real(dp)    :: a21, a22, a3, b21, b22, b3

contains

  module procedure rotating_plasma_cyl_eq
    if (settings%equilibrium%use_defaults) then  ! LCOV_EXCL_START
      call settings%grid%set_geometry("cylindrical")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      call settings%physics%enable_flow()

      a21 = 8.0_dp
      a22 = 0.0_dp
      a3  = 0.0_dp
      b21 = 1.0_dp
      b22 = 0.0_dp
      b3  = 0.0_dp
      cte_rho0 = 1.0
      cte_p0  = 0.1_dp

      k2  = 1.0_dp
      k3  = 0.0_dp
    else  ! LCOV_EXCL_STOP
      a21 = p1
      a22 = p2
      a3 = p3
      b21 = p4
      b22 = p5
      b3 = p6
    end if

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_velocity_2_funcs(v02_func=v02, dv02_func=dv02)
    call background%set_velocity_3_funcs(v03_func=v03)
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02)
    call background%set_magnetic_3_funcs(B03_func=B03)
  end procedure rotating_plasma_cyl_eq


  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function T0(r)
    real(dp), intent(in) :: r
    T0 = (1.0_dp / rho0()) * ( &
      cte_p0 + 0.5_dp * (a21**2 - 2.0_dp * b21**2) * r**2 &
      + (2.0_dp / 3.0_dp) * (a21 * a22 - b21 * b22) * r**3 &
      + (1.0_dp / 4.0_dp) * (a22**2 - b22**2) * r**4 &
    )
  end function T0

  real(dp) function dT0(r)
    real(dp), intent(in) :: r
    dT0 = (1.0_dp / rho0()) * ( &
      (a21**2 - 2.0_dp * b21**2) * r &
      + 2.0_dp * (a21 * a22 - b21 * b22) * r**2 &
      + (a22**2 - b22**2) * r**3 &
    )
  end function dT0

  real(dp) function v02(r)
    real(dp), intent(in) :: r
    v02 = a21 * r + a22 * r**2
  end function v02

  real(dp) function dv02(r)
    real(dp), intent(in) :: r
    dv02 = a21 + 2.0_dp * a22 * r
  end function dv02

  real(dp) function v03()
    v03 = a3
  end function v03

  real(dp) function B02(r)
    real(dp), intent(in) :: r
    B02 = b21*r + b22 * r**2
  end function B02

  real(dp) function dB02(r)
    real(dp), intent(in) :: r
    dB02 = b21 + 2.0_dp * b22 * r
  end function dB02

  real(dp) function B03()
    B03 = b3
  end function B03

end submodule smod_equil_rotating_plasma_cylinder
