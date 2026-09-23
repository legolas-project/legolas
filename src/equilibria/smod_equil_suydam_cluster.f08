! =============================================================================
!> This submodule defines a cylindrical equilibrium in which a Suydam surface
!! is present, such that this gives rises to Suydam cluster modes.
!! The geometry can be overriden using the parfile.
!!
!! This equilibrium is taken from
!! _Nijboer, R. J., Holst, B., Poedts, S., & Goedbloed, J. P. (1997).
!!  Calculating magnetohydrodynamic flow spectra. Computer physics communications, 106(1-2), 39-52_.
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 1
!!     - <tt>k3</tt> = -1.2
!!     - <tt>cte_rho0</tt> = 1 : used to set the density value.
!!     - <tt>cte_p0</tt> = 0.05 : used to set the background pressure.
!!     - <tt>cte_v02</tt> = 0 : sets a constant flow \(v_\theta\).
!!     - <tt>cte_v03</tt> = 0.14 : prefactor for flow profile \(v_z\).
!!     - <tt>p1</tt> = 0.1 : constant that appears in magnetic field and pressure components.
!!     - <tt>alpha</tt> = 2 : used in the Bessel functions.
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_suydam_cluster
  use mod_equilibrium_params, only: cte_rho0, cte_v02, cte_v03, cte_p0, p1, alpha
  implicit none

contains

  module procedure suydam_cluster_eq
    if (settings%equilibrium%use_defaults) then  ! LCOV_EXCL_START
      call settings%grid%set_geometry("cylindrical")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      call settings%physics%enable_flow()

      cte_rho0 = 1.0_dp
      cte_v02 = 0.0_dp
      cte_v03 = 0.14_dp
      cte_p0 = 0.05_dp
      p1 = 0.1_dp
      alpha = 2.0_dp

      k2 = 1.0_dp
      k3 = -1.2_dp
    end if ! LCOV_EXCL_STOP

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_velocity_2_funcs(v02_func=v02)
    call background%set_velocity_3_funcs(v03_func=v03, dv03_func=dv03)
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02)
    call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03)
  end procedure suydam_cluster_eq


  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function T0(r)
    real(dp), intent(in) :: r
    T0 = (cte_p0 + 0.5_dp * p1 * J0(r)**2) / cte_rho0
  end function T0

  real(dp) function dT0(r)
    real(dp), intent(in) :: r
    dT0 = p1 * J0(r) * DJ0(r) / cte_rho0
  end function dT0

  real(dp) function v02()
    v02 = cte_v02
  end function v02

  real(dp) function v03(r)
    real(dp), intent(in) :: r
    v03 = cte_v03 * (1.0_dp - r**2)
  end function v03

  real(dp) function dv03(r)
    real(dp), intent(in) :: r
    dv03 = -2.0_dp * cte_v03 * r
  end function dv03

  real(dp) function B02(r)
    real(dp), intent(in) :: r
    B02 = J1(r)
  end function B02

  real(dp) function dB02(r)
    real(dp), intent(in) :: r
    dB02 = DJ1(r)
  end function dB02

  real(dp) function B03(r)
    real(dp), intent(in) :: r
    B03 = sqrt(1.0_dp - p1) * J0(r)
  end function B03

  real(dp) function dB03(r)
    real(dp), intent(in) :: r
    dB03 = -alpha * sqrt(1.0_dp - p1) * J1(r)
  end function dB03

  real(dp) function J0(r)
    real(dp), intent(in) :: r
    J0 = bessel_jn(0, alpha * r)
  end function J0

  real(dp) function J1(r)
    real(dp), intent(in) :: r
    J1 = bessel_jn(1, alpha * r)
  end function J1

  real(dp) function DJ0(r)
    real(dp), intent(in) :: r
    DJ0 = -alpha * J1(r)
  end function DJ0

  real(dp) function DJ1(r)
    real(dp), intent(in) :: r
    DJ1 = alpha * (0.5_dp * J0(r) - 0.5_dp * bessel_jn(2, alpha * r))
  end function DJ1

end submodule smod_equil_suydam_cluster
