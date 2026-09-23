! =============================================================================
!> This submodule defines Rayleigh-Taylor instabilities in rotating theta pinches.
!! The straight cylinder approximation is used with a constant angular frequency.
!! Density and pressure profiles decrease over the domain, with a uni-directional
!! increasing magnetic field profile. Mode numbers \(k = 0\) correspond to
!! HD Rayleigh-Taylor instabilities, while \( k \neq 0 \) represent MHD RTIs.
!! The geometry is hardcoded to <tt>'cylindrical'</tt>, the domain is forced to
!! \(0 <= r <= 1\) through division by <tt>x_end</tt>.
!!
!! This equilibrium is taken from section IV in
!! _Goedbloed, J. P. "The Spectral Web of stationary plasma equilibria.
!! II. Internal modes." Physics of Plasmas 25.3 (2018): 032110_.
!! and also appears in section 13.4, figs. 13.11 to 13.15 in
!! _Goedbloed, H., Keppens, R., & Poedts, S. (2019). Magnetohydrodynamics of Laboratory
!! and Astrophysical Plasmas. Cambridge University Press._
!! [DOI](http://doi.org/10.1017/9781316403679).
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 1
!!     - <tt>k3</tt> = 0 : so HD RTI
!!     - <tt>cte_rho0</tt> = 1 : maximum density value.
!!     - <tt>alpha</tt> = 2 : represents the stretching parameter.
!!     - <tt>delta</tt> = 1/6 : represents the magnetic field deviation parameter.
!!     - <tt>r0</tt> = 0 : represents the normalised radius \(x_0\) at maximum density.
!!     
!!     and can all be changed in the parfile.
!! @endnote
! SUBMODULE: smod_equil_rotating_theta_pinch
!
! DESCRIPTION:
! Submodule defining Rayleigh-Taylor instabilities in a cylindrical geometry.
! Obtained from Goedbloed, Phys. Plasmas 25, 032110 (2018), Fig. 9, 11
! Also appears in Magnetohydrodynamics (2019), Fig. 13.12, 13.14
submodule (mod_equilibrium) smod_equil_RTI_theta_pinch
  use mod_equilibrium_params, only: cte_rho0, cte_p0, alpha, delta, r0
  implicit none

  real(dp) :: width, B_inf, bigO

contains

  module procedure RTI_theta_pinch_eq
    call settings%physics%enable_flow()
    call settings%grid%set_geometry("cylindrical")

    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      cte_rho0 = 1.0_dp
      alpha = 2.0_dp
      delta = 1.0_dp / 6.0_dp
      r0 = 0.0_dp

      k2 = 1.0_dp
      k3 = 0.0_dp
    end if ! LCOV_EXCL_STOP

    width = settings%grid%get_grid_end() - settings%grid%get_grid_start()
    cte_p0 = 0.5_dp * (1.0_dp - delta)**2
    B_inf = width * sqrt(cte_rho0)
    bigO = alpha * sqrt(2.0_dp * delta * (1.0_dp - delta))

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_velocity_2_funcs(v02_func=v02, dv02_func=dv02)
    call background%set_temperature_funcs(T0_func=T0)
    call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03)
  end procedure RTI_theta_pinch_eq


  real(dp) function fx(r)
    real(dp), intent(in) :: r
    real(dp) :: x
    x = r / width
    fx = alpha**2 * (x**2 - r0**2)
  end function fx

  real(dp) function dfx(r)
    real(dp), intent(in) :: r
    real(dp) :: x
    x = r / width
    dfx = alpha**2 * 2.0_dp * x / width
  end function dfx

  real(dp) function rho0(r)
    real(dp), intent(in) :: r
    rho0 = cte_rho0 / cosh(fx(r))**2
  end function rho0

  real(dp) function drho0(r)
    real(dp), intent(in) :: r
    drho0 = -2.0_dp * cte_rho0 * dfx(r) * tanh(fx(r)) / cosh(fx(r))**2
  end function drho0

  real(dp) function T0()
    T0 = cte_p0 / cte_rho0
  end function T0

  real(dp) function v02(r)
    real(dp), intent(in) :: r
    v02 = bigO * r
  end function v02

  real(dp) function dv02()
    dv02 = bigO
  end function dv02

  real(dp) function B03(r)
    real(dp), intent(in) :: r
    B03 = B_inf * (delta + (1.0_dp - delta) * tanh(fx(r)))
  end function B03

  real(dp) function dB03(r)
    real(dp), intent(in) :: r
    dB03 = B_inf * (1.0_dp - delta) * dfx(r) / cosh(fx(r))**2
  end function dB03

end submodule smod_equil_RTI_theta_pinch
