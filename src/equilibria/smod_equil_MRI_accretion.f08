! =============================================================================
!> This submodule defines magneto-rotational instabilities in an accretion disk.
!! Due to the special nature of this equilibrium <tt>x_start</tt> is hardcoded
!! to one and can not be overridden in the parfile, the same goes for the geometry
!! which is hardcoded to <tt>'cylindrical'</tt>. The outer edge can be chosen freely.
!! This equilibrium is chosen in such a way that the angular rotation is of order unity,
!! implying Keplerian rotation. The thin-disk approximation is valid with small magnetic
!! fields, but still large enough to yield magneto-rotational instabilities.
!! Gravity is assumed to go like \(g \thicksim 1/r^2\).
!!
!! This equilibrium is taken from section V in
!! _Goedbloed, J. P. "The Spectral Web of stationary plasma equilibria.
!! II. Internal modes." Physics of Plasmas 25.3 (2018): 032110_.
!! and also appears in section 13.5, fig. 13.7 in
!! _Goedbloed, H., Keppens, R., & Poedts, S. (2019). Magnetohydrodynamics of Laboratory
!! and Astrophysical Plasmas. Cambridge University Press._
!! [DOI](http://doi.org/10.1017/9781316403679).
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 0
!!     - <tt>k3</tt> = 70
!!     - <tt>beta</tt> = 100 : parameter \(\beta = 2p_1/B_1^2\).
!!     - <tt>tau</tt> = 1 : represents parameter \(\tau = \mu_1 = B_{\theta 1}/B_{z1}\)
!!     - <tt>nu</tt> = 0.1 : represents parameter \(\nu = \epsilon = \sqrt{p_1}\)
!!     - <tt>x_end</tt> = 2 : fixes the parameter \(\delta = x_{end}/x_{start}\)
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_MRI
  use mod_equilibrium_params, only: beta, tau, nu
  implicit none

  real(dp) :: Bth1, Bz1, p1, delta, epsilon, mu1, vth1

contains

  module procedure MRI_accretion_eq
    call settings%grid%set_geometry("cylindrical")

    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_grid_boundaries(1.0_dp, 2.0_dp)

      call settings%physics%enable_flow()
      call settings%physics%enable_gravity()
      k2 = 0.0_dp
      k3 = 70.0_dp
      beta = 100.0_dp
      tau = 1.0_dp
      nu = 0.1_dp
    end if ! LCOV_EXCL_STOP

    mu1 = tau
    epsilon = nu

    delta = settings%grid%get_grid_end() / settings%grid%get_grid_start()
    p1 = epsilon**2
    Bz1 = sqrt(2.0_dp * p1 / (beta * (1.0_dp + mu1**2)))
    Bth1 = mu1 * Bz1
    vth1 = sqrt(1.0_dp - 2.5_dp * p1 - 0.25_dp * Bth1**2 - 1.25_dp * Bz1**2)

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_velocity_2_funcs(v02_func=v02, dv02_func=dv02)
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02)
    call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03)

    call physics%set_gravity_funcs(g0_func=g0)
  end procedure MRI_accretion_eq


  real(dp) function rho0(r)
    real(dp), intent(in) :: r
    rho0 = r**(-1.5_dp)
  end function rho0

  real(dp) function drho0(r)
    real(dp), intent(in) :: r
    drho0 = -1.5_dp * r**(-2.5_dp)
  end function drho0

  real(dp) function T0(r)
    real(dp), intent(in) :: r
    T0 = p0(r) / rho0(r)
  end function T0

  real(dp) function dT0(r)
    real(dp), intent(in) :: r
    dT0 = (dp0(r) * rho0(r) - drho0(r) * p0(r)) / rho0(r)**2
  end function dT0

  real(dp) function p0(r)
    real(dp), intent(in) :: r
    p0 = p1 * r**(-2.5_dp)
  end function p0

  real(dp) function dp0(r)
    real(dp), intent(in) :: r
    dp0 = -2.5_dp * p1 * r**(-3.5_dp)
  end function dp0

  real(dp) function v02(r)
    real(dp), intent(in) :: r
    v02 = vth1 / sqrt(r)
  end function v02

  real(dp) function dv02(r)
    real(dp), intent(in) :: r
    dv02 = -0.5_dp * vth1 * r**(-1.5_dp)
  end function dv02

  real(dp) function B02(r)
    real(dp), intent(in) :: r
    B02 = Bth1 * r**(-1.25_dp)
  end function B02

  real(dp) function dB02(r)
    real(dp), intent(in) :: r
    dB02 = -1.25_dp * Bth1 * r**(-2.25_dp)
  end function dB02

  real(dp) function B03(r)
    real(dp), intent(in) :: r
    B03 = Bz1 * r**(-1.25_dp)
  end function B03

  real(dp) function dB03(r)
    real(dp), intent(in) :: r
    dB03 = -1.25_dp * Bz1 * r**(-2.25_dp)
  end function dB03

  real(dp) function g0(x)
    real(dp), intent(in) :: x
    g0 = 1.0_dp / x**2
  end function g0

end submodule smod_equil_MRI
