! =============================================================================
!> This submodule defines a steady Taylor-Couette flow in a cylindrical geometry
!! where a plasma is confined between two (rotating) coaxial cylinders
!! with an azimuthal magnetic field and constant resistivity.
!!
!! This equilibrium is taken from
!! _Shalybkov, Dima.
!! "Rotational stabilization of pinch instabilities in Taylor-Couette flow.",
!! Physical Review E 75, 047302 (2007)_.
!!
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 0
!!     - <tt>k3</tt> = 1
!!     - <tt>cte_rho0</tt> = 1 : density (constant)
!!     - <tt>alpha</tt> = 1 : rotational speed of the inner cylinder
!!     - <tt>beta</tt> = 2 : rotational speed of the outer cylinder
!!     - <tt>cte_B02</tt> = 1 : azimuthal magnetic field at inner cylinder
!!     - <tt>tau</tt> = 10 : azimuthal magnetic field at outer cylinder
!!     
!!     and can all be changed in the parfile.
!! @endnote
! SUBMODULE: smod_equil_tc_pinch
submodule (mod_equilibrium) smod_equil_tc_pinch
  use mod_equilibrium_params, only: cte_rho0, cte_B02, alpha, beta, tau
  implicit none

  real(dp) :: h, A, B, A2, B2, Bc1, Bc2, Bc3
  real(dp) :: Tstart, Tend
  real(dp) :: Ta, Ha, Re, Pm

contains

  module procedure tc_pinch_eq
    real(dp) :: x_start, x_end, r_mid
    real(dp) :: fixed_eta_value, viscosity_value

    call settings%physics%enable_flow()
    settings%grid%coaxial = .true.

    if (settings%equilibrium%use_defaults) then
      call settings%grid%set_geometry("cylindrical")
      call settings%grid%set_grid_boundaries(1.0_dp, 2.0_dp)
      cte_rho0 = 1.0e3_dp
      alpha = 1.0e-6_dp
      beta = 1.5e-6_dp
      cte_B02 = 1.0e-3_dp
      tau = 4.0e-3_dp

      k2 = 0.0_dp
      k3 = 1.0_dp

      call settings%physics%enable_resistivity(fixed_resistivity_value=1.0e-4_dp)
      call settings%physics%enable_viscosity(viscosity_value=1.0e-6_dp)
    end if
    x_start = settings%grid%get_grid_start()
    x_end = settings%grid%get_grid_end()

    fixed_eta_value = settings%physics%resistivity%get_fixed_resistivity()
    viscosity_value = settings%physics%viscosity%get_viscosity_value()

    h = x_end - x_start
    A = (beta * x_end**2 - alpha * x_start**2) / (x_end**2 - x_start**2)
    B = (x_start * x_end)**2 * (alpha - beta) / (x_end**2 - x_start**2)
    A2 = (x_end * tau - x_start * cte_B02) / (x_end**2 - x_start**2)
    B2 = x_start * x_end * (x_end * cte_B02 - x_start * tau) / (x_end**2 - x_start**2)

    Bc1 = cte_rho0 * A**2 - A2**2
    Bc2 = cte_rho0 * A * B - A2 * B2
    Bc3 = cte_rho0 * B**2 - B2**2

    Tstart = ( &
      0.5_dp * Bc1 * x_start**2 &
      + 2.0_dp * Bc2 * log(x_start) &
      - 0.5_dp * Bc3 / x_start**2 &
      - 0.5_dp * (A2 * x_start + B2 / x_start)**2 &
    ) / cte_rho0
    Tend = ( &
      0.5_dp * Bc1 * x_end**2 &
      + 2.0_dp * Bc2 * log(x_end) &
      - 0.5_dp * Bc3 / x_end**2 &
      - 0.5_dp * (A2 * x_end + B2 / x_end)**2 &
    ) / cte_rho0

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_velocity_2_funcs(v02_func=v02, dv02_func=dv02, ddv02_func=ddv02)
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02, ddB02_func=ddB02)

    r_mid = 0.5_dp * (x_start + x_end)
    Ta = ( &
      cte_rho0 * v02(r_mid) * h / viscosity_value &
    )**2 * 2.0_dp * h / (x_start + x_end)
    call logger%info("Taylor number  :  " // str(Ta, fmt=exp_fmt))
    Pm = viscosity_value / (cte_rho0 * fixed_eta_value)
    call logger%info("Prandtl number : " // str(Pm, fmt=exp_fmt))
    Ha = cte_B02 * sqrt(x_start * (x_end - x_start)) &
      / sqrt(viscosity_value * fixed_eta_value)
    call logger%info("Hartmann number: " // str(Ha, fmt=exp_fmt))
    Re = alpha * cte_rho0 * x_start * (x_end - x_start) / viscosity_value
    call logger%info("Reynolds number: " // str(Re, fmt=exp_fmt))
  end procedure tc_pinch_eq


  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function v02(r)
    real(dp), intent(in) :: r
    v02 = A * r + B / r
  end function v02

  real(dp) function dv02(r)
    real(dp), intent(in) :: r
    dv02 = A - B / r**2
  end function dv02

  real(dp) function ddv02(r)
    real(dp), intent(in) :: r
    ddv02 = 2.0_dp * B / r**3
  end function ddv02

  real(dp) function T0(r)
    real(dp), intent(in) :: r
    T0 = ( &
      0.5_dp * Bc1 * r**2 &
      + 2.0_dp * Bc2 * log(r) &
      - 0.5_dp * Bc3 / r**2 &
      - 0.5_dp * B02(r)**2 &
    ) / cte_rho0
    if (.not. (Tstart > 0.0_dp .and. Tend > 0.0_dp)) then
      T0 = T0 + 2.0_dp * max(abs(Tstart), abs(Tend))
    end if
  end function T0

  real(dp) function dT0(r)
    real(dp), intent(in) :: r
    dT0 = ( &
      (cte_rho0 * v02(r)**2 - B02(r)**2) / r - B02(r) * dB02(r) &
    ) / cte_rho0

  end function dT0

  real(dp) function B02(r)
    real(dp), intent(in) :: r
    B02 = A2 * r + B2 / r
  end function B02

  real(dp) function dB02(r)
    real(dp), intent(in) :: r
    dB02 = A2 - B2 / r**2
  end function dB02

  real(dp) function ddB02(r)
    real(dp), intent(in) :: r
    ddB02 = 2.0_dp * B2 / r**3
  end function ddB02


end submodule smod_equil_tc_pinch
