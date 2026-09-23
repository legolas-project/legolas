! =============================================================================
!> This submodule defines an equilibrium in Cartesian geometry with a
!! flow profile and a constant resistivity value. Parameters are taken in such a way
!! as to allow for resistive tearing modes. The geometry can be overridden using the parfile.
!!
!! This equilibrium is taken from section 14.3, p. 553 in
!! _Goedbloed, H., Keppens, R., & Poedts, S. (2019). Magnetohydrodynamics of Laboratory
!!  and Astrophysical Plasmas. Cambridge University Press._ [DOI](http://doi.org/10.1017/9781316403679).
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 1.5
!!     - <tt>k3</tt> = 0
!!     - <tt>cte_rho0</tt> = 1 : used to set the density value.
!!     - <tt>alpha</tt> = 4.73884 : parameter in the magnetic field prescription.
!!     - <tt>beta</tt> = 0.15 : used to constrain the temperature value.
!!     - fixed resistivity value of 0.0001
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_resistive_tearing_flow
  use mod_equilibrium_params, only: alpha, beta, cte_rho0
  implicit none

contains

  module procedure resistive_tearing_modes_flow_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("Cartesian")
      call settings%grid%set_grid_boundaries(-0.5_dp, 0.5_dp)
      call settings%physics%enable_flow()
      call settings%physics%enable_resistivity(fixed_resistivity_value=0.0001_dp)

      k2 = 1.5_dp
      k3 = 0.0_dp

      alpha = 4.73884_dp
      beta = 0.15_dp
      cte_rho0 = 1.0_dp
    end if ! LCOV_EXCL_STOP

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_velocity_2_funcs(v02_func=v02, dv02_func=dv02)
    call background%set_temperature_funcs(T0_func=T0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02, ddB02_func=ddB02)
    call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03, ddB03_func=ddB03)
  end procedure resistive_tearing_modes_flow_eq


  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function T0(x)
    real(dp), intent(in) :: x
    T0 = beta * (B0(x))**2 / (2.0_dp * cte_rho0)
  end function T0

  real(dp) function v02(x)
    real(dp), intent(in) :: x
    v02 = 0.15_dp * x
  end function v02

  real(dp) function dv02()
    dv02 = 0.15_dp
  end function dv02

  real(dp) function B02(x)
    real(dp), intent(in) :: x
    B02 = sin(alpha * x)
  end function B02

  real(dp) function dB02(x)
    real(dp), intent(in) :: x
    dB02 = alpha * cos(alpha * x)
  end function dB02

  real(dp) function ddB02(x)
    real(dp), intent(in) :: x
    ddB02 = -alpha**2 * sin(alpha * x)
  end function ddB02

  real(dp) function B03(x)
    real(dp), intent(in) :: x
    B03 = cos(alpha * x)
  end function B03

  real(dp) function dB03(x)
    real(dp), intent(in) :: x
    dB03 = -alpha * sin(alpha * x)
  end function dB03

  real(dp) function ddB03(x)
    real(dp), intent(in) :: x
    ddB03 = -alpha**2 * cos(alpha * x)
  end function ddB03

  real(dp) function B0(x)
    real(dp), intent(in) :: x
    B0 = sqrt(B02(x)**2 + B03(x)**2)
  end function B0

end submodule smod_equil_resistive_tearing_flow
