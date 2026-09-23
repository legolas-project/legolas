! =============================================================================
!> This submodule defines a resistive equilibrium with tearing modes created
!! by a Harris sheet.
!!
!! This equilibrium is taken from Shi et al. (2020), Oblique tearing mode
!! instability: guide field and Hall effect. _The Astrophysical Journal_, 902:142.
!! [DOI](https://doi.org/10.3847/1538-4357/abb6fa).
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 0.12
!!     - <tt>k3</tt> = 0
!!     - <tt>cte_rho0</tt> = 1
!!     - <tt>cte_T0</tt> = 1
!!     - <tt>cte_B02</tt> = 1
!!     - <tt>cte_B03</tt> = 0 : guide field parameter.
!!     - <tt>alpha</tt> = 1 : used to set the width of the current sheet.
!!     - <tt>eq_bool</tt> = False : if True, an alternative force-free Harris sheet is used.
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_harris_sheet
  use mod_equilibrium_params, only: cte_rho0, cte_B02, cte_B03, cte_T0, alpha, eq_bool
  implicit none

contains

  module procedure harris_sheet_eq
    if (settings%equilibrium%use_defaults) then
      call settings%grid%set_geometry("Cartesian")
      call settings%grid%set_grid_boundaries(-15.0_dp, 15.0_dp)
      call settings%physics%enable_resistivity(fixed_resistivity_value=0.001_dp)

      k2 = 0.12_dp
      k3 = 0.0_dp

      alpha = 1.0_dp

      cte_rho0 = 1.0_dp
      cte_B02 = 1.0_dp
      cte_B03 = 0.0_dp
      cte_T0 = 1.0_dp

      !> eq_bool >> if True, the alternative force-free Harris sheet is used
      eq_bool = .false.
    end if

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02, ddB02_func=ddB02)
    call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03, ddB03_func=ddB03)
  end procedure harris_sheet_eq


  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function T0(x)
    real(dp), intent(in) :: x
    if (eq_bool) then
      T0 = cte_T0
    else
      T0 = (cte_B03**2 + cte_B02**2 - B0(x)**2) / (2.0_dp * cte_rho0)
    end if
  end function T0

  real(dp) function dT0(x)
    real(dp), intent(in) :: x
    if (eq_bool) then
      dT0 = 0.0_dp
    else
      dT0 = -cte_B02**2 * sinh(x / alpha) / (alpha * cte_rho0 * cosh(x / alpha)**3)
    end if
  end function dT0

  real(dp) function B02(x)
    real(dp), intent(in) :: x
    B02 = cte_B02 * tanh(x / alpha)
  end function B02

  real(dp) function dB02(x)
    real(dp), intent(in) :: x
    dB02 = cte_B02 / (alpha * cosh(x / alpha)**2)
  end function dB02

  real(dp) function ddB02(x)
    real(dp), intent(in) :: x
    ddB02 = - 2.0_dp * cte_B02 * sinh(x / alpha) / (alpha**2 * cosh(x / alpha)**3)
  end function ddB02

  real(dp) function B03(x)
    real(dp), intent(in) :: x
    if (eq_bool) then
      B03 = sqrt(cte_B03**2 + cte_B02**2 / cosh(x / alpha)**2)
    else
      B03 = cte_B03
    end if
  end function B03

  real(dp) function dB03(x)
    real(dp), intent(in) :: x
    if (eq_bool) then
      dB03 = -cte_B02**2 * sinh(x / alpha) / (alpha * cosh(x / alpha)**3 * B03(x))
    else
      dB03 = 0.0_dp
    end if
  end function dB03

  real(dp) function ddB03(x)
    real(dp), intent(in) :: x
    if (eq_bool) then
      ddB03 = cte_B02**2 * ( &
        - 1.0_dp &
        + 2.0_dp * sinh(x / alpha)**2 &
        - cte_B02**2 * tanh(x / alpha)**2 &
        / (B03(x))**2 &
      ) / (alpha**2 * cosh(x / alpha)**4 * B03(x))
    else
      ddB03 = 0.0_dp
    end if
  end function ddB03

  real(dp) function B0(x)
    real(dp), intent(in) :: x
    B0 = sqrt(B02(x)**2 + B03(x)**2)
  end function B0

end submodule smod_equil_harris_sheet
