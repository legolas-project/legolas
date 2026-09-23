! =============================================================================
!> This submodule defines an inhomogeneous medium in Cartesian geometry with
!! a constant resistivity value. Two (constant) density profiles are defined
!! which are connected by a sine profile, representing the interface inbetween.
!! This density profile allows for resonant absorption, the geometry
!! can be overridden in the parfile.
!!
!! This equilibrium is taken from
!! _Van Doorsselaere, T., & Poedts, S. (2007).
!!  Modifications to the resistive MHD spectrum due to changes in the equilibrium.
!!  Plasma Physics and Controlled Fusion, 49(3), 261_.
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 1
!!     - <tt>k3</tt> = 0.05
!!     - <tt>p1</tt> = 0.9 : used to set the left density value.
!!     - <tt>p2</tt> = 0.1 : used to set the right density value.
!!     - <tt>r0</tt> = 0.2 : used to set the width of the interface.
!!     - <tt>cte_B02</tt> = 0 : used to set the By value.
!!     - <tt>cte_B03</tt> = 1 : used to set the Bz value.
!!     - <tt>cte_T0</tt> = 0 : used to set the temperature. Zero by default.
!!     - fixed eta value of \(10^{-3.2}\)
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_resonant_absorption
  use mod_equilibrium_params, only: p1, p2, r0, cte_T0, cte_B02, cte_B03
  implicit none

  real(dp) :: s, rho_left, rho_right, zeta
  real(dp) :: x_start, x_end

contains

  !> Sets the equilibrium
  module procedure resonant_absorption_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("Cartesian")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      call settings%physics%enable_resistivity( &
        fixed_resistivity_value=10.0_dp**(-3.2_dp) &
      )

      k2 = 1.0_dp
      k3 = 0.05_dp

      rho_left = 0.9_dp
      rho_right = 0.1_dp
      r0 = 0.2_dp
      cte_T0 = 0.0_dp
      cte_B02 = 0.0_dp
      cte_B03 = 1.0_dp
    else
      rho_left = p1
      rho_right = p2
    end if ! LCOV_EXCL_STOP
    x_start = settings%grid%get_grid_start()
    x_end = settings%grid%get_grid_end()

    s = 0.5_dp * (x_start + x_end)
    zeta = rho_left / rho_right

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_temperature_funcs(T0_func=T0)
    call background%set_magnetic_2_funcs(B02_func=B02)
    call background%set_magnetic_3_funcs(B03_func=B03)
  end procedure resonant_absorption_eq


  real(dp) function rho0(x)
    real(dp), intent(in) :: x
    if (x >= x_start .and. x < s - 0.5_dp * r0) then
      rho0 = rho_left
    else if (s - 0.5_dp * r0 <= x .and. x <= s + 0.5_dp * r0) then
      rho0 = 0.5_dp * rho_left * ( &
        1.0_dp + 1.0_dp / zeta - (1.0_dp - 1.0_dp / zeta) * sin(dpi * (x - s) / r0) &
      )
    else
      rho0 = rho_right
    end if
  end function rho0

  real(dp) function drho0(x)
    real(dp), intent(in) :: x
    if (s - 0.5_dp * r0 <= x .and. x <= s + 0.5_dp * r0) then
      drho0 = dpi * rho_left * (1.0_dp / zeta - 1.0_dp) * cos(dpi * (x - s) / r0) &
        / (2.0_dp * r0)
    else
      drho0 = 0.0_dp
    end if
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

end submodule smod_equil_resonant_absorption
