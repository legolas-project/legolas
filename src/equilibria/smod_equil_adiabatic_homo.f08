! =============================================================================
!> This submodule defines a simple, adiabatic homogeneous medium in Cartesian
!! geometry. The geometry can be overridden using the parfile.
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 0
!!     - <tt>k3</tt> = \(\pi\)
!!     - <tt>cte_rho0</tt> = 1 : used to set the density value.
!!     - <tt>cte_T0</tt> = 1 : used to set the temperature value.
!!     - <tt>cte_B02</tt> = 0 : used to set the By value.
!!     - <tt>cte_B03</tt> = 1 : used to set the Bz value.
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_adiabatic_homo
  use mod_equilibrium_params, only: cte_rho0, cte_T0, cte_B02, cte_B03
  implicit none

contains

  !> Sets the equilibrium.
  module procedure adiabatic_homo_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("Cartesian")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      k2 = 0.0_dp
      k3 = dpi

      cte_rho0 = 1.0_dp
      cte_T0 = 1.0_dp
      cte_B02 = 0.0_dp
      cte_B03 = 1.0_dp
    end if ! LCOV_EXCL_STOP

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_temperature_funcs(T0_func=T0)
    call background%set_magnetic_2_funcs(B02_func=B02)
    call background%set_magnetic_3_funcs(B03_func=B03)
  end procedure adiabatic_homo_eq

  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function T0()
    T0 = cte_T0
  end function T0

  real(dp) function B02()
    B02 = cte_B02
  end function B02

  real(dp) function B03()
    B03 = cte_B03
  end function B03

end submodule smod_equil_adiabatic_homo
