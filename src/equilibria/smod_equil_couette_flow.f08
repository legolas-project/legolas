! =============================================================================
!> This submodule defines a steady plane Couette flow in a Cartesian geometry
!! with flow and viscosity.
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 0
!!     - <tt>k3</tt> = 1
!!     - <tt>cte_rho0</tt> = 1
!!     - <tt>cte_T0</tt> = 1
!!     - <tt>cte_v02</tt> = 0
!!     - <tt>cte_v03</tt> = 1
!!     - <tt>viscosity</tt> = True
!!     - <tt>viscosity_value</tt> = 1e-3
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_couette_flow
  use mod_equilibrium_params, only: cte_rho0, cte_v02, cte_v03, cte_T0
  implicit none

  real(dp) :: width

contains

  module procedure couette_flow_eq
    call settings%physics%enable_flow()
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("Cartesian")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      cte_v02  = 0.0_dp
      cte_v03  = 1.0_dp
      cte_T0   = 1.0_dp
      cte_rho0 = 1.0_dp

      k2 = 0.0_dp
      k3 = 1.0_dp

      call settings%physics%enable_viscosity(viscosity_value=0.001_dp)
    end if ! LCOV_EXCL_STOP

    width = settings%grid%get_grid_end() - settings%grid%get_grid_start()

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_velocity_2_funcs(v02_func=v02, dv02_func=dv02)
    call background%set_velocity_3_funcs(v03_func=v03, dv03_func=dv03)
    call background%set_temperature_funcs(T0_func=T0)
  end procedure couette_flow_eq


  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function T0()
    T0 = cte_T0
  end function T0

  real(dp) function v02(x)
    real(dp), intent(in) :: x
    v02 = cte_v02 * x / width
  end function v02

  real(dp) function dv02()
    dv02 = cte_v02 / width
  end function dv02

  real(dp) function v03(x)
    real(dp), intent(in) :: x
    v03 = cte_v03 * x / width
  end function v03

  real(dp) function dv03()
    dv03 = cte_v03 / width
  end function dv03

end submodule smod_equil_couette_flow
