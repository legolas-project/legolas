! =============================================================================
!> Demonstration of the initial-value solver.
!! A uniform isothermal medium in Cartesian geometry with a Gaussian density
!! perturbation centred in the domain. This is the simplest possible IVP setup
!! and is intended as a reference example.
!!
!! @note
!!     Default values are:
!!
!!     - <tt>cte_rho0</tt> = 1 : background density
!!     - <tt>cte_T0</tt>   = 1 : background temperature (= isothermal sound speed squared)
!!     - <tt>k2</tt>  = 0
!!     - <tt>k3</tt>  = 1
!!
!!     IVP mode is enabled by default. Relevant parfile settings:
!!     @code{fortran}
!!     &physicslist
!!       physics_type = "isothermal-1d"
!!     /
!!     &ivplist
!!       enabled     = .true.
!!       t_end       = 5.0
!!       n_steps     = 500
!!       n_snapshots = 50
!!     /
!!     @endcode
!! @endnote
submodule (mod_equilibrium) smod_equil_ivp_demo
  use mod_equilibrium_params, only: cte_rho0, cte_T0
  implicit none

contains

  !> Sets the equilibrium.
  module procedure ivp_demo_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("Cartesian")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      k2 = 0.0_dp
      k3 = 1.0_dp

      cte_rho0 = 1.0_dp
      cte_T0   = 1.0_dp

      ! Enable IVP mode by default
      iv_initial_conditions%density%rho  => gaussian_rho
      iv_initial_conditions%density%drho => gaussian_drho
      settings%iv%enabled     = .true.
      settings%iv%t_end       = 5.0_dp
      settings%iv%n_steps     = 500
    end if ! LCOV_EXCL_STOP

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_temperature_funcs(T0_func=T0)

    call iv_initial_conditions%set_ic_density_funcs( &
      rho_func=gaussian_rho, drho_func=gaussian_drho &
    )
  end procedure ivp_demo_eq


  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function T0()
    T0 = cte_T0
  end function T0

  !> Gaussian density perturbation centred at x = 0.5.
  pure function gaussian_rho(x) result(rho1)
    real(dp), intent(in) :: x(:)
    real(dp) :: rho1(size(x))
    real(dp), parameter :: x0 = 0.5_dp, sigma = 0.05_dp
    rho1 = exp(-((x - x0) / sigma)**2)
  end function gaussian_rho

  !> Derivative of the Gaussian density perturbation.
  pure function gaussian_drho(x) result(drho1)
    real(dp), intent(in) :: x(:)
    real(dp) :: drho1(size(x))
    real(dp), parameter :: x0 = 0.5_dp, sigma = 0.05_dp
    drho1 = -2.0_dp * (x - x0) / sigma**2 * exp(-((x - x0) / sigma)**2)
  end function gaussian_drho

end submodule smod_equil_ivp_demo
