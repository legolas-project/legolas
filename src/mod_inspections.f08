! =============================================================================
!> Module to inspect if certain conditions are fulfilled by doing
!! additional sanity checks on the equilibrium configuration.
!! For cylindrical geometries we check if \(k_2\) is an integer and if the
!! on-axis values obey regularity conditions. Equilibrium balance
!! for both the Cartesian and cylindrical cases is checked.
module mod_inspections
  use mod_global_variables, only: dp
  use mod_logging, only: logger, str, exp_fmt
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_physics, only: physics_t
  use mod_grid, only: grid_t
  use mod_function_utils, only: from_function
  use mod_check_values, only: is_NaN, is_negative, is_zero
  implicit none

  private

  ! needs explicit interface to pass procedure pointers
  interface
    real(dp) function dp_func(x)
      import dp
      real(dp), intent(in) :: x
    end function dp_func
  end interface

  public :: do_equilibrium_inspections

contains

  subroutine do_equilibrium_inspections(settings, grid, background, physics)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(inout) :: physics

    call physics%heatloss%check_if_thermal_balance_needs_enforcing( &
      physics%conduction, grid &
    )
    if (nan_values_present(background, physics, grid)) return
    if (negative_values_present(background, grid)) return
    if (.not. wave_numbers_are_valid(geometry=settings%grid%get_geometry())) return
    if (B01_and_cylindrical(settings, grid, background)) return
    call validate_on_axis_values(settings, background)
    call validate_equilibrium_conditions(settings, grid, background, physics)
  end subroutine do_equilibrium_inspections


  logical function contains_negative(func, grid)
    procedure(dp_func), pointer :: func
    type(grid_t), intent(in) :: grid
    contains_negative = any(is_negative(from_function(func, grid%gaussian_grid)))
  end function contains_negative


  logical function contains_NaN(func, grid)
    procedure(dp_func), pointer :: func
    type(grid_t), intent(in) :: grid
    contains_NaN = any(is_NaN(from_function(func, grid%gaussian_grid)))
  end function contains_NaN


  logical function nan_values_present(background, physics, grid)
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics
    type(grid_t), intent(in) :: grid
    character(50) :: name

    nan_values_present = .false.
    name = ""
    if (contains_NaN(background%density%rho0, grid)) then
      name = "rho0"
    else if (contains_NaN(background%temperature%T0, grid)) then
      name = "T0"
    else if (contains_NaN(background%magnetic%B01, grid)) then
      name = "B01"
    else if (contains_NaN(background%magnetic%B02, grid)) then
      name = "B02"
    else if (contains_NaN(background%magnetic%B03, grid)) then
      name = "B03"
    else if (contains_NaN(background%velocity%v01, grid)) then
      name = "v01"
    else if (contains_NaN(background%velocity%v02, grid)) then
      name = "v02"
    else if (contains_NaN(background%velocity%v03, grid)) then
      name = "v03"
    else if (contains_NaN(physics%gravity%g0, grid)) then
      name = "gravity"
    end if
    if (name /= "") then
      call logger%error("NaN encountered in " // adjustl(trim(name)))
      nan_values_present = .true.
    end if
  end function nan_values_present


  logical function negative_values_present(background, grid)
    type(background_t), intent(in) :: background
    type(grid_t), intent(in) :: grid
    character(50) :: name

    negative_values_present = .false.
    name = ""
    if (contains_negative(background%density%rho0, grid)) then
      name = "rho0"
    else if (contains_negative(background%temperature%T0, grid)) then
      name = "T0"
    end if
    if (name /= "") then
      call logger%error("negative values encountered in " // adjustl(trim(name)))
      negative_values_present = .true.
    end if
  end function negative_values_present


  logical function wave_numbers_are_valid(geometry)
    use mod_equilibrium_params, only: k2
    character(len=*), intent(in) :: geometry

    wave_numbers_are_valid = .true.
    ! in cylindrical geometry k2 should be an integer
    if (geometry == "cylindrical" .and. .not. is_zero(abs(int(k2) - k2))) then
      call logger%error( &
        "cylindrical geometry but k2 is not an integer! Value: " // str(k2) &
      )
      wave_numbers_are_valid = .false.
    end if
  end function wave_numbers_are_valid


  logical function B01_and_cylindrical(settings, grid, background)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    logical :: B01_is_zero

    B01_and_cylindrical = .false.
    if (.not. settings%grid%get_geometry() == "cylindrical") return

    B01_is_zero = all( &
      is_zero(from_function(background%magnetic%B01, grid%gaussian_grid)) &
    )
    if (.not. B01_is_zero) then
      call logger%error( &
        "B01 component currently not supported for cylindrical geometries!" &
      )
      B01_and_cylindrical = .true.
    end if
  end function B01_and_cylindrical


  subroutine validate_on_axis_values(settings, background)
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background

    if (settings%grid%get_geometry() == "Cartesian") return
    if (.not. is_zero(settings%grid%get_grid_start())) return

    ! LCOV_EXCL_START
    if (.not. is_zero(background%magnetic%B02(0.0_dp))) then
      call logger%warning( &
        "B_theta non-zero on axis! Value: " // str(background%magnetic%B02(0.0_dp)) &
      )
    end if
    if (.not. is_zero(background%magnetic%dB03(0.0_dp))) then
      call logger%warning( &
        "dBz/dr non-zero on axis! Value: " // str(background%magnetic%dB03(0.0_dp)) &
      )
    end if
    if (.not. is_zero(background%velocity%v02(0.0_dp))) then
      call logger%warning( &
        "v_theta non-zero on axis! Value: " // str(background%velocity%v02(0.0_dp)) &
      )
    end if
    if (.not. is_zero(background%velocity%dv03(0.0_dp))) then
      call logger%warning( &
        "dvz_dr non-zero on axis! Value: " // str(background%velocity%dv03(0.0_dp)) &
      )
    end if
    ! LCOV_EXCL_STOP
  end subroutine validate_on_axis_values


  subroutine validate_equilibrium_conditions(settings, grid, background, physics)
    use mod_check_values, only: is_zero
    use mod_global_variables, only: dp_limit
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics
    real(dp) :: values(size(grid%gaussian_grid))
    real(dp) :: tol

    tol = 10.0_dp * dp_limit
    values = 0.0_dp
    ! continuity equation
    values = continuity_condition(grid%gaussian_grid, grid, background)
    if (.not. all(is_zero(values, tol))) then
      call logger%warning("continuity equation not satisfied!")
      call logger%warning("location of largest discrepancy: x = " // str(getx()))
      call logger%warning("value: " // str(getval(), fmt=exp_fmt))
    end if
    ! force balance
    values = force_balance_1_condition(grid%gaussian_grid, grid, background, physics)
    if (.not. all(is_zero(values, tol))) then
      call logger%warning("force balance 1 equation not satisfied!")
      call logger%warning("location of largest discrepancy: x = " // str(getx()))
      call logger%warning("value: " // str(getval(), fmt=exp_fmt))
    end if
    values = force_balance_2_condition(grid%gaussian_grid, grid, background)
    if (.not. all(is_zero(values, tol))) then
      call logger%warning("force balance 2 equation not satisfied!")
      call logger%warning("location of largest discrepancy: x = " // str(getx()))
      call logger%warning("value: " // str(getval(), fmt=exp_fmt))
    end if
    values = force_balance_3_condition(grid%gaussian_grid, background)
    if (.not. all(is_zero(values, tol))) then
      call logger%warning("force balance 3 equation not satisfied!")
      call logger%warning("location of largest discrepancy: x = " // str(getx()))
      call logger%warning("value: " // str(getval(), fmt=exp_fmt))
    end if
    ! thermal balance
    values = energy_balance_condition( &
      grid%gaussian_grid, settings, grid, background, physics &
    )
    if (.not. all(is_zero(values, tol))) then
      call logger%warning("energy balance equation not satisfied!")
      call logger%warning("location of largest discrepancy: x = " // str(getx()))
      call logger%warning("value: " // str(getval(), fmt=exp_fmt))
    end if
    ! induction equation
    values = induction_1_condition(grid%gaussian_grid, background)
    if (.not. all(is_zero(values, tol))) then
      call logger%warning("induction 1 equation not satisfied!")
      call logger%warning("location of largest discrepancy: x = " // str(getx()))
      call logger%warning("value: " // str(getval(), fmt=exp_fmt))
    end if
    values = induction_2_condition(grid%gaussian_grid, grid, background)
    if (.not. all(is_zero(values, tol))) then
      call logger%warning("induction 2 equation not satisfied!")
      call logger%warning("location of largest discrepancy: x = " // str(getx()))
      call logger%warning("value: " // str(getval(), fmt=exp_fmt))
    end if


  contains
    real(dp) function getx()
      getx = grid%gaussian_grid(maxloc(abs(values), dim=1))
    end function getx
    real(dp) function getval()
      getval = values(maxloc(abs(values), dim=1))
    end function getval
  end subroutine validate_equilibrium_conditions


  impure elemental real(dp) function continuity_condition(x, grid, background)
    real(dp) , intent(in) :: x
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    real(dp) :: rho0, drho0, v01, dv01, eps, deps

    rho0 = background%density%rho0(x)
    drho0 = background%density%drho0(x)
    v01 = background%velocity%v01(x)
    dv01 = background%velocity%dv01(x)
    eps = grid%get_eps(x)
    deps = grid%get_deps()

    continuity_condition = drho0 * v01 + rho0 * dv01 + rho0 * v01 * deps / eps
  end function continuity_condition


  impure elemental real(dp) function force_balance_1_condition( &
    x, grid, background, physics &
  )
    real(dp) , intent(in) :: x
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics
    real(dp) :: rho0, drho0, B02, dB02, B03, dB03, T0, dT0, grav
    real(dp) :: v01, v02, dv01
    real(dp) :: eps, deps

    rho0 = background%density%rho0(x)
    drho0 = background%density%drho0(x)
    B02 = background%magnetic%B02(x)
    B03 = background%magnetic%B03(x)
    dB02 = background%magnetic%dB02(x)
    dB03 = background%magnetic%dB03(x)
    T0 = background%temperature%T0(x)
    dT0 = background%temperature%dT0(x)
    v01 = background%velocity%v01(x)
    v02 = background%velocity%v02(x)
    dv01 = background%velocity%dv01(x)
    grav = physics%gravity%g0(x)
    eps = grid%get_eps(x)
    deps = grid%get_deps()

    force_balance_1_condition = ( &
      drho0 * T0 &
      + rho0 * dT0 &
      + B02 * dB02 &
      + B03 * dB03 &
      + rho0 * grav &
      - (deps/eps) * (rho0 * v02**2 - B02**2) &
      + rho0 * v01 * dv01 &
    )
  end function force_balance_1_condition


  impure elemental real(dp) function force_balance_2_condition(x, grid, background)
    real(dp), intent(in) :: x
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background

    real(dp) :: rho0, B01, B02, dB02, v01, v02, dv02, eps, deps

    rho0 = background%density%rho0(x)
    B01 = background%magnetic%B01(x)
    B02 = background%magnetic%B02(x)
    dB02 = background%magnetic%dB02(x)
    v01 = background%velocity%v01(x)
    v02 = background%velocity%v02(x)
    dv02 = background%velocity%dv02(x)
    eps = grid%get_eps(x)
    deps = grid%get_deps()

    force_balance_2_condition = ( &
      rho0 * v01 * (dv02 + v02 * deps / eps) - B01 * (dB02 + B02 * deps / eps) &
    )
  end function force_balance_2_condition


  impure elemental real(dp) function force_balance_3_condition(x, background)
    real(dp), intent(in) :: x
    type(background_t), intent(in) :: background
    real(dp) :: rho0, B01, dB03, v01, dv03

    rho0 = background%density%rho0(x)
    B01 = background%magnetic%B01(x)
    dB03 = background%magnetic%dB03(x)
    v01 = background%velocity%v01(x)
    dv03 = background%velocity%dv03(x)

    force_balance_3_condition = rho0 * v01 * dv03 - B01 * dB03
  end function force_balance_3_condition


  impure elemental real(dp) function energy_balance_condition( &
    x, settings, grid, background, physics &
  )
    real(dp), intent(in) :: x
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics

    real(dp) :: rho0, T0, dT0, ddT0, B01, v01, dv01
    real(dp) :: kappa_perp, dkappa_perp_dr, Kp, dKp, L0
    real(dp) :: eps, deps

    rho0 = background%density%rho0(x)
    B01 = background%magnetic%B01(x)
    T0 = background%temperature%T0(x)
    dT0 = background%temperature%dT0(x)
    ddT0 = background%temperature%ddT0(x)
    v01 = background%velocity%v01(x)
    dv01 = background%velocity%dv01(x)
    eps = grid%get_eps(x)
    deps = grid%get_deps()
    kappa_perp = physics%conduction%tcperp(x)
    dkappa_perp_dr = physics%conduction%get_dtcperpdr(x)
    Kp = physics%conduction%get_tcprefactor(x)
    dKp = physics%conduction%get_dtcprefactordr(x)
    L0 = physics%heatloss%get_L0(x)

    energy_balance_condition = ( &
      T0 * rho0 * (deps * v01 + eps * dv01) / eps &
      + rho0 * L0 &
      - B01**2 * (Kp * dT0 + dKp * T0) &
      - (1.0_dp / eps) * ( &
        deps * kappa_perp * dT0 &
        + eps * dkappa_perp_dr * dT0 &
        + eps * kappa_perp * ddT0 &
      ) &
      + (1.0_dp / settings%physics%get_gamma_1()) * dT0 * rho0 * v01 &
    )
  end function energy_balance_condition


  impure elemental real(dp) function induction_1_condition(x, background)
    real(dp), intent(in) :: x
    type(background_t), intent(in) :: background
    real(dp)  :: B01, B02, dB02, v01, dv01, dv02

    B01 = background%magnetic%B01(x)
    B02 = background%magnetic%B02(x)
    dB02 = background%magnetic%dB02(x)
    v01 = background%velocity%v01(x)
    dv01 = background%velocity%dv01(x)
    dv02 = background%velocity%dv02(x)

    induction_1_condition = B01 * dv02 - B02 * dv01 - dB02 * v01
  end function induction_1_condition


  impure elemental real(dp) function induction_2_condition(x, grid, background)
    real(dp), intent(in) :: x
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    real(dp)  :: B01, B03, dB03, v01, v03, dv01, dv03, eps, deps

    B01 = background%magnetic%B01(x)
    B03 = background%magnetic%B03(x)
    dB03 = background%magnetic%dB03(x)
    v01 = background%velocity%v01(x)
    v03 = background%velocity%v03(x)
    dv01 = background%velocity%dv01(x)
    dv03 = background%velocity%dv03(x)
    eps = grid%get_eps(x)
    deps = grid%get_deps()

    induction_2_condition = ( &
      B01 * dv03 - dB03 * v01 - B03 * dv01 + deps * (B01 * v03 - B03 * v01) / eps &
    )
  end function induction_2_condition

end module mod_inspections
