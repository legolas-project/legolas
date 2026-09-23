module mod_heatloss
  use mod_global_variables, only: dp
  use mod_logging, only: logger
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_radiative_cooling, only: cooling_t, new_cooling
  use mod_heating, only: heating_t, new_heating
  use mod_thermal_conduction, only: conduction_t
  use mod_grid, only: grid_t
  implicit none
  private

  type(settings_t), pointer :: settings => null()
  type(background_t), pointer :: background => null()

  type(cooling_t), pointer :: cooling => null()
  type(conduction_t), pointer :: conduction => null()
  type(grid_t), pointer :: grid => null()

  character(len=*), parameter :: msg_heatloss_balance = ( &
    "unable to enforce thermal balance when heating is disabled. " // &
    "Set 'force_thermal_balance = .false.' in the parfile or enable heating." &
  )

  type, public :: heatloss_t
    type(cooling_t) :: cooling
    type(heating_t) :: heating

  contains
    procedure, public :: get_L0
    procedure, public :: get_dLdT
    procedure, public :: get_dLdrho
    procedure, public :: check_if_thermal_balance_needs_enforcing
    procedure, public :: delete
  end type heatloss_t

  public :: new_heatloss
  public :: msg_heatloss_balance

contains

  function new_heatloss(settings_tgt, background_tgt) result(heatloss)
    type(settings_t), target, intent(in) :: settings_tgt
    type(background_t), target, intent(in) :: background_tgt
    type(heatloss_t) :: heatloss

    settings => settings_tgt
    background => background_tgt
    heatloss%cooling = new_cooling(settings_tgt, background_tgt)
    heatloss%heating = new_heating(settings_tgt, background_tgt)
  end function new_heatloss


  impure elemental real(dp) function get_L0(this, x)
    class(heatloss_t), intent(in) :: this
    real(dp), intent(in) :: x
    get_L0 = background%density%rho0(x) * this%cooling%lambdaT(x) - this%heating%H(x)
  end function get_L0


  impure elemental real(dp) function get_dLdT(this, x)
    class(heatloss_t), intent(in) :: this
    real(dp), intent(in) :: x
    get_dLdT = ( &
      background%density%rho0(x) * this%cooling%dlambdadT(x) - this%heating%dHdT(x) &
    )
  end function get_dLdT


  impure elemental real(dp) function get_dLdrho(this, x)
    class(heatloss_t), intent(in) :: this
    real(dp), intent(in) :: x
    get_dLdrho = this%cooling%lambdaT(x) - this%heating%dHdrho(x)
  end function get_dLdrho


  subroutine set_module_pointers(conduction_tgt, grid_tgt, cooling_tgt)
    type(conduction_t), target, intent(in) :: conduction_tgt
    type(grid_t), target, intent(in) :: grid_tgt
    type(cooling_t), target, intent(in) :: cooling_tgt

    conduction => conduction_tgt
    grid => grid_tgt
    cooling => cooling_tgt
  end subroutine set_module_pointers


  subroutine check_if_thermal_balance_needs_enforcing(this, conduction_tgt, grid_tgt)
    use mod_function_utils, only: zero_func

    class(heatloss_t), intent(inout) :: this
    type(conduction_t), intent(in) :: conduction_tgt
    type(grid_t), intent(in) :: grid_tgt

    if (is_adiabatic()) return
    if (.not. settings%physics%heating%force_thermal_balance) return
    if (.not. settings%physics%heating%is_enabled()) then
      call logger%error(msg_heatloss_balance)
      return
    end if
    if (.not. associated(this%heating%H, zero_func)) call log_usr_H_func_warning()

    call set_module_pointers(conduction_tgt, grid_tgt, this%cooling)
    call logger%info("enforcing thermal balance by setting a steady heating term")
    this%heating%H => H_for_thermal_balance
  end subroutine check_if_thermal_balance_needs_enforcing


  real(dp) function H_for_thermal_balance(x)
    real(dp), intent(in) :: x
    real(dp) :: rho0, T0, dT0, ddT0, v01, dv01, B01
    real(dp) :: eps, deps
    real(dp) :: Kp, dKp, tcperp, dtcperpdr

    rho0 = background%density%rho0(x)
    T0 = background%temperature%T0(x)
    dT0 = background%temperature%dT0(x)
    ddT0 = background%temperature%ddT0(x)
    v01 = background%velocity%v01(x)
    dv01 = background%velocity%dv01(x)
    B01 = background%magnetic%B01(x)
    eps = grid%get_eps(x)
    deps = grid%get_deps()
    Kp = conduction%get_tcprefactor(x)
    dKp = conduction%get_dtcprefactordr(x)
    tcperp = conduction%tcperp(x)
    dtcperpdr = conduction%get_dtcperpdr(x)

    H_for_thermal_balance = rho0 * cooling%lambdaT(x) + (1.0_dp / rho0) * ( &
      T0 * rho0 * (deps * v01 + eps * dv01) / eps &
      - B01**2 * (dKp * dT0 + Kp * ddT0) &
      - (1.0_dp / eps) * ( &
          deps * tcperp * dT0 &
          + eps * dtcperpdr * dT0 &
          + eps * tcperp * ddT0 &
        ) &
      + (1.0_dp / settings%physics%get_gamma_1()) * dT0 * rho0 * v01 &
    )
  end function H_for_thermal_balance


  subroutine log_usr_H_func_warning() ! LCOV_EXCL_START
    call logger%warning( &
      "found user-defined heating function but forced thermal balance enabled!" &
    )
    call logger%disable_prefix()
    call logger%warning( &
      "The provided function will be ignored and thermal balance will be enforced." &
    )
    call logger%warning( &
      "To disable this behaviour, set force_thermal_balance = .false. in the parfile" &
    )
    call logger%enable_prefix()
  end subroutine log_usr_H_func_warning ! LCOV_EXCL_STOP


  logical function is_adiabatic()
    logical :: no_cooling, no_heating, no_conduction
    no_cooling = .not. settings%physics%cooling%is_enabled()
    no_heating = .not. settings%physics%heating%is_enabled()
    no_conduction = .not. settings%physics%conduction%is_enabled()
    is_adiabatic = no_cooling .and. no_heating .and. no_conduction
  end function is_adiabatic


  subroutine delete(this)
    class(heatloss_t), intent(inout) :: this
    nullify(settings)
    nullify(background)
    nullify(conduction)
    nullify(grid)
    nullify(cooling)
    call this%cooling%delete()
    call this%heating%delete()
  end subroutine delete

end module mod_heatloss
