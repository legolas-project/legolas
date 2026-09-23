module mod_physics
  use mod_global_variables, only: dp
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_resistivity, only: resistivity_t, new_resistivity
  use mod_gravity, only: gravity_t, new_gravity
  use mod_hall, only: hall_t, new_hall
  use mod_thermal_conduction, only: conduction_t, new_conduction
  use mod_heatloss, only: heatloss_t, new_heatloss
  implicit none

  private

  type, public :: physics_t
    type(settings_t), pointer, private :: settings
    type(resistivity_t) :: resistivity
    type(gravity_t) :: gravity
    type(hall_t) :: hall
    type(conduction_t) :: conduction
    type(heatloss_t) :: heatloss
  contains
    procedure, public :: set_resistivity_funcs
    procedure, public :: set_gravity_funcs
    procedure, public :: set_parallel_conduction_funcs
    procedure, public :: set_perpendicular_conduction_funcs
    procedure, public :: set_cooling_funcs
    procedure, public :: set_heating_funcs
    procedure, public :: delete
  end type physics_t

  public :: new_physics

contains

  function new_physics(settings, background) result(physics)
    type(settings_t), target, intent(in) :: settings
    type(background_t), target, intent(in) :: background
    type(physics_t) :: physics

    physics%settings => settings
    physics%resistivity = new_resistivity(settings, background)
    physics%gravity = new_gravity()
    physics%hall = new_hall(settings, background)
    physics%conduction = new_conduction(settings, background)
    physics%heatloss = new_heatloss(settings, background)
  end function new_physics


  subroutine set_resistivity_funcs(this, eta_func, detadT_func, detadr_func)
    use mod_logging, only: logger
    class(physics_t), intent(inout) :: this
    procedure(real(dp)) :: eta_func
    procedure(real(dp)), optional :: detadT_func
    procedure(real(dp)), optional :: detadr_func

    if (.not. this%settings%physics%resistivity%is_enabled()) then
      call log_function_warning(name="resistivity")
      return
    end if

    if (this%settings%physics%resistivity%has_fixed_resistivity()) then
      call logger%warning( &
        "User-defined resistivity functions override the fixed_resistivity_value." &
      )
    end if

    this%resistivity%eta => eta_func
    if (present(detadT_func)) this%resistivity%detadT => detadT_func
    if (present(detadr_func)) this%resistivity%detadr => detadr_func
  end subroutine set_resistivity_funcs


  subroutine set_gravity_funcs(this, g0_func)
    class(physics_t), intent(inout) :: this
    procedure(real(dp)) :: g0_func

    if (.not. this%settings%physics%gravity%is_enabled()) then
      call log_function_warning(name="gravity")
      return
    end if
    this%gravity%g0 => g0_func
  end subroutine set_gravity_funcs


  subroutine set_parallel_conduction_funcs(this, tcpara_func, dtcparadT_func)
    class(physics_t), intent(inout) :: this
    procedure(real(dp)) :: tcpara_func
    procedure(real(dp)), optional :: dtcparadT_func

    if (.not. this%settings%physics%conduction%has_parallel_conduction()) then
      call log_function_warning(name="parallel conduction")
      return
    end if
    this%conduction%tcpara => tcpara_func
    if (present(dtcparadT_func)) this%conduction%dtcparadT => dtcparadT_func
  end subroutine set_parallel_conduction_funcs


  subroutine set_perpendicular_conduction_funcs( &
    this, tcperp_func, dtcperpdT_func, dtcperpdrho_func, dtcperpdB2_func &
  )
    class(physics_t), intent(inout) :: this
    procedure(real(dp)) :: tcperp_func
    procedure(real(dp)), optional :: dtcperpdT_func
    procedure(real(dp)), optional :: dtcperpdrho_func
    procedure(real(dp)), optional :: dtcperpdB2_func

    if (.not. this%settings%physics%conduction%has_perpendicular_conduction()) then
      call log_function_warning(name="perpendicular conduction")
      return
    end if
    this%conduction%tcperp => tcperp_func
    if (present(dtcperpdT_func)) this%conduction%dtcperpdT => dtcperpdT_func
    if (present(dtcperpdrho_func)) this%conduction%dtcperpdrho => dtcperpdrho_func
    if (present(dtcperpdB2_func)) this%conduction%dtcperpdB2 => dtcperpdB2_func
  end subroutine set_perpendicular_conduction_funcs


  subroutine set_cooling_funcs(this, lambdaT_func, dlambdadT_func)
    class(physics_t), intent(inout) :: this
    procedure(real(dp)) :: lambdaT_func
    procedure(real(dp)), optional :: dlambdadT_func

    if (.not. this%settings%physics%cooling%is_enabled()) then
      call log_function_warning(name="cooling")
      return
    end if
    this%heatloss%cooling%lambdaT => lambdaT_func
    if (present(dlambdadT_func)) this%heatloss%cooling%dlambdadT => dlambdadT_func
  end subroutine set_cooling_funcs


  subroutine set_heating_funcs(this, H_func, dHdT_func, dHdrho_func)
    class(physics_t), intent(inout) :: this
    procedure(real(dp)) :: H_func
    procedure(real(dp)), optional :: dHdT_func
    procedure(real(dp)), optional :: dHdrho_func

    if (.not. this%settings%physics%heating%is_enabled()) then
      call log_function_warning(name="heating")
      return
    end if
    this%heatloss%heating%H => H_func
    if (present(dHdT_func)) this%heatloss%heating%dHdT => dHdT_func
    if (present(dHdrho_func)) this%heatloss%heating%dHdrho => dHdrho_func
  end subroutine set_heating_funcs


  subroutine delete(this)
    class(physics_t), intent(inout) :: this
    call this%resistivity%delete()
    call this%gravity%delete()
    call this%hall%delete()
    call this%conduction%delete()
    call this%heatloss%delete
  end subroutine delete


  subroutine log_function_warning(name)
    use mod_logging, only: logger
    character(len=*), intent(in) :: name
    call logger%warning( &
      trim(name) &
      // " is disabled. User-defined " &
      // trim(name) &
      // " function(s) not set." &
    )
  end subroutine log_function_warning

end module mod_physics
