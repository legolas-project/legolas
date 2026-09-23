! =============================================================================
!> Module containing radiative cooling-related routines.
!! This module is responsible for initialising the radiative cooling
!! variables and a correct handling of the cooling curves.
!! If an interpolated cooling curve is selected this module calls the
!! interpolation module to create one.
module mod_radiative_cooling
  use mod_global_variables, only: dp
  use mod_logging, only: logger
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_cooling_curve_names, only: ROSNER, NOTHING
  implicit none

  private

  type(settings_t), pointer :: settings => null()
  type(background_t), pointer :: background => null()

  type, public :: cooling_t
    procedure(real(dp)), pointer, nopass :: lambdaT
    procedure(real(dp)), pointer, nopass :: dlambdadT
    logical, private :: is_initialised

  contains
    procedure, public :: initialise
    procedure, public :: delete
  end type cooling_t

  public :: new_cooling

contains

  function new_cooling(settings_tgt, background_tgt) result(cooling)
    type(settings_t), target, intent(in) :: settings_tgt
    type(background_t), target, intent(in) :: background_tgt
    type(cooling_t) :: cooling
    settings => settings_tgt
    background => background_tgt
    cooling%is_initialised = .false.
    cooling%lambdaT => get_lambdaT
    cooling%dlambdadT => get_dlambdadT
  end function new_cooling


  subroutine initialise(this)
    use mod_cooling_curves, only: is_valid_cooling_curve, interpolate_cooling_curves

    class(cooling_t), intent(inout) :: this
    character(:), allocatable :: curve
    logical :: use_interpolated_curve

    if (this%is_initialised) return

    curve = settings%physics%cooling%get_cooling_curve()
    if (.not. is_valid_cooling_curve(curve)) return
    if (curve == NOTHING) return

    use_interpolated_curve = (curve /= ROSNER)
    deallocate(curve)

    if (use_interpolated_curve) then
      call interpolate_cooling_curves(settings)
      this%is_initialised = .true.
    end if
  end subroutine initialise


  real(dp) function get_lambdaT(x)
    use mod_cooling_curves, only: get_Rosner_lambdaT, get_interpolated_lambdaT
    real(dp), intent(in) :: x

    get_lambdaT = 0.0_dp
    if (.not. settings%physics%cooling%is_enabled()) return

    select case(settings%physics%cooling%get_cooling_curve())
    case(NOTHING)
      return
    case(ROSNER)
      get_lambdaT = get_Rosner_lambdaT(x, settings, background)
    case default
      get_lambdaT = get_interpolated_lambdaT(x, settings, background)
    end select
  end function get_lambdaT


  real(dp) function get_dlambdadT(x)
    use mod_cooling_curves, only: get_Rosner_dlambdadT, get_interpolated_dlambdadT
    real(dp), intent(in) :: x

    get_dlambdadT = 0.0_dp
    if (.not. settings%physics%cooling%is_enabled()) return

    select case(settings%physics%cooling%get_cooling_curve())
    case(NOTHING)
      return
    case(ROSNER)
      get_dlambdadT = get_Rosner_dlambdadT(x, settings, background)
    case default
      get_dlambdadT = get_interpolated_dlambdadT(x, settings, background)
    end select
  end function get_dlambdadT


  subroutine delete(this)
    use mod_cooling_curves, only: deallocate_cooling_curves

    class(cooling_t), intent(inout) :: this
    nullify(settings)
    nullify(background)
    nullify(this%lambdaT)
    nullify(this%dlambdadT)
    this%is_initialised = .false.
    call deallocate_cooling_curves()
  end subroutine delete

end module mod_radiative_cooling
