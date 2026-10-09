module mod_iv_settings
  use mod_global_variables, only: dp
  implicit none

  private

  type, public :: iv_settings_t
    logical :: enabled

    ! Solver params
    real(dp) :: t_end
    real(dp) :: dt  ! fixed time step if > 0, otherwise t_end / n_steps
    integer :: n_steps
    real(dp) :: alpha

    integer :: snapshot_stride  ! save every n-th step
  contains
    procedure, public :: get_step_size
    procedure, public :: get_end_time
    procedure, public :: get_n_snapshots
    procedure, public :: get_snapshot_interval
  end type iv_settings_t

  public :: new_iv_settings

contains

  function new_iv_settings() result(iv_settings)
    type(iv_settings_t) :: iv_settings

    ! Set defaults
    iv_settings%enabled = .false.
    iv_settings%snapshot_stride = 10

    iv_settings%alpha = 0.5_dp  ! 0.0/0.5/1.0 for FW Euler / Trapezoidal method / BW Euler

    iv_settings%t_end = 0.1_dp
    iv_settings%n_steps = 500
    iv_settings%dt = 0.0_dp
  end function new_iv_settings


  pure real(dp) function get_step_size(self)
    class(iv_settings_t), intent(in) :: self
    if (self%dt > 0.0_dp) then
      get_step_size = self%dt
    else
      get_step_size = self%t_end / real(self%n_steps, dp)
    end if
  end function get_step_size


  !> Time of the last step, n_steps * dt; at most one step past t_end.
  pure real(dp) function get_end_time(self)
    class(iv_settings_t), intent(in) :: self
    get_end_time = self%n_steps * self%get_step_size()
  end function get_end_time


  !> Snapshots are the initial condition, every snapshot_stride-th step, and the last step.
  pure integer function get_n_snapshots(self)
    class(iv_settings_t), intent(in) :: self
    get_n_snapshots = (self%n_steps - 1) / self%snapshot_stride + 2
  end function get_n_snapshots


  pure real(dp) function get_snapshot_interval(self)
    class(iv_settings_t), intent(in) :: self
    get_snapshot_interval = self%snapshot_stride * self%get_step_size()
  end function get_snapshot_interval

end module mod_iv_settings
