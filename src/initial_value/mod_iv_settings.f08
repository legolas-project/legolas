module mod_iv_settings
  use mod_global_variables, only: dp
  use mod_logging, only: logger, str
  implicit none

  private

  type, public :: iv_settings_t
    logical :: enabled

    integer :: snapshot_stride  ! save every n-th step

    ! Solver params
    real(dp) :: t_end
    integer :: n_steps
    real(dp) :: alpha
  contains
    procedure, public :: get_step_size
    procedure, public :: get_n_snapshots
    procedure, public :: set_stride_from_n_snapshots
    procedure, public :: validate
  end type iv_settings_t

  public :: new_iv_settings

contains

  function new_iv_settings() result(iv_settings)
    type(iv_settings_t) :: iv_settings

    ! Set defaults
    iv_settings%enabled = .false.
    iv_settings%snapshot_stride = 10

    ! 0.0/0.5/1.0 for FW Euler / Trapezoidal method / BW Euler. The default is
    ! slightly more implicit than 0.5, which adds a little numerical damping.
    iv_settings%alpha = 0.52_dp

    iv_settings%t_end = 0.1_dp
    iv_settings%n_steps = 500
  end function new_iv_settings


  pure real(dp) function get_step_size(self)
    class(iv_settings_t), intent(in) :: self
    get_step_size = self%t_end / real(self%n_steps, kind=dp)
  end function get_step_size


  !> Number of stored snapshots: the initial condition, every
  !! `snapshot_stride`-th step before the last one, and the final step.
  pure integer function get_n_snapshots(self)
    class(iv_settings_t), intent(in) :: self
    get_n_snapshots = (self%n_steps - 1) / self%snapshot_stride + 2
  end function get_n_snapshots


  !> Sets the snapshot stride such that at most `n_snaps` snapshots are stored
  !! (including the initial condition and the final step).
  subroutine set_stride_from_n_snapshots(self, n_snaps)
    class(iv_settings_t), intent(inout) :: self
    integer, intent(in) :: n_snaps

    if (n_snaps < 2) then
      call logger%error( &
        "ivplist: n_snapshots must be at least 2 (initial and final state), got " &
        // str(n_snaps) &
      )
      return
    end if
    if (n_snaps == 2) then
      self%snapshot_stride = max(1, self%n_steps)
    else
      ! ceiling((n_steps - 1) / (n_snaps - 2)) so we never exceed n_snaps
      self%snapshot_stride = max( &
        1, (self%n_steps - 1 + (n_snaps - 2) - 1) / (n_snaps - 2) &
      )
    end if
  end subroutine set_stride_from_n_snapshots


  !> Checks the IV settings for consistency, raises an error if invalid.
  subroutine validate(self)
    class(iv_settings_t), intent(in) :: self

    if (self%n_steps < 1) then
      call logger%error("ivplist: n_steps must be positive, got " // str(self%n_steps))
      return
    end if
    if (self%snapshot_stride < 1) then
      call logger%error( &
        "ivplist: snapshot_stride must be positive, got " &
        // str(self%snapshot_stride) &
      )
      return
    end if
    if (.not. self%t_end > 0.0_dp) then
      call logger%error("ivplist: t_end must be positive, got " // str(self%t_end))
      return
    end if
    if (self%alpha < 0.0_dp .or. self%alpha > 1.0_dp) then
      call logger%error("ivplist: alpha must be in [0, 1], got " // str(self%alpha))
      return
    end if
    if (self%alpha < 0.5_dp) then
      call logger%warning( &
        "ivplist: alpha < 0.5 is only conditionally stable, results may diverge" &
      )
    end if
  end subroutine validate

end module mod_iv_settings
