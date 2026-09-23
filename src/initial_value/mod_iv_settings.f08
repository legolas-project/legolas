module mod_iv_settings
  use mod_global_variables, only: dp
  implicit none

  private

  type, public :: iv_settings_t
    logical :: enabled

    integer, private :: n_snapshots
    integer :: snapshot_stride  ! save every n-th snapshot

    ! Solver params
    real(dp) :: t_end
    integer :: n_steps
    real(dp) :: alpha
  contains
    procedure, public :: get_step_size
    procedure, public :: set_n_snapshots
    procedure, public :: get_n_snapshots
  end type iv_settings_t

  public :: new_iv_settings

contains

  function new_iv_settings() result(iv_settings)
    type(iv_settings_t) :: iv_settings

    ! Set defaults
    iv_settings%enabled = .false.
    iv_settings%snapshot_stride = 10

    iv_settings%alpha = 0.52  ! 0.0/0.5/1.0 for FW Euler / Trapezoidal method / BW Euler

    iv_settings%t_end = 0.1
    iv_settings%n_steps = 500

    iv_settings%n_snapshots = floor(real(iv_settings%n_steps - 1) / real(iv_settings%snapshot_stride)) + 2

  end function new_iv_settings

  
  pure real(dp) function get_step_size(self)
    class(iv_settings_t), intent(in) :: self
    get_step_size = (self%t_end) / real(self%n_steps)
  end function get_step_size


  pure integer function get_n_snapshots(self)
    class(iv_settings_t), intent(in) :: self
    get_n_snapshots = floor(real(self%n_steps - 1) / real(self%snapshot_stride)) + 2
  end function get_n_snapshots


  subroutine set_n_snapshots(self, n_snaps)
    class(iv_settings_t), intent(inout) :: self
    integer, intent(in) :: n_snaps
    self%n_snapshots = n_snaps
  end subroutine set_n_snapshots

end module mod_iv_settings