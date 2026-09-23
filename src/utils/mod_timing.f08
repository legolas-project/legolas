!======================================
!> Module to provide timing facilities.
!!
!!### Usage
!! The simplest way is using the module variable.
!!
!! Using wall time
!!```fortran
!!call tic()
!!call slow_operation()
!!call toc("slow operation is done")
!!```
!! or CPU time
!!```fortran
!!call cputic()
!!call slow_operation()
!!call cputoc("slow operation is done")
!!```
!!
!! Alternatively, if thread safety is required,
!! you can keep track of the start time yourself.
!!```fortran
!!integer :: start_time
!!call tic(start_time)
!!call slow_operation()
!!call toc("slow operation is done", start_time)
!!```
!! or
!!```fortran
!!real :: start_time
!!call cputic(start_time)
!!call slow_operation()
!!call cputoc("slow operation is done", start_time)
!!```
module mod_timing
  use mod_global_variables, only: dp
  use mod_logging, only: logger, str
  implicit none

  private
  public tic, toc, cputic, cputoc

  !> Wall clock time at last call of `tic()`, init -1000 to easily detect incorrect use.
  integer, save :: wall_clock_tic = -1000
  !> CPU time at last call of `cputic()`, init -1000 to easily detect incorrect use.
  real(dp), save :: cpu_clock_tic = -1000.0_dp

  type, public :: timer_t
    integer, private :: start_time
    integer, private :: program_start_time
    real(dp) :: init_time
    real(dp) :: matrix_time
    real(dp) :: ivp_time
    real(dp) :: evp_time
    real(dp) :: eigenfunction_time
    real(dp) :: datfile_time

    contains

    procedure :: start_timer
    procedure :: end_timer
    procedure :: get_total_time
  end type timer_t

  public :: new_timer

contains

  function new_timer() result(timer)
    type(timer_t) :: timer
    timer%start_time = -1000
    call system_clock(timer%program_start_time)
  end function new_timer


  subroutine start_timer(this)
    class(timer_t), intent(inout) :: this
    call system_clock(this%start_time)
  end subroutine start_timer


  function end_timer(this) result(elapsed_time)
    class(timer_t), intent(inout) :: this
    integer :: end_time, rate
    real(dp) :: elapsed_time

    call system_clock(end_time, rate)
    elapsed_time = real(end_time - this%start_time, kind=dp) / rate
    this%start_time = -1000
  end function end_timer


  function get_total_time(this) result(total_time)
    class(timer_t), intent(in) :: this
    integer :: end_time, rate
    real(dp) :: total_time

    call system_clock(end_time, rate)
    total_time = real(end_time - this%program_start_time, kind=dp) / rate
  end function get_total_time

  ! LCOV_EXCL_START

  !> Subroutine to start a wall clock timer.
  !!
  !! `tic()` starts the wall clock timer. The subroutines records the wall clock
  !! time at execution of this command. Use `toc(message)` to log the elapsed time
  !! since the last call of `tic()`.
  !!
  !! @warning
  !!     As `tic()` and `toc(message)` rely on a module variable, they are
  !!     not thread safe.
  !! @endwarning
  !!
  !! `tic(start_time)` writes the value of the wall clock time at execution
  !! of this command to **start_time**. Calling this command, shall not
  !! change the module variable associated with calling `tic()`.
  subroutine tic(start_time)
    !> Optional output for the start time. If not present the time is
    !! written to a module variable.
    integer, optional, intent(out) :: start_time

    if (present(start_time)) then
        call system_clock(start_time)
    else
        call system_clock(wall_clock_tic)
    end if
  end subroutine tic

  !> Subroutine to end a wall clock timer.
  !!
  !! `toc(message)` logs the elapsed wall clock time in seconds since the
  !! most recent call to `tic()` along with **message** as a debug message.
  !!
  !! @warning
  !!     As `tic()` and `toc(message)` rely on a module variable, they are
  !!     not thread safe.
  !! @endwarning
  !!
  !! @warning
  !!     If `toc(message)` is called without first calling `tic()` the result
  !!     will be meaningless.
  !! @endwarning
  !!
  !! `toc(message, start_time)` logs the elapsed wall clock time in seconds since
  !! the call of `tic(start_time)`, along with **message**.
  subroutine toc(message, start_time, level)
    !> Message to log along elapsed time.
    character(len=*), intent(in)            :: message
    !> Optional starting time. If not present the time recorded in
    !! the module variable is used.
    integer, intent(in), optional           :: start_time
    !> The level (severity) of the message, default is <tt>"debug"</tt>.
    character(len=*), intent(in), optional  :: level

    integer                   :: selected_start_time, rate, end_time
    real(dp)                  :: elapsed_time
    character(:), allocatable :: loglevel

    if (present(start_time)) then
        selected_start_time = start_time
    else
        selected_start_time = wall_clock_tic
    end if

    if (present(level)) then
      loglevel = level
    else
      loglevel = "debug"
    end if

    call system_clock(end_time, rate)

    elapsed_time = real(end_time - selected_start_time, kind=dp) / rate
    call logger%debug( &
      message // " (" // str(elapsed_time, fmt="f20.3") // " s)" &
    )
  end subroutine toc

  !> Subroutine to start a CPU timer.
  !!
  !! `cputic()` starts the CPU timer. The subroutines records the processor time
  !! at execution of this command. Use `cputoc(message)` to log the elapsed time
  !! since the last call of `cputic()`.
  !!
  !! @warning
  !!     As `cputic()` and `cputoc(message)` rely on a module variable, they
  !!     are not thread safe.
  !! @endwarning
  !!
  !! `cputic(start_time)` writes the value of the processor time at execution
  !! of this command to **start_time**. Calling this command, shall not change
  !! the module variable associated with calling `cputic()`.
  subroutine cputic(start_time)
    !> Optional output for the start time. If not present the time is
    !! written to a module variable.
    real, optional, intent(out) :: start_time

    if (present(start_time)) then
        call cpu_time(start_time)
    else
        call cpu_time(cpu_clock_tic)
    end if
  end subroutine cputic

  !> Subroutine to end a CPU timer.
  !!
  !! `cputoc(message)` logs the elapsed CPU time in seconds since the most recent
  !! call to `cputic()` along with **message** as a debug message.
  !!
  !! @warning
  !!     As `cputic()` and `cputoc(message)` rely on a module variable, they
  !!     are not thread safe.
  !! @endwarning
  !!
  !! @warning
  !!     If `cputoc(message)` is called without first calling `cputic()` the
  !!     result will be meaningless.
  !! @endwarning
  !!
  !! `cputoc(message, start_time)` logs the elapsed CPU time in seconds since the
  !! call of `cputic(start_time)`, along with **message**.
  subroutine cputoc(message, start_time, level)
    !> Message to log along elapsed time.
    character(len=*), intent(in)           :: message
    !> Optional starting time. If not present the time recorded in
    !! the module variable is used.
    real(dp), intent(in), optional :: start_time
    !> The level (severity) of the message, default is <tt>"debug"</tt>.
    character(len=*), intent(in), optional :: level

    real(dp) :: selected_start_time, end_time
    real(dp) :: elapsed_time
    character(:), allocatable :: loglevel

    if (present(start_time)) then
        selected_start_time = start_time
    else
        selected_start_time = cpu_clock_tic
    end if

    if (present(level)) then
      loglevel = level
    else
      loglevel = "debug"
    end if

    call cpu_time(end_time)

    elapsed_time = end_time - selected_start_time
    call logger%debug( &
      message // " (" // str(elapsed_time, fmt="f20.3") // " s, CPU time)" &
    )
  end subroutine cputoc

  ! LCOV_EXCL_END

end module mod_timing
