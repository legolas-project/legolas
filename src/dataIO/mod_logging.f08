module mod_logging
  use mod_global_variables, only: dp
  use mod_painting, only: paint_string
  implicit none

  private

  !> exponential format
  character(8), parameter :: exp_fmt = "(e20.8)"
  !> shorter float format
  character(8), parameter :: dp_fmt = "(f20.8)"
  !> integer format
  character(4), parameter :: int_fmt  = "(i8)"

  interface str
    module procedure logical_tostring
    module procedure integer_tostring
    module procedure real_tostring
    module procedure complex_tostring
    module procedure character_array_tostring
    module procedure integer_array_tostring
    module procedure real_array_tostring
  end interface str

  type, private :: logger_t
    integer, private :: logging_level
    logical, private :: use_prefix

  contains

    procedure, nopass, public :: error
    procedure, public :: warning
    procedure, public :: info
    procedure, public :: debug

    procedure, public :: initialise
    procedure, public :: set_logging_level
    procedure, public :: get_logging_level
    procedure, public :: enable_prefix
    procedure, public :: disable_prefix

  end type logger_t

  type(logger_t), public :: logger

  public :: str
  public :: exp_fmt, dp_fmt, int_fmt

contains

  pure subroutine initialise(this)
    class(logger_t), intent(inout) :: this
    this%logging_level = 2
    this%use_prefix = .true.
  end subroutine initialise


  pure subroutine set_logging_level(this, logging_level)
    class(logger_t), intent(inout) :: this
    integer, intent(in) :: logging_level
    this%logging_level = logging_level
  end subroutine set_logging_level


  pure integer function get_logging_level(this) result(logging_level)
    class(logger_t), intent(in) :: this
    logging_level = this%logging_level
  end function get_logging_level


  pure subroutine enable_prefix(this)
    class(logger_t), intent(inout) :: this
    this%use_prefix = .true.
  end subroutine enable_prefix


  pure subroutine disable_prefix(this)
    class(logger_t), intent(inout) :: this
    this%use_prefix = .false.
  end subroutine disable_prefix


  subroutine error(msg)
    use mod_exceptions, only: raise_exception
    character(len=*), intent(in) :: msg
    call raise_exception(msg)
  end subroutine error


  subroutine warning(this, msg)
    class(logger_t), intent(in) :: this
    character(len=*), intent(in) :: msg
    character(:), allocatable :: msg_raised

    if (.not. this%logging_level >= 1) return
    if (this%use_prefix) then
      msg_raised = " WARNING | " // msg
    else
      msg_raised = "         | " // msg
    end if
    write(*, *) paint_string(msg_raised, "yellow")
  end subroutine warning


  subroutine info(this, msg)  ! LCOV_EXCL_START
    class(logger_t), intent(in) :: this
    character(len=*), intent(in) :: msg
    character(:), allocatable :: msg_raised

    if (.not. this%logging_level >= 2) return
    if (this%use_prefix) then
      msg_raised = " INFO    | " // msg
    else
      msg_raised = "         | " // msg
    end if
    write(*, *) msg_raised
  end subroutine info  ! LCOV_EXCL_STOP


  subroutine debug(this, msg)  ! LCOV_EXCL_START
    class(logger_t), intent(in) :: this
    character(len=*), intent(in) :: msg
    character(:), allocatable :: msg_raised

    if (.not. this%logging_level >= 3) return
    if (this%use_prefix) then
      msg_raised = " DEBUG   | " // msg
    else
      msg_raised = "         | " // msg
    end if
    write(*, *) paint_string(msg_raised, "green")
  end subroutine debug  ! LCOV_EXCL_STOP


  pure function logical_tostring(boolean) result(string)
    logical, intent(in) :: boolean
    character(:), allocatable :: string
    if (boolean) then
      string = "True"
    else
      string = "False"
    end if
  end function logical_tostring


  pure function integer_tostring(int_value, fmt) result(string)
    integer, intent(in) :: int_value
    character(len=*), intent(in), optional :: fmt
    character(len=20) :: tmp
    character(:), allocatable :: fmt_string
    character(:), allocatable :: string

    if (present(fmt)) then
      fmt_string = "(" // trim(fmt) // ")"
    else
      fmt_string = int_fmt
    end if
    write(tmp, fmt_string) int_value
    string = trim(adjustl(tmp))
  end function integer_tostring


  pure function real_tostring(real_value, fmt) result(string)
    real(dp), intent(in) :: real_value
    character(len=*), intent(in), optional :: fmt
    character(len=20) :: tmp
    character(:), allocatable :: fmt_string
    character(:), allocatable :: string

    if (present(fmt)) then
      fmt_string = "(" // trim(fmt) // ")"
    else
      fmt_string = dp_fmt
    end if
    write(tmp, fmt_string) real_value
    string = trim(adjustl(tmp))
  end function real_tostring


  pure function complex_tostring(complex_value, fmt) result(string)
    complex(dp), intent(in) :: complex_value
    character(len=*), intent(in), optional :: fmt
    character(:), allocatable :: string
    character(:), allocatable :: fmt_string
    character(len=20) :: str_real, str_imag

    if (present(fmt)) then
      fmt_string = "(" // trim(fmt) // ")"
    else
      fmt_string = "(f18.8)"
    end if
    write(str_real, fmt_string) real(complex_value)
    write(str_imag, "(SP," // fmt_string // ",'i')") aimag(complex_value)
    string = trim(adjustl(str_real)) // trim(adjustl(str_imag))
  end function complex_tostring


  pure function character_array_tostring(array) result(string)
    !> the array to convert
    character(len=*), intent(in)  :: array(:)
    !> returned result, trimmed
    character(:), allocatable :: string
    integer :: i

    string = "["
    do i = 1, size(array)
      string = string // trim(array(i)) // ", "
    end do
    string = string(:len(string) - 2) // "]"
  end function character_array_tostring


  pure function integer_array_tostring(array) result(string)
    !> the array to convert
    integer, intent(in)  :: array(:)
    !> returned result, trimmed
    character(:), allocatable :: string
    integer :: i

    string = "["
    do i = 1, size(array)
      string = string // integer_tostring(array(i)) // ", "
    end do
    string = string(:len(string) - 2) // "]"
  end function integer_array_tostring


  pure function real_array_tostring(array, fmt) result(string)
  !> The input array of real numbers
  real(dp), intent(in) :: array(:)         
  !> Optional format for real numbers
  character(len=*), intent(in), optional :: fmt

  character(:), allocatable :: string
  integer :: i

  string = "["
  do i = 1, size(array)
    string = string // real_tostring(array(i), fmt) // ", "
  end do

  ! Remove the trailing ", " and close the array with "]"
  if (len(string) > 1) then
    string = string(:len(string) - 2) // "]"
  else
    string = "[]"
  end if

end function real_array_tostring


end module mod_logging
