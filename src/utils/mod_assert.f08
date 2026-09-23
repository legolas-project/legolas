!=================================================================
!> Module defining the assertion routine used by [[assert.fpp]].
!! @note
!!     See [[assert.fpp]] for usage of the `assert` macro. The
!!     subroutine in this module should not be used directly.
!! @endnote
module mod_assert
  use mod_logging, only: logger, str

  implicit none

contains

  !> Utility function used by [[assert.fpp]]'s `assert` macro.
  !!
  !! If **cond** is false, log an error described by **cond_str**,
  !! **file** and **line**, terminating the program.
  !! @note
  !!     Do not use this subroutine directly, rather use
  !!     the `assert` macro from [[assert.fpp]].
!! @endnote
  subroutine assert(cond, cond_str, file, line)
    !> The assertion result.
    logical,       intent(in) :: cond
    !> The assertion code.
    character(*),  intent(in) :: cond_str
    !> The file containing the assertion.
    character(*),  intent(in) :: file
    !> The line of the assertion.
    integer,       intent(in) :: line

    if (.not. cond) call logger%error( &
      "assertion '" // cond_str // "' failed at " //  file // ":" // str(line) &
    )
  end subroutine assert

end module mod_assert
