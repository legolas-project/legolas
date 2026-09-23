module mod_iv_globals
  use mod_global_variables, only: dp  
  implicit none
  
  public

  abstract interface
    function profile_fcn(x) result(res)
      import dp
      real(dp), intent(in) :: x(:)
      real(dp) :: res(size(x))
    end function profile_fcn
  end interface

  ! Type to hold the profile function pointers
  type :: iv_fcn_ptr_t
    procedure(profile_fcn), pointer, nopass :: ptr
  end type iv_fcn_ptr_t

contains
  pure function linspace(x0, x1, xvals) result(xarray)
    real(dp), intent(in)  :: x0, x1
    integer, intent(in)   :: xvals
    real(dp) :: xarray(xvals)
    real(dp)  :: dx
    integer   :: i

    dx = (x1 - x0) / (xvals - 1)
    do i = 1, xvals
      xarray(i) = x0 + (i - 1) * dx
    end do
  end function linspace

  function zero_fcn(x) result(res)
    real(dp), intent(in) :: x(:)
    real(dp) :: res(size(x))

    res = 0.0d0
  end function zero_fcn


end module mod_iv_globals