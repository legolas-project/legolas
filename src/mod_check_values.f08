! =============================================================================
!> This module contains various methods to check for small, NaN or negative values,
!! equal values or inf values. Interfaces are provided for functionality with
!! real and complex variables.
module mod_check_values
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_is_finite
  use mod_global_variables, only: dp, dp_LIMIT
  implicit none

  private

  !> interface to check for small values
  interface set_small_values_to_zero
    module procedure small_values_real
    module procedure small_values_complex
  end interface set_small_values_to_zero

  !> interface to check for NaN values
  interface is_NaN
    module procedure contains_NaN_real
    module procedure contains_NaN_complex
  end interface is_NaN

  !> interface to check for inf values
  interface is_infinite
    module procedure contains_inf_real
    module procedure contains_inf_complex
  end interface is_infinite

  !> interface to check equality between values/arrays
  interface is_equal
    module procedure real_is_equal
    module procedure complex_is_equal
  end interface is_equal

  !> interface to check if values/arrays are zero
  interface is_zero
    module procedure real_is_zero
    module procedure complex_is_zero
  end interface is_zero

  !> interface to check for negative values
  interface is_negative
    module procedure real_is_negative
  end interface is_negative

  !> interface to check if an array is constant
  interface is_constant
    module procedure real_array_is_constant
  end interface is_constant

  public :: set_small_values_to_zero
  public :: is_NaN
  public :: is_infinite
  public :: is_equal
  public :: is_zero
  public :: is_negative
  public :: is_constant

contains

  !> Small value checks for a real variable/array/matrix. Values that are
  !! smaller than the specified tolerance <tt>tol</tt> are set to zero. If
  !! <tt>tol</tt> is not present, <tt>DP_LIMIT</tt> is used as tolerance.
  elemental subroutine small_values_real(var, tol)
    !> the real variable/array/matrix to check, modified on output
    real(dp), intent(inout) :: var
    !> optional tolerance to check against
    real(dp), intent(in), optional  :: tol
    real(dp) :: tolerance

    if (present(tol)) then
      tolerance = tol
    else
      tolerance = dp_LIMIT
    end if
    if (abs(var) < tolerance) then
      var = 0.0d0
    end if
  end subroutine small_values_real


  !> Small value checks for a complex variable/array/matrix, with the real and
  !! imaginary parts checked separately. Values that are
  !! smaller than the specified tolerance <tt>tol</tt> are set to zero. If
  !! <tt>tol</tt> is not present, <tt>DP_LIMIT</tt> is used as tolerance.
  elemental subroutine small_values_complex(var, tol)
    !> the complex variable/array/matrix to check, modified on output
    complex(dp), intent(inout) :: var
    !> optional tolerance to check against
    real(dp), intent(in), optional  :: tol
    real(dp) :: tolerance
    real(dp) :: var_real, var_imag

    if (present(tol)) then
      tolerance = tol
    else
      tolerance = dp_LIMIT
    end if
    var_real = real(var)
    if (abs(var_real) < tolerance) then
      var_real = 0.0d0
    end if
    var_imag = aimag(var)
    if (abs(var_imag) < tolerance) then
      var_imag = 0.0d0
    end if
    var = cmplx(var_real, var_imag, kind=dp)
  end subroutine small_values_complex


  !> Checks a given real value/array/matrix for NaN.
  elemental logical function contains_NaN_real(var)
    !> the real variable/array/matrix to check
    real(dp), intent(in)  :: var

    contains_NaN_real = (ieee_is_nan(var))
  end function contains_NaN_real


  !> Checks a given complex value/array/matrix for NaN.
  elemental logical function contains_NaN_complex(var)
    !> the complex variable/array/matrix to check
    complex(dp), intent(in) :: var

    contains_NaN_complex = (ieee_is_nan(real(var)) .or. ieee_is_nan(aimag(var)))
  end function contains_NaN_complex


  !> checks a given real value/array/matrix for infinity.
  elemental logical function contains_inf_real(var)
    !> the real variable/array/matrix to check
    real(dp), intent(in)  :: var

    contains_inf_real = (.not. ieee_is_finite(var))
  end function contains_inf_real


  !> checks a given complex value/array/matrix for infinity.
  elemental logical function contains_inf_complex(var)
    !> the complex variable/array/matrix to check
    complex(dp), intent(in) :: var

    contains_inf_complex = ( &
      .not. ieee_is_finite(real(var)) &
      .or. .not. ieee_is_finite(aimag(var)) &
    )
  end function contains_inf_complex


  !> Equality check between real values
  elemental logical function real_is_equal(value, base, tol)
    !> the real value(s) to check
    real(dp), intent(in)  :: value
    !> the value(s) to compare against
    real(dp), intent(in)  :: base
    !> optional tolerance
    real(dp), intent(in), optional :: tol
    real(dp)  :: tolerance

    if (present(tol)) then
      tolerance = tol
    else
      tolerance = dp_LIMIT
    end if
    real_is_equal = (abs(value - base) <= tolerance)
  end function real_is_equal


  !> Equality check between complex values
  elemental logical function complex_is_equal(value, base, tol)
    !> the real value(s) to check
    complex(dp), intent(in)  :: value
    !> the value(s) to compare against
    complex(dp), intent(in)  :: base
    !> optional tolerance
    real(dp), intent(in), optional :: tol
    real(dp)  :: tolerance

    if (present(tol)) then
      tolerance = tol
    else
      tolerance = dp_LIMIT
    end if
    complex_is_equal = ( &
      abs(real(value - base)) <= tolerance &
      .and. abs(aimag(value - base)) <= tolerance &
    )
  end function complex_is_equal


  !> Checks if real values are zero
  elemental logical function real_is_zero(value, tol)
    !> the real value(s) to check
    real(dp), intent(in)  :: value
    !> optional tolerance
    real(dp), intent(in), optional :: tol

    if (present(tol)) then
      real_is_zero = is_equal(value, 0.0d0, tol=tol)
    else
      real_is_zero = is_equal(value, 0.0d0)
    end if
  end function real_is_zero


  !> Checks if complex values are zero
  elemental logical function complex_is_zero(value, tol)
    !> the complex value(s) to check
    complex(dp), intent(in) :: value
    !> optional tolerance
    real(dp), intent(in), optional  :: tol

    if (present(tol)) then
      complex_is_zero = is_equal(value, (0.0d0, 0.0d0), tol=tol)
    else
      complex_is_zero = is_equal(value, (0.0d0, 0.0d0))
    end if
  end function complex_is_zero


  !> Check if values are or contain negative numbers
  elemental logical function real_is_negative(value)
    !> the real value(s) to check
    real(dp), intent(in)  :: value

    real_is_negative = (value < 0.0d0)
  end function real_is_negative


  !> Check if an array has constant values
  logical function real_array_is_constant(array, tol)
    !> the real array to check
    real(dp), intent(in) :: array(:)
    !> optional tolerance
    real(dp), intent(in), optional :: tol
    real(dp) :: tolerance

    if (present(tol)) then
      tolerance=tol
    else
      tolerance=dp_LIMIT
    end if

    real_array_is_constant = all(abs(array - array(1)) <= tolerance)
  end function real_array_is_constant

end module mod_check_values
