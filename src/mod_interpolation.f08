! =============================================================================
!> Module responsible for table interpolations and array lookups.
!! Contains subroutines for table interpolations, numerical derivatives
!! of arrays and lookup functions.
!! Subroutines are loosely based on routines implemented in the
!! [MPI-AMRVAC](amrvac.org) code.
module mod_interpolation
  use mod_global_variables, only: dp
  use mod_logging, only: logger
  implicit none

  private

  public :: interpolate_table
  public :: get_numerical_derivative
  public :: get_second_numerical_derivative
  public :: lookup_table_value

contains


  !> Interpolates a given set of tables (x, y(x)) into a smooth curve.
  !! Assumes that x_table is an array with a monotone increase in values.
  !! Interpolation is done using <tt>n_interp</tt> points, in general a second
  !! order polynomial approximation is used except near sharp jumps.
  !! @warning Throws an error if <tt>x_table</tt> is not monotone. @endwarning
  subroutine interpolate_table(n_interp, x_table, y_table, x_interp, y_interp)
    !> number of points used for interpolation
    integer, intent(in)   :: n_interp
    !> x-values in the table
    real(dp), intent(in)  :: x_table(:)
    !> y-values in the table
    real(dp), intent(in)  :: y_table(:)
    !> interpolated x-values
    real(dp), intent(out) :: x_interp(n_interp)
    !> interpolated y-values
    real(dp), intent(out) :: y_interp(n_interp)

    integer   :: i, j, n_table
    real(dp)  :: fact1, fact2, fact3
    real(dp)  :: xmin, xmax
    real(dp)  :: dx, dy1, dy2
    logical   :: jump

    n_table = size(x_table)
    ! check if x_table is a monotonically increasing array
    do i = 1, n_table - 1
      if (x_table(i + 1) < x_table(i)) then
        call logger%error("interpolation: x-values are not monotonically increasing!")
        return
      end if
    end do

    xmin = x_table(1)
    xmax = x_table(n_table)
    ! outer edges remain the same
    x_interp(1) = xmin
    x_interp(n_interp) = xmax
    y_interp(1) = y_table(1)
    y_interp(n_interp) = y_table(n_table)

    dx = (xmax - xmin) / (n_interp - 1)
    do i = 2, n_interp - 1
      x_interp(i) = x_interp(i - 1) + dx
      do j = 1, n_table - 1
        jump = .false.
        if (x_interp(i) < x_table(j + 1)) then
          if (j < n_table - 1) then
            ! check if we have a sharp jump
            dy1 = y_table(j + 1) - y_table(j)
            dy2 = y_table(j + 2) - y_table(j + 1)
            jump = ( max(dabs(dy1), dabs(dy2)) > 2 * min(dabs(dy1), dabs(dy2)) )
          end if
          ! no interpolation at outer edge or near sharp jumps
          if ((j == n_table - 1) .or. jump) then
            fact1 = (x_interp(i) - x_table(j + 1)) / (x_table(j) - x_table(j + 1))
            fact2 = (x_interp(i) - x_table(j)) / (x_table(j + 1) - x_table(j))
            y_interp(i) = fact1 * y_table(j) + fact2 * y_table(j + 1)
            exit
          end if

          fact1 = ((x_interp(i) - x_table(j + 1)) * (x_interp(i) - x_table(j + 2))) &
                / ((x_table(j) - x_table(j + 1)) * (x_table(j) - x_table(j + 2)))
          fact2 = ((x_interp(i) - x_table(j)) * (x_interp(i) - x_table(j + 2))) &
                / ((x_table(j + 1) - x_table(j)) * (x_table(j + 1) - x_table(j + 2)))
          fact3 = ((x_interp(i) - x_table(j)) * (x_interp(i) - x_table(j + 1))) &
                / ((x_table(j + 2) - x_table(j)) * (x_table(j + 2) - x_table(j + 1)))
          y_interp(i) = fact1 * y_table(j) + fact2 * y_table(j + 1) &
                      + fact3 * y_table(j + 2)
          exit
        end if
      end do
    end do
  end subroutine interpolate_table


  !> Calculates the numerical derivative of a given array.
  !! A sixth-order accurate central difference stencil is used to calculate the
  !! derivative. Near the edges a sixth-order accurate forward and backward
  !! difference stencil is used for the left and right boundary, respectively.
  !! It is assumed that the x values are all equally spaced. If this is not the case,
  !! a polynomial interpolation on a uniform grid can be done and that one can be
  !! differentiated instead. The stencils are as follows:
  !!
  !! - 6th order central differences:
  !!   $$ dy_i = \frac{-y_{i-3} + 9y_{i-2} - 45y_{i-1} + 45y_{i+1}
  !!                - 9y_{i+2} + y_{i+3}}{60dx} $$
  !! - 6th order forward differences:
  !!   $$ dy_i = \frac{-147y_i + 360y_{i+1} - 450y_{i+2} + 400y_{i+3}
  !!                - 225y_{i+4} + 72y_{i+5} - 10y_{i+6}}{60dx} $$
  !! - 6th order backward differences:
  !!   $$ dy_i = \frac{10y_{i-6} - 72y_{i-5} + 225y_{i-4} - 400y_{i-3}
  !!                + 450y_{i-2} - 360y_{i-1} + 147y_i}{60dx} $$
  !!
  !! @warning Throws an error if <tt>x_values</tt> and <tt>y_values</tt> differ
  !!          in size. @endwarning
  subroutine get_numerical_derivative(x, y, dy, dxtol)
    use mod_check_values, only: is_equal
    use mod_logging, only: str

    !> x-values against which to differentiate
    real(dp), intent(in)  :: x(:)
    !> array of y-values, assuming \(y(x)\) relation
    real(dp), intent(in)  :: y(:)
    !> derivative of \(y\) with respect to \(x\), same size as input arrays
    real(dp), intent(out) :: dy(size(y))
    !> optional tolerance for equally spaced arrays
    real(dp), intent(in), optional  :: dxtol

    integer   :: i, nvals, nbprints
    real(dp)  :: dx, dxi, tol

    ! x_values and y_values should be the same length
    if (size(x) /= size(y)) then
      call logger%error("numerical derivative: x and y should have the same size!")
      return
    end if
    nbprints = 0
    tol = 1.0d-10
    if (present(dxtol)) then
      tol = dxtol ! LCOV_EXCL_LINE
    end if

    nvals = size(x)
    dx = x(2) - x(1)
    ! LCOV_EXCL_START
    do i = 2, nvals-1
      dxi = x(i) - x(i-1)
      if (.not. is_equal(dx, dxi, tol=tol)) then
        call logger%warning( &
          "numerical derivative: x is not equally spaced, derivative may be wrong!" &
        )
        call logger%warning( &
          "at index " // str(i) // " expected dx=" // str(dx, fmt="e20.10") // &
          " but got dx=" // str(dxi, fmt="e20.10") &
        )
        call logger%warning("---> diff = " // str(abs(dx - dxi), fmt="e20.6"))
        nbprints = nbprints + 1
        if (nbprints == 10) then
          call logger%warning("...")
          exit
        end if
      end if
    end do
    ! LCOV_EXCL_STOP

    ! left side: 6th order forward differences for first 3 points
    do i = 1, 3
      dy(i) = ( &
        -147 * y(i) &
        + 360 * y(i + 1) &
        - 450 * y(i + 2) &
        + 400 * y(i + 3) &
        - 225 * y(i + 4) &
        + 72 * y(i + 5) &
        - 10 * y(i + 6)&
      ) / (60 * dx)
    end do
    ! middle: 6th order central differences
    do i = 4, nvals-3
      dy(i) = ( &
        -y(i - 3) &
        + 9 * y(i - 2) &
        - 45 * y(i - 1) &
        + 45 * y(i + 1) &
        - 9 * y(i + 2) &
        + y(i + 3) &
      ) / (60 * dx)
    end do
    ! right side: 6th order backwards differences for last 3 points
    do i = nvals-2, nvals
      dy(i) = ( &
        10 * y(i - 6) &
        - 72 * y(i - 5) &
        + 225 * y(i - 4) &
        - 400 * y(i - 3) &
        + 450 * y(i - 2) &
        - 360 * y(i - 1) &
        + 147 * y(i) &
      ) / (60 * dx)
    end do
  end subroutine get_numerical_derivative


  !> Calculates the second numerical derivative of a given array.
  !! A sixth-order accurate central difference stencil is used to calculate the
  !! derivative. Near the edges a sixth-order accurate forward and backward
  !! difference stencil is used for the left and right boundary, respectively.
  !! It is assumed that the x values are all equally spaced. If this is not the case,
  !! a polynomial interpolation on a uniform grid can be done and that one can be
  !! differentiated instead.
  !!
  !! @warning Throws an error if <tt>x_values</tt> and <tt>y_values</tt> differ
  !!          in size. @endwarning
  subroutine get_second_numerical_derivative(x, y, dy, dxtol)
    use mod_check_values, only: is_equal
    use mod_logging, only: str

    !> x-values against which to differentiate
    real(dp), intent(in)  :: x(:)
    !> array of y-values, assuming \(y(x)\) relation
    real(dp), intent(in)  :: y(:)
    !> derivative of \(y\) with respect to \(x\), same size as input arrays
    real(dp), intent(out) :: dy(size(y))
    !> optional tolerance for equally spaced arrays
    real(dp), intent(in), optional  :: dxtol

    integer   :: i, nvals, nbprints
    real(dp)  :: dx, dxi, tol

    ! x_values and y_values should be the same length
    if (size(x) /= size(y)) then
      call logger%error("numerical derivative: x and y should have the same size!")
      return
    end if
    nbprints = 0
    tol = 1.0d-10
    if (present(dxtol)) then
      tol = dxtol ! LCOV_EXCL_LINE
    end if

    nvals = size(x)
    dx = x(2) - x(1)
    ! LCOV_EXCL_START
    do i = 2, nvals-1
      dxi = x(i) - x(i-1)
      if (.not. is_equal(dx, dxi, tol=tol)) then
        call logger%warning( &
          "numerical derivative: x is not equally spaced, derivative may be wrong!" &
        )
        call logger%warning( &
          "at index " // str(i) // " expected dx=" // str(dx, fmt="e20.10") // &
          " but got dx=" // str(dxi, fmt="e20.10") &
        )
        call logger%warning("---> diff = " // str(abs(dx - dxi), fmt="e20.6"))
        nbprints = nbprints + 1
        if (nbprints == 10) then
          call logger%warning("...")
          exit
        end if
      end if
    end do
    ! LCOV_EXCL_STOP

    ! left side: 6th order forward differences for first 3 points
    do i = 1, 3
      dy(i) = ( &
        938 * y(i) &
        - 4014 * y(i + 1) &
        + 7911 * y(i + 2) &
        - 9490 * y(i + 3) &
        + 7380 * y(i + 4) &
        - 3618 * y(i + 5) &
        + 1019 * y(i + 6) &
        - 126 * y(i + 7) &
      ) / (180 * dx**2)
    end do
    ! middle: 6th order central differences
    do i = 4, nvals-3
      dy(i) = ( &
        2 * y(i - 3) &
        - 27 * y(i - 2) &
        + 270 * y(i - 1) &
        - 490 * y(i) &
        + 270 * y(i + 1) &
        - 27 * y(i + 2) &
        + 2 * y(i + 3) &
      ) / (180 * dx**2)
    end do
    ! right side: 6th order backwards differences for last 3 points
    do i = nvals-2, nvals
      dy(i) = ( &
        - 126 * y(i - 7) &
        + 1019 * y(i - 6) &
        - 3618 * y(i - 5) &
        + 7380 * y(i - 4) &
        - 9490 * y(i - 3) &
        + 7911 * y(i - 2) &
        - 4014 * y(i - 1) &
        + 938 * y(i) &
      ) / (180 * dx**2)
    end do
  end subroutine get_second_numerical_derivative


  !> Function for fast table-lookup, returns the corresponding y-value
  !! in <tt>y_values</tt> based on a given based on a given \(x0\).
  !! If the <tt>allow_outside</tt> flag is given as <tt>.true.</tt> then values
  !! on the edge of the table are returned when the lookup value is outside the array.
  !! Uses simple linear interpolation.
  function lookup_table_value(x, x_values, y_values, allow_outside) result(y_found)
    use mod_global_variables, only: NaN

    !> value to look up
    real(dp), intent(in)  :: x
    !> array of x-values
    real(dp), intent(in)  :: x_values(:)
    !> array of y-values, assuming \(y(x)\) relation
    real(dp), intent(in)  :: y_values(:)
    !> flag to allow for lookups outside of the array
    logical, optional :: allow_outside
    !> interpolated y-value based on \(x0\)
    real(dp)  :: y_found

    integer   :: idx, nvals
    real(dp)  :: x0, x1, y0, y1
    logical   :: return_edge_value_if_outside

    nvals = size(x_values)
    return_edge_value_if_outside = .false.
    if (present(allow_outside)) then
      return_edge_value_if_outside = allow_outside
    end if

    if (return_edge_value_if_outside) then
      if (x < x_values(1)) then
        y_found = y_values(1)
        return
      else if (x > x_values(nvals)) then
        y_found = y_values(nvals)
        return
      end if
    end if

    ! check if we are outside of the table
    if (x < x_values(1)) then
      call logger%error("lookup_value: x outside x_values (too small)")
      y_found = NaN
      return
    else if (x > x_values(nvals)) then
      call logger%error("lookup_value: x outside x_values (too large)")
      y_found = NaN
      return
    end if

    ! index of nearest value to x (dim=1 to return a scalar)
    idx = minloc(abs(x_values - x), dim=1)

    ! check if we are on left or right side of nearest point, or on-edge
    if (x < x_values(idx) .or. idx == nvals) then
      x0 = x_values(idx - 1)
      x1 = x_values(idx)
      y0 = y_values(idx - 1)
      y1 = y_values(idx)
    else
      x0 = x_values(idx)
      x1 = x_values(idx + 1)
      y0 = y_values(idx)
      y1 = y_values(idx + 1)
    end if

    ! do linear interpolation
    y_found = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
  end function lookup_table_value

end module mod_interpolation
