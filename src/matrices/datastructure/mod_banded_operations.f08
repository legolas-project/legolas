module mod_banded_operations
  use mod_global_variables, only: dp
  use mod_logging, only: logger
  use mod_banded_matrix, only: banded_matrix_t
  implicit none

  private

  interface multiply
    module procedure banded_matrix_x_vector
  end interface multiply

  public :: multiply
  public :: banded_copy
  public :: constant_times_banded_plus_banded

contains

  !> Calculates the matrix-vector product of a general complex banded matrix and a
  !! complex vector. Uses the level 2 BLAS routine <tt>zgbmv</tt>.
  function banded_matrix_x_vector(bandmatrix, vector) result(rvector)
    !> the banded matrix
    type(banded_matrix_t), intent(in) :: bandmatrix
    !> the vector
    complex(dp), intent(in) :: vector(bandmatrix%n)
    !> the resulting matrix-vector product
    complex(dp) :: rvector(bandmatrix%m)

    call zgbmv( &
      "N", &
      bandmatrix%m, &
      bandmatrix%n, &
      bandmatrix%kl, &
      bandmatrix%ku, &
      (1.0_dp, 0.0_dp), &
      bandmatrix%AB, &
      size(bandmatrix%AB, dim=1), &
      vector, &
      1, &
      (0.0_dp, 0.0_dp), &
      rvector, &
      1 &
    )
  end function banded_matrix_x_vector

  !> Copy banded matrix B into A
  subroutine banded_copy(A, B)
    !> this banded matrix will be overwritten
    type(banded_matrix_t), intent(inout) :: A
    !> this banded matrix will be copied into A
    type(banded_matrix_t), intent(in)    :: B

    integer :: A_ku, A_kl, B_ku, B_kl
    integer :: i, j, m

    ! Check that A and B are compatible
    if (.not. (A%m == B%m .and. A%n == B%n)) then
      call logger%error("A or B not square, or not compatible")
      return
    end if

    m = B%m
    A_ku = A%ku
    A_kl = A%kl
    B_ku = B%ku
    B_kl = B%kl

    ! Check that A has enough diagonals to hold B
    if (.not. (A_ku >= B_ku .and. A_kl >= B_kl)) then
      call logger%error("A does not have enough sub or super diagonals to hold B")
      return
    end if

    ! Do the copy
    do j = 1, m
      do i = max(1, j - B_ku), min(m, j + B_kl)
        A%AB(A_ku + 1 + (i - j), j) = B%AB(B_ku + 1 + (i - j), j)
      end do
    end do

  end subroutine banded_copy

  !> Multiply a constant times a banded matrix plus another banded matrix;
  !! A = A + alpha*B
  subroutine constant_times_banded_plus_banded(A, B, alpha)
    type(banded_matrix_t), intent(inout) :: A
    type(banded_matrix_t), intent(in)    :: B
    complex(dp), intent(in)              :: alpha

    integer :: A_ku, A_kl, B_ku, B_kl
    integer :: i, j, m

    ! Check that A and B are compatible
    if (.not. (A%m == B%m .and. A%n == B%n)) then
      call logger%error("A or B not square, or not compatible")
      return
    end if

    m = A%m
    A_ku = A%ku
    A_kl = A%kl
    B_ku = B%ku
    B_kl = B%kl
    
    ! Check that A has enough diagonals to hold B
    if (.not. (A_ku >= B_ku .and. A_kl >= B_kl)) then
      call logger%error("A does not have enough sub or super diagonals to hold B")
      return
    end if

    ! Do the operation
    do j = 1, m
      do i = max(1, j - A_ku), min(m, j + A_kl)
        A%AB(A_ku + 1 + (i - j), j) = A%AB(A_ku + 1 + (i - j), j) + alpha*B%AB(B_ku + 1 + (i - j), j)
      end do
    end do
    
  end subroutine constant_times_banded_plus_banded

end module mod_banded_operations
