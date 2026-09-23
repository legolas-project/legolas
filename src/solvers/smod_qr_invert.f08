! =============================================================================
!> Submodule containing the implementation of the QR-invert algorithm.
!! The original problem is written as a standard eigenvalue problem through
!! $$ \mathcal{B}^{-1}\mathcal{A}\textbf{X} = \omega\textbf{X}\ $$. This is
!! done using a LU decomposition via LAPACKS's <tt>zgbsv</tt>.
!! Eventually a call to LAPACK's <tt>zgeev</tt> routine is done to obtain
!! all eigenvalues and eigenvectors.
submodule (mod_solvers) smod_qr_invert
  use mod_banded_matrix, only: banded_matrix_t, new_banded_matrix
  use mod_transform_matrix, only: matrix_to_banded
  implicit none

contains

  !> Solves the eigenvalue problem by rewriting it to a standard form
  !! through inversion of the B-matrix.
  !! @warning
  !!     Throws an error if <tt>matrix_A</tt> or <tt>matrix_B</tt>
  !!     is not a square matrix.
  !! @endwarning
  module procedure qr_invert
    !> full array containing the \(B^{-1}A\)-matrix
    complex(dp), allocatable :: array_B_invA(:, :)
    !> pivoting array
    integer :: ipiv(matrix_B%matrix_dim)
    !> banded B-matrix
    type(banded_matrix_t) :: B_band
    !> order of matrices
    integer :: N
    !> calculate right eigenvectors if "V", omit if "N"
    character :: jobvr
    !> dimension of work array
    integer :: lwork
    !> work array
    complex(dp), allocatable  :: work(:)
    !> info parameter, 0 on successful exit
    integer :: info
    !> second work array
    real(dp), allocatable :: rwork(:)
    !> dummy for left eigenvectors, jobvl = "N" so this is never referenced
    complex(dp) :: vl(2, 2)
    !> number of superdiagonals
    integer :: ku
    !> number of subdiagonals
    integer :: kl
    integer :: irow, icol

    call matrix_B%get_nb_diagonals(ku=ku, kl=kl)
    call logger%debug( &
      "B has " // str(ku) // " superdiagonals and " // str(kl) // " subdiagonals" &
    )
    N = matrix_B%matrix_dim
    call logger%debug("converting B to banded form")
    ! for zgbsv later on we need double the amount of subdiagonals
    B_band = new_banded_matrix(rows=N, cols=N, subdiags=2*kl, superdiags=ku)
    ! fill banded matrix, see documentation: "The j-th column of B is stored in the
    ! j-th column of the array AB as follows:
    ! AB(KL+KU+1+i-j,j) = B(i,j) for max(1,j-KU)<=i<=min(N,j+KL)"
    do icol = 1, N
      do irow = max(1, icol - ku), min(N, icol + kl)
        B_band%AB(kl+ku+1+irow-icol, icol) = matrix_B%get_complex_element(irow, icol)
      end do
    end do

    allocate(array_B_invA(matrix_A%matrix_dim, matrix_A%matrix_dim))
    call logger%debug("converting A to dense form")
    call matrix_to_array(matrix=matrix_A, array=array_B_invA)

    ! zgbsv calculates X from BX = A, where B is a banded matrix.
    ! solving this system yields X = B^{-1}A without explicitly inverting
    call logger%debug("calling LAPACK's zgbsv")
    call zgbsv( &
      B_band%m, &
      kl, &
      ku, &
      matrix_A%matrix_dim, &
      B_band%AB, &
      size(B_band%AB, dim=1), &
      ipiv, &
      array_B_invA, &
      size(array_B_invA, dim=1), &
      info &
    )
    if (info /= 0) then
      call logger%error("LAPACK's zgbsv failed with info = " // str(info))
      return
    end if
    call B_band%destroy()

    ! calculate eigenvectors, we don't use the left ones
    jobvr = "N"
    if (settings%io%should_compute_eigenvectors()) jobvr = "V"
    ! allocate rwork array
    allocate(rwork(2 * N))
    ! get lwork
    call logger%debug("calculating optimal length of work array")
    allocate(work(1))
    call zgeev( &
      "N", &
      jobvr, &
      N, &
      array_B_invA, &
      N, &
      omega, &
      vl, &
      size(vl, dim=1), &
      vr, &
      size(vr, dim=1), &
      work, &
      -1, &
      rwork, &
      info &
    )
    lwork = int(work(1))
    deallocate(work)
    ! allocate work array
    call logger%debug("allocating work array with N = " // str(lwork))
    allocate(work(lwork))

    ! solve eigenvalue problem
    call logger%debug("solving evp using QR algorithm zgeev (LAPACK)")
    call zgeev( &
      "N", &
      jobvr, &
      N, &
      array_B_invA, &
      N, &
      omega, &
      vl, &
      size(vl, dim=1), &
      vr, &
      size(vr, dim=1), &
      work, &
      lwork, &
      rwork, &
      info &
    )
    if (info /= 0) then ! LCOV_EXCL_START
      call logger%warning("LAPACK routine zgeev failed!")
      call logger%warning("value for the info parameter: " // str(info))
    end if ! LCOV_EXCL_STOP

    ! tear down work arrays
    deallocate(array_B_invA)
    deallocate(work)
    deallocate(rwork)
  end procedure qr_invert

end submodule smod_qr_invert
