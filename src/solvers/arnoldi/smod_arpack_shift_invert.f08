! =============================================================================
!> Module containing the implementation for the ARPACK shift-invert-type solver,
!! that is, given the general eigenvalue problem $$ AX = \omega BX, $$ choose a
!! shift \(\sigma\) and solve the problem $$ (A - \sigma B)X = \omega X, $$ thereby
!! finding \(k\) eigenvalues of the shifted problem that satisfy a given criterion.
submodule (mod_solvers:smod_arpack_main) smod_arpack_shift_invert
  use mod_banded_matrix, only: banded_matrix_t
  use mod_banded_operations, only: multiply
  use mod_transform_matrix, only: matrix_to_banded
  implicit none

contains

  !> Implementation of the ARPACK shift-invert solver
  module procedure solve_arpack_shift_invert
    !> contains the basis vectors
    complex(dp) :: basis_vectors(arpack_cfg%get_evpdim(), arpack_cfg%get_ncv())
    !> work array of length 3N
    complex(dp) :: workd(3 * arpack_cfg%get_evpdim())
    !> work array
    complex(dp) :: workl(arpack_cfg%get_lworkl())
    !> work array of length ncv
    real(dp) :: rwork(arpack_cfg%get_ncv())
    !> integer array with pointers to mark work array locations
    integer :: ipntr(14)
    !> logical array of dimension ncv, sets which Ritz vectors to compute
    logical :: select_vectors(arpack_cfg%get_ncv())
    !> work array of length 2*ncv
    complex(dp) :: workev(2 * arpack_cfg%get_ncv())

    integer :: diags
    logical :: converged
    type(banded_matrix_t) :: amat_min_sigmab_banded
    type(banded_matrix_t) :: bmat_banded
    integer :: xstart, xend, ystart, yend
    complex(dp) :: bxvector(arpack_cfg%get_evpdim())
    complex(dp), allocatable :: amat_min_sigmab_LU(:, :)
    integer, allocatable :: ipiv_LU(:)
    complex(dp) :: sigma

    sigma = settings%solvers%sigma
    call logger%debug("creating banded A - sigma*B")
    diags = settings%dims%get_dim_quadblock() - 1
    call matrix_to_banded(matrix_A, diags, diags, amat_min_sigmab_banded)
    call matrix_to_banded(matrix_B, diags, diags, bmat_banded)

    ! check compatibility
    if (.not. amat_min_sigmab_banded%is_compatible_with(bmat_banded)) then
      call logger%error("Arnoldi shift-invert: banded matrices are not compatible!")
      call amat_min_sigmab_banded%destroy()
      call bmat_banded%destroy()
      return
    end if

    ! form A - sigma*B, we ensured the banded matrices are compatible
    amat_min_sigmab_banded%AB = amat_min_sigmab_banded%AB - sigma * bmat_banded%AB
    call get_LU_factorisation_banded( &
      bandmatrix=amat_min_sigmab_banded, LU=amat_min_sigmab_LU, ipiv=ipiv_LU &
    )

    call logger%debug("doing Arnoldi shift-invert")
    converged = .false.
    do while(.not. converged)
      call znaupd( &
        arpack_cfg%ido, &
        arpack_cfg%get_bmat(), &
        arpack_cfg%get_evpdim(), &
        arpack_cfg%get_which(), &
        arpack_cfg%get_nev(), &
        arpack_cfg%get_tolerance(), &
        arpack_cfg%residual, &
        arpack_cfg%get_ncv(), &
        basis_vectors, &
        size(basis_vectors, dim=1), &
        arpack_cfg%iparam, &
        ipntr, &
        workd, &
        workl, &
        arpack_cfg%get_lworkl(), &
        rwork, &
        arpack_cfg%info &
      )

      ! x is given by workd(ipntr(1))
      xstart = ipntr(1)
      xend = xstart + arpack_cfg%get_evpdim() - 1
      ! y is given by workd(ipntr(2))
      ystart = ipntr(2)
      yend = ystart + arpack_cfg%get_evpdim() - 1

      select case(arpack_cfg%ido)
      case(-1, 1)
        ! ido = -1 on first call, forces starting vector in OP range
        ! get y <--- OP * x
        ! we need R = OP*x = inv[A - sigma*B]*B*x
        ! 1. calculate u = B*x
        ! 2. solve linear system [A - sigma*B] * R = u for R
        bxvector = multiply(bmat_banded, workd(xstart:xend))
        workd(ystart:yend) = solve_linear_system_complex_banded_LU( &
          bandmatrix=amat_min_sigmab_banded, &
          vector=bxvector, &
          LU=amat_min_sigmab_LU, &
          ipiv=ipiv_LU &
        )
      case default
        ! when convergence is achieved or maxiter is reached
        exit
      end select
    end do

    deallocate(ipiv_LU)
    deallocate(amat_min_sigmab_LU)
    call bmat_banded%destroy()
    call amat_min_sigmab_banded%destroy()

    ! check info parameter from znaupd, this errors if necessary
    call arpack_cfg%parse_znaupd_info(converged)
    ! if we have a normal exit, extract the eigenvalues through zneupd
    call zneupd( &
      .true., &  ! always calculate eigenvectors, negligible additional cost in ARPACK
      "A", &  ! calculate Ritz vectors
      select_vectors, &
      omega(1:arpack_cfg%get_nev()), &
      vr(:, 1:arpack_cfg%get_nev()), &
      size(vr, dim=1), &
      sigma, &
      workev, &
      arpack_cfg%get_bmat(), &
      arpack_cfg%get_evpdim(), &
      arpack_cfg%get_which(), &
      arpack_cfg%get_nev(), &
      arpack_cfg%get_tolerance(), &
      arpack_cfg%residual, &
      arpack_cfg%get_ncv(), &
      basis_vectors, &
      size(basis_vectors, dim=1), &
      arpack_cfg%iparam, &
      ipntr, &
      workd, &
      workl, &
      arpack_cfg%get_lworkl(), &
      rwork, &
      arpack_cfg%info &
    )

    call logger%debug( &
      "performing eigenvalue backtransformation to original problem (nu -> omega)" &
    )
    !> @note
    !!     In applying shift-invert we made the transformation \(C = inv[B]*A\) and
    !!     solved the standard eigenvalue problem \(CX = \nu X\) instead since B isn't
    !!     always Hermitian (e.g. if we include Hall).
    !!     According to the ARPACK documentation, section 3.2.2, this
    !!     implies that we must manually transform the eigenvalues \(\nu_j\) from \(C\)
    !!     to the eigenvalues \(\omega_j\) from the original system. This uses the relation
    !!     $$ \omega_j = \sigma + \frac{1}{\nu_j} $$
    !! @endnote
    omega = sigma + (1.0d0 / omega)

    call arpack_cfg%parse_zneupd_info()
    call arpack_cfg%parse_finished_stats()
  end procedure solve_arpack_shift_invert

end submodule smod_arpack_shift_invert
