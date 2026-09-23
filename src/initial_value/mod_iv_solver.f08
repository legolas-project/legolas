! =============================================================================
!> Initial value solver
!!
!!
module mod_iv_solver
  use mod_logging, only: logger, str
  use mod_matrix_structure, only: matrix_t
  use mod_global_variables, only: dp, ic
  use mod_settings, only: settings_t

  use mod_banded_matrix, only: banded_matrix_t, new_banded_matrix
  use mod_banded_operations, only: banded_copy, constant_times_banded_plus_banded
  use mod_linear_systems, only: solve_linear_system_complex_banded_LU, &
    get_LU_factorisation_banded
  use mod_transform_matrix, only: matrix_to_banded
  implicit none

contains

  !> Solve the initial value problem
  subroutine solve(matrix_A, matrix_B, x, settings, snapshots, snap_times)
    !> FEM matrix A
    type(matrix_t) :: matrix_A
    !> FEM matrix B
    type(matrix_t) :: matrix_B
    !> initial condition, gets updated with final result
    complex(dp), dimension(:), intent(inout) :: x
    !> settings
    type(settings_t), intent(in) :: settings
    !> optionally save every n-th step in this 2D array
    complex(dp), dimension(:,:), optional, intent(out) :: snapshots  ! TODO: Make this a required arg
    !> snapshot times
    real(dp), dimension(:), optional, intent(out) :: snap_times  ! TODO: Make this a required arg

    type(banded_matrix_t) :: A, B, M
    integer :: A_ku, A_kl               ! # upper diagonals, # lower diagonals
    integer :: B_ku, B_kl
    integer :: M_ku, M_kl
    complex(dp) :: beta, gamma
    complex(dp), allocatable :: z(:)    ! intermediate result
    complex(dp), allocatable :: rhs(:)
    complex(dp), allocatable :: M_LU(:, :)  ! pre-factorised M (reused each step)
    integer, allocatable :: M_ipiv(:)
    integer :: n                        ! dimension of matrices and x
    integer :: i

    integer :: i_save                   ! snapshot counter
    integer :: num_save                 ! number of snapshots to store in hist
    integer :: save_stride              ! save every save_stride-th step
    real(dp) :: alpha                   ! solver implicitness
    real(dp) :: dt
    real(dp) :: t_end
    integer :: num_steps


    ! Check input sanity
    if (.not. (matrix_A%matrix_dim == matrix_B%matrix_dim)) then
      call logger%error("A or B not square, or not compatible")
      return
    end if

    if (.not. (matrix_A%matrix_dim == size(x))) then
      call logger%error("Initial condition x0 size is not compatible with A, B matrices")
    end if

    alpha = settings%iv%alpha
    save_stride = settings%iv%snapshot_stride
    t_end = settings%iv%t_end
    num_steps = settings%iv%n_steps
    dt = settings%iv%get_step_size()

    n = matrix_A%matrix_dim
    call matrix_A%get_nb_diagonals(ku=A_ku, kl=A_kl)
    call matrix_B%get_nb_diagonals(ku=B_ku, kl=B_kl)

    ! We will only work with banded matrices
    call matrix_to_banded(matrix_A, A_kl, A_ku, A)
    call matrix_to_banded(matrix_B, B_kl, B_ku, B)

    ! We need to restore the explicit time derivative by multiplying A by -i
    call transform_matrix(A)

    allocate(rhs, mold = x)
    allocate(z, mold = x)

    ! Figure out how many snapshots to store
    if (present(snapshots)) then
      num_save = floor(real((num_steps - 1))/real(save_stride)) + 2
      if (size(snapshots, dim=1) /= n .or. size(snapshots, dim=2) < num_save) then
        call logger%error("hist array not allocated or too small to store snapshots.")
        return
      end if
  
      if (size(snap_times) /= num_save) then
        call logger%error("snap_times is the wrong shape.")
      end if

      ! Save the initial condition as the first snapshot
      i_save = 1
      snapshots(:, i_save) = x
      snap_times(i_save) = 0.0d0
      i_save = i_save + 1
    end if

    ! M will contain the result of a banded addition between A and B
    M_ku = max(A_ku, B_ku)
    M_kl = max(A_kl, B_kl)
    M = new_banded_matrix(n, n, M_kl, M_ku)

    ! Trapezoidal (theta) method
    ! =============================================================================

    ! Compute M = B - dt * alpha * A
    ! 1. Copy B into M
    call banded_copy(M, B)

    ! 2. Scale A by (-dt*alpha) and add to M
    gamma = -dt * alpha
    call constant_times_banded_plus_banded(M, A, gamma)

    ! 3. Pre-factorise M once
    call get_LU_factorisation_banded(M, M_LU, M_ipiv)

    do i = 1, num_steps
      ! rhs = (B + beta * A)x = Bx + beta * Ax
      ! compute as 3 banded level 2 BLAS operations
      beta = (1 - alpha) * dt  ! scalar

      ! 1. compute rhs=Bx
      call zgbmv('N', B%m, B%n, B%kl, B%ku, (1.0_dp, 0.0_dp), B%AB, size(B%AB, dim = 1), x, 1, (0.0_dp, 0.0_dp), rhs, 1)

      ! 2. compute z=Ax
      call zgbmv('N', A%m, A%n, A%kl, A%ku, (1.0_dp, 0.0_dp), A%AB, size(A%AB, dim = 1), x, 1, (0.0_dp, 0.0_dp), z, 1)

      ! 3. compute rhs = rhs + beta*z
      call zaxpy(n, beta, z, 1, rhs, 1)

      ! Solve using pre-factorised LU (avoids re-factorising the constant M each step)
      x = solve_linear_system_complex_banded_LU(M, rhs, M_LU, M_ipiv)

      ! Save in history
      if (present(snapshots)) then
        ! Save every n-th snapshot
        if (mod(i, save_stride) == 0 .and. i < num_steps) then
          snapshots(:, i_save) = x
          snap_times(i_save) = i * dt
          i_save = i_save + 1
        end if
  
        ! Save at final step
        if (i == num_steps) then
          snapshots(:, i_save) = x
          snap_times(i_save) = i * dt
          i_save = i_save + 1
        end if
      end if
    end do

    deallocate(rhs)
    deallocate(z)
    deallocate(M_LU)
    deallocate(M_ipiv)

  end subroutine solve


  subroutine transform_matrix(mat)
    type(banded_matrix_t), intent(inout) :: mat
    integer                              :: col, row, rowInAB
  
    do col = 1, mat%n
      ! The band goes roughly from row=col - mat%ku to row=col + mat%kl
      do row = max(1, col - mat%ku), min(mat%m, col + mat%kl)
        rowInAB = mat%ku + 1 + row - col
        mat%AB(rowInAB, col) = -ic * mat%AB(rowInAB, col)
      end do
    end do
  end subroutine transform_matrix


end module mod_iv_solver