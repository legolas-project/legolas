! =============================================================================
!> Parent module for everything solver-related. Interfaces to the different
!! submodules are defined here, and the <tt>solve_evp</tt> routine calls the
!! correct solver based on parfile settings.
module mod_solvers
  use mod_global_variables, only: dp, NaN
  use mod_logging, only: logger, str
  use mod_check_values, only: set_small_values_to_zero
  use mod_matrix_structure, only: matrix_t
  use mod_transform_matrix, only: matrix_to_array
  use mod_settings, only: settings_t
  implicit none

  private

  !> interface to the different solution methods implemented in submodules
  interface
    module subroutine qr_invert(matrix_A, matrix_B, settings, omega, vr)
      !> matrix A
      type(matrix_t), intent(in) :: matrix_A
      !> matrix B
      type(matrix_t), intent(in) :: matrix_B
      !> settings object
      type(settings_t), intent(in) :: settings
      !> array with eigenvalues
      complex(dp), intent(out)  :: omega(:)
      !> array with right eigenvectors
      complex(dp), intent(out)  :: vr(:, :)
    end subroutine

    module subroutine qr_cholesky(matrix_A, matrix_B, settings, omega, vr)
      !> matrix A
      type(matrix_t), intent(in) :: matrix_A
      !> matrix B
      type(matrix_t), intent(in) :: matrix_B
      !> settings object
      type(settings_t), intent(in) :: settings
      !> array with eigenvalues
      complex(dp), intent(out)  :: omega(:)
      !> array with right eigenvectors
      complex(dp), intent(out)  :: vr(:, :)
    end subroutine qr_cholesky

    module subroutine qz_direct(matrix_A, matrix_B, settings, omega, vr)
      !> matrix A
      type(matrix_t), intent(in) :: matrix_A
      !> matrix B
      type(matrix_t), intent(in) :: matrix_B
      !> settings object
      type(settings_t), intent(in) :: settings
      !> array with eigenvalues
      complex(dp), intent(out)  :: omega(:)
      !> array with right eigenvectors
      complex(dp), intent(out)  :: vr(:, :)
    end subroutine qz_direct

    module subroutine arnoldi(matrix_A, matrix_B, settings, omega, vr)
      !> matrix A
      type(matrix_t), intent(in) :: matrix_A
      !> matrix B
      type(matrix_t), intent(in) :: matrix_B
      !> settings object
      type(settings_t), intent(inout) :: settings
      !> array with eigenvalues
      complex(dp), intent(out)  :: omega(:)
      !> array with right eigenvectors
      complex(dp), intent(out)  :: vr(:, :)
    end subroutine arnoldi

    module subroutine inverse_iteration(matrix_A, matrix_B, settings, omega, vr)
      !> matrix A
      type(matrix_t), intent(in) :: matrix_A
      !> matrix B
      type(matrix_t), intent(in) :: matrix_B
      !> settings object
      type(settings_t), intent(in) :: settings
      !> array with eigenvalues
      complex(dp), intent(out)  :: omega(:)
      !> array with right eigenvectors
      complex(dp), intent(out)  :: vr(:, :)
    end subroutine inverse_iteration
  end interface

  public  :: solve_evp

contains


  !> Main subroutine to solve the eigenvalue problem. Depending on the solvelist
  !! passed in the parfile, different solvers are called.
  !! @warning
  !!     Throws an error if an unknown solver is passed.
  !! @endwarning
  subroutine solve_evp(matrix_A, matrix_B, settings, omega, vr)
    !> A-matrix
    type(matrix_t), intent(in) :: matrix_A
    !> B-matrix
    type(matrix_t), intent(in) :: matrix_B
    !> settings object
    type(settings_t), intent(inout) :: settings
    !> eigenvalues
    complex(dp), intent(out)  :: omega(:)
    !> right eigenvectors
    complex(dp), intent(out)  :: vr(:, :)

    select case(settings%solvers%get_solver())
    case("QR-invert")
      call qr_invert(matrix_A, matrix_B, settings, omega, vr)
    case("QR-cholesky")
      call qr_cholesky(matrix_A, matrix_B, settings, omega, vr)
    case("QZ-direct")
      call qz_direct(matrix_A, matrix_B, settings, omega, vr)
    case("arnoldi")
      call arnoldi(matrix_A, matrix_B, settings, omega, vr)
    case("inverse-iteration")
      call inverse_iteration(matrix_A, matrix_B, settings, omega, vr)
    case("none")
      ! Set eigenvalues and vectors to NaN.
      omega = NaN * (1, 1)
      if (settings%io%should_compute_eigenvectors()) vr = NaN * (1, 1)
    case default
      call logger%error("unknown solver passed: " // settings%solvers%get_solver())
      return
    end select
  end subroutine solve_evp

end module mod_solvers
