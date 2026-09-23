!> Contains a dedicated type for the various settings of the ARPACK solvers.
!! All variables that are used in different solver settings are defined, initialised
!! and set in this module.
module mod_arpack_type
  use mod_logging, only: logger, str
  use mod_global_variables, only: dp
  use mod_solver_settings, only: solver_settings_t
  implicit none

  !> General type containing the ARPACK configuration.
  type, public :: arpack_t
    !> mode of the solver
    integer, private :: mode
    !> reverse communication flag
    integer :: ido
    !> type of the matrix B ("I" = unit matrix, "G" = general matrix)
    character(len=1), private :: bmat
    !> dimension of the eigenvalue problem
    integer, private :: evpdim
    !> which eigenvalues to calculate
    character(len=2), private :: which
    !> number of eigenvalues to calculate
    integer, private :: nev
    !> stopping criteria, relative accuracy of Ritz eigenvalues
    real(dp), private :: tolerance
    !> residual vector
    complex(dp), allocatable :: residual(:)
    !> indicates how many Arnoldi vectors are generated each iteration
    integer, private :: ncv
    !> integer array containing mode and parameters
    integer :: iparam(11)
    !> length of workl array, must be at least 3 * ncv**2 + 5 * ncv
    integer, private :: lworkl
    !> info parameter
    integer :: info
    !> maximum number of iterations
    integer, private :: maxiter


    contains

    procedure, public :: get_bmat
    procedure, public :: get_evpdim
    procedure, public :: get_which
    procedure, public :: get_nev
    procedure, public :: get_tolerance
    procedure, public :: get_ncv
    procedure, public :: get_maxiter
    procedure, public :: get_lworkl
    procedure, public :: parse_znaupd_info
    procedure, public :: parse_zneupd_info
    procedure, public :: parse_finished_stats
    procedure, public :: destroy

    procedure, private :: set_mode
    procedure, private :: set_bmat
    procedure, private :: set_which
    procedure, private :: set_nev
    procedure, private :: set_residual
    procedure, private :: set_ncv
    procedure, private :: set_maxiter
  end type arpack_t

  private

  public :: new_arpack_config


contains

  !> Constructor for a new ARPACK configuration based on the dimension of the eigenvalue
  !! problem, mode of the solver and type of the B-matrix. Initialises required
  !! variables and allocates work arrays to be used when calling the solvers.
  function new_arpack_config(evpdim, mode, bmat, solver_settings) result(arpack_config)
    !> dimension of the eigenvalue problem
    integer, intent(in) :: evpdim
    !> mode for the solver
    integer, intent(in) :: mode
    !> type of the matrix B
    character(len=1), intent(in) :: bmat
    type(solver_settings_t), intent(inout) :: solver_settings
    !> initialised arpack configuration
    type(arpack_t) :: arpack_config

    call logger%debug("configuring Arnoldi parameters")
    arpack_config%evpdim = evpdim
    call arpack_config%set_mode(mode)

    arpack_config%ido = 0  ! 0 means first call to reverse communication interface
    call arpack_config%set_bmat(bmat)
    call arpack_config%set_which(solver_settings%which_eigenvalues)
    call arpack_config%set_nev(solver_settings%number_of_eigenvalues)
    arpack_config%tolerance = solver_settings%tolerance
    call arpack_config%set_residual(evpdim)
    ! if not user-set they are calculated and settings should be updated
    call arpack_config%set_ncv(solver_settings%ncv)
    call arpack_config%set_maxiter(solver_settings%maxiter)
    solver_settings%maxiter = arpack_config%get_maxiter()
    solver_settings%ncv = arpack_config%get_ncv()
    ! iparam(1) = ishift = 1 means restart with shifts from Hessenberg matrix
    arpack_config%iparam(1) = 1
  end function new_arpack_config


  !> Sets the mode for the solver.
  subroutine set_mode(this, mode)
    !> type instance
    class(arpack_t), intent(inout) :: this
    !> solver mode
    integer, intent(in) :: mode
    integer, parameter :: allowed_modes(3) = [1, 2, 3]

    if (.not. any(mode == allowed_modes)) then
      call logger%error( &
        "Arnoldi: mode = " // str(mode) // " is invalid, expected 1, 2 or 3" &
      )
      return
    end if
    this%mode = mode
    this%iparam(7) = this%mode
    call logger%debug("Arnoldi: mode set to " // str(this%mode))
  end subroutine set_mode


  !> Sets the type of B-matrix.
  subroutine set_bmat(this, bmat)
    !> type instance
    class(arpack_t), intent(inout) :: this
    !> type of B-matrix
    character, intent(in) :: bmat
    character :: allowed_bmats(2) = ["I", "G"]

    if (.not. any(bmat == allowed_bmats)) then
      call logger%error( &
        "Arnoldi: bmat = " // bmat // " is invalid, expected 'I' or 'G'" &
      )
      return
    end if
    this%bmat = bmat
    call logger%debug("Arnoldi: bmat set to " // this%bmat)
  end subroutine set_bmat


  !> Setter for the "which" argument of ARPACK routines.
  subroutine set_which(this, which)
    !> type instance
    class(arpack_t), intent(inout) :: this
    !> which kind of eigenvalues to calculate
    character(2), intent(in) :: which
    character(2) :: allowed_which(6) = ["LM", "SM", "LR", "SR", "LI", "SI"]

    if (.not. any(which == allowed_which)) then
      call logger%error( &
        "Arnoldi: which_eigenvalues = " // which &
        // " is invalid, expected one of " // str(allowed_which) &
      )
      return
    end if
    this%which = which
    call logger%debug("Arnoldi: which set to " // this%which)
  end subroutine set_which


  !> Setter for number of eigenvalues to calculate.
  subroutine set_nev(this, nev)
    !> type instance
    class(arpack_t), intent(inout) :: this
    !> number of eigenvalues to calculate
    integer, intent(in) :: nev

    if (nev <= 0) then
      call logger%error( &
        "Arnoldi: number of eigenvalues must be >= 0 but got " // str(nev) &
      )
      this%nev = 0
      return
    end if
    if (nev >= this%evpdim) then
      call logger%error( &
        "Arnoldi: number of eigenvalues (" // str(nev) &
        // ") >= " // "matrix size (" // str(this%evpdim) // ")" &
      )
      this%nev = 0
      return
    end if
    this%nev = nev
    call logger%debug("Arnoldi: nev set to " // str(this%nev))
  end subroutine set_nev


  !> Setter for the residual vector, allocates and manually initialises the
  !! residual (= starting) vector using a uniform distribution on (-1, 1)
  !! for both the real and imaginary parts. Relies on the LAPACK routine `zlarnv`.
  subroutine set_residual(this, evpdim)
    !> type instance
    class(arpack_t), intent(inout) :: this
    !> dimension of the eigenvalue problem
    integer, intent(in) :: evpdim
    !> which distribution to take, we set idist = 2 for uniform in (-1, 1)
    integer :: idist
    !> seed for the random number generator, last entry must be odd (see `zlarnv`)
    integer :: iseed(4)

    allocate(this%residual(evpdim))

    iseed = [2022, 9, 30, 179]
    idist = 2  ! real and imaginary parts each uniform (-1,1)
    call zlarnv(idist, iseed, evpdim, this%residual)
    ! tell arpack that we generated starting vector ourselves (info = 1)
    this%info = 1
  end subroutine set_residual


  !> Setter for ncv, the number of Arnoldi basis vectors to calculate.
  !! This should satisfy 1 <= ncv - nev and ncv <= evpdim, with recommended
  !! value ncv = 2 * nev (see arpack docs).
  subroutine set_ncv(this, ncv)
    !> type instance
    class(arpack_t), intent(inout) :: this
    !> value for ncv
    integer, intent(in) :: ncv

    if (ncv == 0) then
      this%ncv = max(this%nev+1, min(2 * this % nev, this % evpdim))
    else
      this%ncv = ncv
    end if
    if (1 > this%ncv - this%get_nev()) then
      call logger%error( &
        "ncv too low, expected ncv - nev >= 1 but got ncv - nev = " &
        // str(this%ncv - this%get_nev()) &
      )
      return
    end if
    if (this%ncv > this%get_evpdim()) then
      call logger%error( &
        "ncv too high, expected ncv < N but got ncv = " // str(this%ncv) &
        // " and N = " // str(this%get_evpdim()) &
      )
      return
    end if
    call logger%debug("Arnoldi: ncv set to " // str(this%ncv))
  end subroutine set_ncv


  !> Sets the maximum number of iterations that ARPACK is allowed to take, defaults
  !! to max(100, 10 * k) with k the number of eigenvalues.
  !! @warning
  !!     Throws a warning if <tt>maxiter</tt> is smaller than 10*N.
  !! @endwarning
  subroutine set_maxiter(this, maxiter)
    !> type instance
    class(arpack_t), intent(inout) :: this
    !> maximum number of iterations
    integer, intent(in) :: maxiter
    integer :: min_maxiter

    min_maxiter = max(100, 10 * this%get_nev())
    if (maxiter < 0) then
      call logger%error( &
        "Arnoldi: maxiter must be positive, but is equal to " // str(maxiter) &
      )
      return
    end if
    if (maxiter == 0) then
      this%maxiter = min_maxiter
    else
      this%maxiter = maxiter
    end if
    if (this%maxiter < min_maxiter) then ! LCOV_EXCL_START
      call logger%warning( &
        "Arnoldi: maxiter (" // str(maxiter) // ") below recommended max(100, 10*k) (" &
        // str(min_maxiter) // ")" &
      )
    end if ! LCOV_EXCL_STOP
    this%iparam(3) = this%maxiter
  end subroutine set_maxiter


  !> Getter for kind of B-matrix in eigenvalue problem.
  pure character function get_bmat(this)
    !> type instance
    class(arpack_t), intent(in) :: this

    get_bmat = this%bmat
  end function get_bmat


  !> Getter for dimension of eigenvalue problem.
  pure integer function get_evpdim(this)
    !> type instance
    class(arpack_t), intent(in) :: this

    get_evpdim = this%evpdim
  end function get_evpdim


  !> Getter for which eigenvalues to return.
  pure character(2) function get_which(this)
    !> type instance
    class(arpack_t), intent(in) :: this

    get_which = this%which
  end function get_which


  !> Getter for number of eigenvalues to calculate.
  pure integer function get_nev(this)
    !> type instance
    class(arpack_t), intent(in) :: this

    get_nev = this%nev
  end function get_nev


  !> Getter for tolerance (relative accuracy) to indicate eigenvalue convergence.
  pure real(dp) function get_tolerance(this)
    !> type instance
    class(arpack_t), intent(in) :: this

    get_tolerance = this%tolerance
  end function get_tolerance


  !> Getter for number of Arnoldi basis vectors that should be calculated.
  pure integer function get_ncv(this)
    !> type instance
    class(arpack_t), intent(in) :: this

    get_ncv = this%ncv
  end function get_ncv


  !> Getter for maximum number of iterations.
  pure integer function get_maxiter(this)
    !> type instance
    class(arpack_t), intent(in) :: this
    get_maxiter = this%maxiter
  end function get_maxiter


  !> Getter for length of workl array, returns 3 * ncv**2 + 5 * ncv
  pure integer function get_lworkl(this)
    !> type instance
    class(arpack_t), intent(in) :: this
    integer :: ncv

    ncv = this%get_ncv()
    get_lworkl = 3 * ncv**2 + 5 * ncv
  end function get_lworkl


  !> Destructor, deallocates variables.
  pure subroutine destroy(this)
    !> type instance
    class(arpack_t), intent(inout) :: this

    if (allocated(this%residual)) deallocate(this%residual)
  end subroutine destroy


  !> Parses the info parameter that comes out of ARPACK's <tt>znaupd</tt> method.
  !! If info = 0, everything behaved nicely and the reverse communication subroutines
  !! exited properly. If info is any other value something went wrong and
  !! we handle it accordingly.
  subroutine parse_znaupd_info(this, converged)
    !> reference to type object
    class(arpack_t), intent(in) :: this
    !> if .true. the reverse communication routines converged, .false. otherwise
    logical, intent(out)  :: converged

    call logger%debug("checking znaupd info parameter")
    converged = .false.
    select case(this % info)
    case(0)
      converged = .true.
    case(1) ! LCOV_EXCL_START
      call logger%warning("ARPACK failed to converge! (maxiter reached)")
      call logger%warning("number of iterations: " // str(this % maxiter))
      call logger%warning( &
        "number of converged eigenvalues: " // str(this % iparam(5)) // &
        " / " // str(this % nev) &
      )
    case(3)
      call logger%error( &
        "znaupd: no shifts could be applied during a cycle of the Arnoldi iteration." &
        // " Try increasing the size of ncv relative to number_of_eigenvalues." &
      )
      return
    case(-6)
      call logger%error("znaupd: bmat must be 'I' or 'G'")
    case(-8)
      call logger%error("znaupd: error LAPACK eigenvalue calculation")
      return
    case(-9)
      call logger%error("znaupd: starting vector is zero, try rerunning?")
      return
    case(-11)
      call logger%error("mode = 1 and bmat = 'G' are incompatible")
      return
    case(-9999)
      call logger%error("ARPACK could not build, something went wrong")
      return
    case default
      call logger%error( &
        "znaupd: unexpected info = " // str(this % info) // " encountered" &
      )
      return ! LCOV_EXCL_STOP
    end select
  end subroutine parse_znaupd_info


  !> Parses the info parameter that comes out of ARPACK's <tt>zneupd</tt> method.
  !! If info = 0, the eigenvalues extraction routines exited properly, if info
  !! is any other value something went wrong and we handle it accordingly.
  subroutine parse_zneupd_info(this)
    !> reference to type object
    class(arpack_t), intent(in)  :: this

    call logger%debug("checking zneupd info parameter")

    select case(this % info)
    case(0)
      return
    case(-8) ! LCOV_EXCL_START
      call logger%error("zneupd: error from LAPACK eigenvalue calculation")
      return
    case(-9)
      call logger%error("zneupd: error from LAPACK eigenvector calculation (ztrevc)")
      return
    case(-14)
      call logger%error("zneupd: no eigenvalues with sufficient accuracy found")
      return
    case(-15)
      call logger%error("zneupd: different count for converged eigenvalues than znaupd")
      return
    case default
      call logger%error("zneupd: unexpected info = " // str(this % info) // " value")
      return ! LCOV_EXCL_STOP
    end select
  end subroutine parse_zneupd_info


  !> Parses the statistics that come out of ARPACK when the run is finished. Displays
  !! the number of OP*X and B*X operations and the number of re-orthogonalisation
  !! steps that were needed.
  subroutine parse_finished_stats(this)
    !> type instance
    class(arpack_t), intent(in) :: this

    call logger%info("Arnoldi iteration finished. Statistics: ")
    call logger%disable_prefix()
    call logger%info("  Total number of OP*x operations: " // str(this%iparam(9)))
    call logger%info("  Total number of B*x operations: " // str(this%iparam(10)))
    call logger%info( &
      "  Total number of re-orthogonalisation steps: " // str(this%iparam(11)) &
    )
    call logger%enable_prefix()
  end subroutine parse_finished_stats
end module mod_arpack_type
