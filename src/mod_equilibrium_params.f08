! =============================================================================
!> Module containing all equilibrium parameters.
!! All parameters used in the equilibrium submodules are defined here
!! for convenience, including the wave numbers \(k_2\) and \(k_3\).
!! All of these values are NaN initially, such that variables that are
!! not properly set propagate their value and are easy to spot in follow-up checks.
!! @note 
!!     Variables that are not used remain equal to NaN throughout program execution.
!!     We define all of these in one module so they allow for easy setting through
!!     the parfile, and hence allow for more flexible control in the submodules.
!! @endnote
module mod_equilibrium_params
  use mod_global_variables, only: dp, str_len
  implicit none

  public

  !> wavenumber in y-direction (Cartesian) or theta-direction (cylindrical)
  real(dp)  :: k2
  !> wavenumber in z-direction
  real(dp)  :: k3

  !> general constant usually used for density
  real(dp)  :: cte_rho0
  !> general constant usually used for temperature
  real(dp)  :: cte_T0
  !> general constant usually used for the 01 magnetic field component
  real(dp)  :: cte_B01
  !> general constant usually used for the 02 magnetic field component
  real(dp)  :: cte_B02
  !> general constant usually used for the 03 magnetic field component
  real(dp)  :: cte_B03
  !> general constant usually used for the 02 velocity component
  real(dp)  :: cte_v02
  !> general constant usually used for the 03 velocity component
  real(dp)  :: cte_v03
  !> general constant usually used for pressure
  real(dp)  :: cte_p0
  !> general constant usually used in polynomials
  real(dp)  :: p1
  !> general constant usually used in polynomials
  real(dp)  :: p2
  !> general constant usually used in polynomials
  real(dp)  :: p3
  !> general constant usually used in polynomials
  real(dp)  :: p4
  !> general constant usually used in polynomials
  real(dp)  :: p5
  !> general constant usually used in polynomials
  real(dp)  :: p6
  !> general constant usually used in polynomials
  real(dp)  :: p7
  !> general constant usually used in polynomials
  real(dp)  :: p8
  !> general constant used in various expressions
  real(dp)  :: alpha
  !> general constant used in various expressions
  real(dp)  :: beta
  !> general constant used in various expressions
  real(dp)  :: delta
  !> general constant used in various expressions
  real(dp)  :: theta
  !> general constant used in various expressions
  real(dp)  :: tau
  !> general constant used in various expressions
  real(dp)  :: lambda
  !> general constant used in various expressions
  real(dp)  :: nu
  !> general constant usually used for radius
  real(dp)  :: r0
  !> general constant usually used for radius
  real(dp)  :: rc
  !> general constant usually used for radius
  real(dp)  :: rj
  !> general constant usually used in cylindrical geometry
  real(dp)  :: Bth0
  !> general constant usually used in cylindrical geometry
  real(dp)  :: Bz0
  !> general constant usually used in shear-related setups
  real(dp)  :: V
  !> general constant usually used for current
  real(dp)  :: j0
  !> general constant usually used for gravity
  real(dp)  :: g
  !> general boolean for varied use, defaults to False
  logical   :: eq_bool
  !> path to the file containing numerical equilibrium data
  character(len=str_len) :: input_file
  !> number of points to use for equidistant sampling of imported numerical data
  integer  :: n_input

contains


  !> Initialises all variables defined at module scope to NaN,
  !! including the wave numbers. This ensures that these have to
  !! be explicitly set.
  subroutine init_equilibrium_params()
    use mod_global_variables, only: NaN

    k2 = NaN
    k3 = NaN

    cte_rho0 = NaN
    cte_T0 = NaN
    cte_B01 = NaN
    cte_B02 = NaN
    cte_B03 = NaN
    cte_v02 = NaN
    cte_v03 = NaN
    cte_p0 = NaN

    p1 = NaN
    p2 = NaN
    p3 = NaN
    p4 = NaN
    p5 = NaN
    p6 = NaN
    p7 = NaN
    p8 = NaN

    alpha = NaN
    beta = NaN
    delta = NaN
    theta = NaN
    tau = NaN
    lambda = NaN
    nu = NaN

    r0 = NaN
    rc = NaN
    rj = NaN
    Bth0 = NaN
    Bz0 = NaN
    V = NaN
    j0 = NaN
    g = NaN

    eq_bool = .false.
    input_file = ''
    n_input = 1000
  end subroutine init_equilibrium_params

end module mod_equilibrium_params
