module mod_global_variables
  use, intrinsic :: iso_fortran_env
  implicit none

  public

  !> single-precision value
  integer, parameter :: sp = real32
  !> double-precision value
  integer, parameter :: dp = real64
  !> quadruple-precision value
  integer, parameter :: qp = real128
  !> default length for strings
  integer, parameter :: str_len = 500
  !> default length for strings in arrays
  integer, parameter :: str_len_arr = 16

  !> tolerance for real(dp) zero, bit higher than machine precision
  real(dp), parameter :: dp_LIMIT = 5.0d-16
  !> NaN value (ieee_quiet_nan)
  real(dp), protected :: NaN

  !> complex number i
  complex(dp), parameter    :: ic = (0.0d0, 1.0d0)
  !> complex real
  complex(dp), parameter    :: ir = (1.0d0, 0.0d0)

  !> number of Gaussian nodes
  integer, parameter           :: n_gauss = 4
  !> values for the Gaussian nodes in [-1, 1]
  real(dp), parameter :: gaussian_nodes(n_gauss) = [ &
    -0.861136311594053_dp, &
    -0.339981043584856_dp, &
    0.339981043584856_dp, &
    0.861136311594053_dp &
  ]
  !> weights for the Gaussian nodes in [-1, 1]
  real(dp), parameter :: gaussian_weights(n_gauss) = [ &
    0.347854845137454_dp, &
    0.652145154862546_dp, &
    0.652145154862546_dp, &
    0.347854845137454_dp &
  ]

contains


  subroutine initialise_globals()
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan

    NaN = ieee_value(NaN, ieee_quiet_nan)
  end subroutine initialise_globals

end module mod_global_variables
