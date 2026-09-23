module mod_cooling_curve_names
  use mod_global_variables, only: str_len
  implicit none

  public

  character(len=str_len), parameter :: NOTHING = "nothing"
  character(len=str_len), parameter :: JC_CORONA = "JCcorona"
  character(len=str_len), parameter :: DALGARNO = "DM"
  character(len=str_len), parameter :: DALGARNO2 = "DM_2"
  character(len=str_len), parameter :: ML_SOLAR = "MLsolar1"
  character(len=str_len), parameter :: SPEX = "SPEX"
  character(len=str_len), parameter :: SPEX_DALGARNO = "SPEX_DM"
  character(len=str_len), parameter :: ROSNER = "Rosner"
  character(len=str_len), parameter :: COLGAN = "Colgan"
  character(len=str_len), parameter :: COLGAN_DM = "Colgan_DM"

  character(len=str_len), parameter :: KNOWN_CURVES(10) = [ &
    NOTHING, JC_CORONA, DALGARNO, DALGARNO2, ML_SOLAR, SPEX, &
    SPEX_DALGARNO, ROSNER, COLGAN, COLGAN_DM &
  ]
  ! these names match those in MPI-AMRVAC
end module mod_cooling_curve_names
