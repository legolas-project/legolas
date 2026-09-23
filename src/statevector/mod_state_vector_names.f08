module mod_state_vector_names
  use mod_global_variables, only: str_len_arr
  implicit none

  public

  character(len=str_len_arr), parameter, public :: sv_rho1_name = "rho"
  character(len=str_len_arr), parameter, public :: sv_v1_name = "v1"
  character(len=str_len_arr), parameter, public :: sv_v2_name = "v2"
  character(len=str_len_arr), parameter, public :: sv_v3_name = "v3"
  character(len=str_len_arr), parameter, public :: sv_T1_name = "T"
  character(len=str_len_arr), parameter, public :: sv_a1_name = "a1"
  character(len=str_len_arr), parameter, public :: sv_a2_name = "a2"
  character(len=str_len_arr), parameter, public :: sv_a3_name = "a3"

  character(len=str_len_arr), parameter, public :: KNOWN_SV_NAMES(8) = [ &
    sv_rho1_name, &
    sv_v1_name, &
    sv_v2_name, &
    sv_v3_name, &
    sv_T1_name, &
    sv_a1_name, &
    sv_a2_name, &
    sv_a3_name &
  ]

end module mod_state_vector_names
