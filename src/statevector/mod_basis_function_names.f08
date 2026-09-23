module mod_basis_function_names
  use mod_global_variables, only: str_len_arr
  implicit none

  public

  character(len=str_len_arr), parameter :: QUADRATIC = "quadratic"
  character(len=str_len_arr), parameter :: DQUADRATIC = "dquadratic"
  character(len=str_len_arr), parameter :: CUBIC = "cubic"
  character(len=str_len_arr), parameter :: DCUBIC = "dcubic"
  character(len=str_len_arr), parameter :: DDCUBIC = "ddcubic"

end module mod_basis_function_names
