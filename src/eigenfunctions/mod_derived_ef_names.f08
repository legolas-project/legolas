module mod_derived_ef_names
  use mod_global_variables, only: str_len_arr
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_get_indices, only: get_index
  implicit none

  private

  character(len=str_len_arr), parameter, public :: S_name = "S"
  character(len=str_len_arr), parameter, public :: div_v_name = "div v"
  character(len=str_len_arr), parameter, public :: curl_v_1_name = "(curl v)1"
  character(len=str_len_arr), parameter, public :: curl_v_2_name = "(curl v)2"
  character(len=str_len_arr), parameter, public :: curl_v_3_name = "(curl v)3"
  character(len=str_len_arr), parameter, public :: B1_name = "B1"
  character(len=str_len_arr), parameter, public :: B2_name = "B2"
  character(len=str_len_arr), parameter, public :: B3_name = "B3"
  character(len=str_len_arr), parameter, public :: div_B_name = "div B"
  character(len=str_len_arr), parameter, public :: curl_B_1_name = "(curl B)1"
  character(len=str_len_arr), parameter, public :: curl_B_2_name = "(curl B)2"
  character(len=str_len_arr), parameter, public :: curl_B_3_name = "(curl B)3"
  character(len=str_len_arr), parameter, public :: B_para_name = "B_para"
  character(len=str_len_arr), parameter, public :: B_perp_name = "B_perp"
  character(len=str_len_arr), parameter, public :: curl_B_para_name = "(curl B)_para"
  character(len=str_len_arr), parameter, public :: curl_B_perp_name = "(curl B)_perp"
  character(len=str_len_arr), parameter, public :: v_para_name = "v_para"
  character(len=str_len_arr), parameter, public :: v_perp_name = "v_perp"
  character(len=str_len_arr), parameter, public :: curl_v_para_name = "(curl v)_para"
  character(len=str_len_arr), parameter, public :: curl_v_perp_name = "(curl v)_perp"
  character(len=str_len_arr), parameter, public :: dB1_name = "dB1"

  character(len=:), allocatable :: state_vector(:)
  logical :: can_get_pp

  public :: create_and_set_derived_state_vector

contains

  function create_and_set_derived_state_vector(settings, background) &
    result(derived_state_vector)
    type(settings_t), intent(inout) :: settings
    type(background_t), intent(in) :: background
    character(len=str_len_arr), allocatable :: derived_state_vector(:)
    logical, allocatable :: derived_state_vector_mask(:)

    state_vector = settings%get_state_vector()
    can_get_pp = can_calculate_pp_quantities(background)
    derived_state_vector = [ &
      S_name, &
      div_v_name, &
      curl_v_1_name, &
      curl_v_2_name, &
      curl_v_3_name, &
      B1_name, &
      B2_name, &
      B3_name, &
      div_B_name, &
      curl_B_1_name, &
      curl_B_2_name, &
      curl_B_3_name, &
      B_para_name, &
      B_perp_name, &
      curl_B_para_name, &
      curl_B_perp_name, &
      v_para_name, &
      v_perp_name, &
      curl_v_para_name, &
      curl_v_perp_name, &
      dB1_name &
    ]
    derived_state_vector_mask = [ &
      can_get_entropy(), &
      can_get_div_v(), &
      can_get_curl_v_1(), &
      can_get_curl_v_2(), &
      can_get_curl_v_3(), &
      can_get_B1(), &
      can_get_B2(), &
      can_get_B3(), &
      can_get_div_B(), &
      can_get_curl_B_i(), &
      can_get_curl_B_i(), &
      can_get_curl_B_i(), &
      can_get_B_pp(), &
      can_get_B_pp(), &
      can_get_curl_B_pp(), &
      can_get_curl_B_pp(), &
      can_get_v_pp(), &
      can_get_v_pp(), &
      can_get_curl_v_pp(), &
      can_get_curl_v_pp(), &
      can_get_dB1() &
    ]
    derived_state_vector = pack(derived_state_vector, mask=derived_state_vector_mask)

    call settings%set_derived_state_vector(derived_state_vector)

    if (allocated(state_vector)) deallocate(state_vector)
    if (allocated(derived_state_vector_mask)) deallocate(derived_state_vector_mask)
  end function create_and_set_derived_state_vector


  pure logical function is_in_state_vector(name)
    character(len=*), intent(in) :: name
    is_in_state_vector = get_index(name, state_vector) > 0
  end function is_in_state_vector


  pure logical function can_get_entropy()
    can_get_entropy = is_in_state_vector("rho") .and. is_in_state_vector("T")
  end function can_get_entropy


  pure logical function can_get_div_v()
    can_get_div_v = ( &
      is_in_state_vector("v1") &
      .and. is_in_state_vector("v2") &
      .and. is_in_state_vector("v3") &
    )
  end function can_get_div_v


  pure logical function can_get_curl_v_1()
    can_get_curl_v_1 = is_in_state_vector("v2") .and. is_in_state_vector("v3")
  end function can_get_curl_v_1


  pure logical function can_get_curl_v_2()
    can_get_curl_v_2 = is_in_state_vector("v1") .and. is_in_state_vector("v3")
  end function can_get_curl_v_2


  pure logical function can_get_curl_v_3()
    can_get_curl_v_3 = is_in_state_vector("v1") .and. is_in_state_vector("v2")
  end function can_get_curl_v_3


  pure logical function can_get_B1()
    can_get_B1 = is_in_state_vector("a3") .and. is_in_state_vector("a2")
  end function can_get_B1


  pure logical function can_get_B2()
    can_get_B2 = is_in_state_vector("a1") .and. is_in_state_vector("a3")
  end function can_get_B2


  pure logical function can_get_B3()
    can_get_B3 = is_in_state_vector("a2") .and. is_in_state_vector("a1")
  end function can_get_B3


  pure logical function can_get_div_B()
    can_get_div_B = ( &
      is_in_state_vector("a1") &
      .and. is_in_state_vector("a2") &
      .and. is_in_state_vector("a3") &
    )
  end function can_get_div_B


  pure logical function can_get_curl_B_i()
    can_get_curl_B_i = ( &
      is_in_state_vector("a1") &
      .and. is_in_state_vector("a2") &
      .and. is_in_state_vector("a3") &
    )
  end function can_get_curl_B_i


  pure logical function can_get_B_pp()
    can_get_B_pp = can_get_pp .and. can_get_B2() .and. can_get_B3()
  end function can_get_B_pp


  pure logical function can_get_curl_B_pp()
    can_get_curl_B_pp = can_get_pp .and. can_get_curl_B_i()
  end function can_get_curl_B_pp


  pure logical function can_get_v_pp()
    can_get_v_pp = ( &
      can_get_pp .and. is_in_state_vector("v2") .and. is_in_state_vector("v3") &
    )
  end function can_get_v_pp


  pure logical function can_get_curl_v_pp()
    can_get_curl_v_pp = ( &
      can_get_pp .and. can_get_curl_v_2() .and. can_get_curl_v_3() &
    )
  end function can_get_curl_v_pp


  pure logical function can_get_dB1()
    can_get_dB1 = ( &
      is_in_state_vector("a2") &
      .and. is_in_state_vector("a3") &
    )
  end function can_get_dB1


  logical function can_calculate_pp_quantities(background)
    use mod_function_utils, only: zero_func
    use mod_logging, only: logger

    type(background_t), intent(in) :: background

    can_calculate_pp_quantities = .false.
    if (.not. associated(background%magnetic%B01, zero_func)) then
      call logger%warning( &
        "parallel/perpendicular derived quantities currently not supported &
        &for non-zero B01 components" &
      )
      return
    end if
    if (.not. associated(background%velocity%v01, zero_func)) then
      call logger%warning( &
        "parallel/perpendicular derived quantities currently not supported &
        &for non-zero v01 components" &
      )
      return
    end if
    can_calculate_pp_quantities = ( &
      .not. associated(background%magnetic%B02, zero_func) &
      .or. .not. associated(background%magnetic%B03, zero_func) &
    )
  end function can_calculate_pp_quantities

end module mod_derived_ef_names
