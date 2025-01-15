module mod_derived_efs
  use mod_global_variables, only: dp, str_len_arr, ic
  use mod_equilibrium_params, only: k2, k3
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_grid, only: grid_t
  use mod_ef_assembly, only: assemble_eigenfunction, retransform_eigenfunction
  use mod_derived_ef_names
  use mod_logging, only: logger
  use mod_get_indices, only: get_index
  use mod_state_vector_component, only: sv_component_t
  use mod_state_vector, only: sv_rho1, sv_v1, sv_v2, sv_v3, sv_T1, sv_a1, sv_a2, sv_a3
  implicit none

  private

  interface
    function derived_ef_func(settings, grid, eigenvector) result(ef)
      use mod_global_variables, only: dp
      use mod_settings, only: settings_t
      use mod_grid, only: grid_t
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      complex(dp), intent(in) :: eigenvector(:)
      complex(dp) :: ef(size(grid%ef_grid))
    end function derived_ef_func
  end interface

  type, public :: derived_ef_t
    character(str_len_arr) :: name
    complex(dp), allocatable :: quantities(:, :)
    procedure(derived_ef_func), pointer, private, nopass :: get_derived_ef

  contains

    procedure, public :: initialise
    procedure, public :: assemble
    procedure, public :: delete

    procedure, private :: set_function_pointer

  end type derived_ef_t

  real(dp), allocatable :: rho0_on_ef_grid(:)
  real(dp), allocatable :: T0_on_ef_grid(:)
  real(dp), allocatable :: B02_on_ef_grid(:)
  real(dp), allocatable :: B03_on_ef_grid(:)

  public :: deallocate_derived_ef_module_variables

contains

  subroutine initialise(this, name, ef_grid_size, nb_efs)
    class(derived_ef_t), intent(inout) :: this
    character(str_len_arr), intent(in) :: name
    integer, intent(in) :: ef_grid_size
    integer, intent(in) :: nb_efs

    this%name = name
    allocate(this%quantities(ef_grid_size, nb_efs))
    call this%set_function_pointer()
  end subroutine initialise


  subroutine assemble( &
    this, settings, grid, background, idxs_to_assemble, right_eigenvectors &
  )
    class(derived_ef_t), intent(inout) :: this
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    integer, intent(in) :: idxs_to_assemble(:)
    complex(dp), intent(in) :: right_eigenvectors(:, :)
    integer :: i, idx

    call set_equilibrium_arrays_on_ef_grid(background, grid)

    do i = 1, size(idxs_to_assemble)
      idx = idxs_to_assemble(i)
      this%quantities(:, i) = this%get_derived_ef( &
        settings=settings, &
        grid=grid, &
        eigenvector=right_eigenvectors(:, idx) &
      )
    end do
  end subroutine assemble


  pure subroutine delete(this)
    class(derived_ef_t), intent(inout) :: this
    if (allocated(this%quantities)) deallocate(this%quantities)
    nullify(this%get_derived_ef)
  end subroutine delete


  subroutine set_function_pointer(this)
    class(derived_ef_t), intent(inout) :: this

    select case(this%name)
    case(S_name)
      this%get_derived_ef => get_entropy
    case(div_v_name)
      this%get_derived_ef => get_div_v
    case(curl_v_1_name)
      this%get_derived_ef => get_curl_v_1
    case(curl_v_2_name)
      this%get_derived_ef => get_curl_v_2
    case(curl_v_3_name)
      this%get_derived_ef => get_curl_v_3
    case(B1_name)
      this%get_derived_ef => get_B1
    case(B2_name)
      this%get_derived_ef => get_B2
    case(B3_name)
      this%get_derived_ef => get_B3
    case(div_B_name)
      this%get_derived_ef => get_div_B
    case(curl_B_1_name)
      this%get_derived_ef => get_curl_B_1
    case(curl_B_2_name)
      this%get_derived_ef => get_curl_B_2
    case(curl_B_3_name)
      this%get_derived_ef => get_curl_B_3
    case(B_para_name)
      this%get_derived_ef => get_B_para
    case(B_perp_name)
      this%get_derived_ef => get_B_perp
    case(curl_B_para_name)
      this%get_derived_ef => get_curl_B_para
    case(curl_B_perp_name)
      this%get_derived_ef => get_curl_B_perp
    case(v_para_name)
      this%get_derived_ef => get_v_para
    case(v_perp_name)
      this%get_derived_ef => get_v_perp
    case(curl_v_para_name)
      this%get_derived_ef => get_curl_v_para
    case(curl_v_perp_name)
      this%get_derived_ef => get_curl_v_perp
    case(dB1_name)
      this%get_derived_ef => get_dB1
    case default
      call logger%error( &
        "derived ef assembly -- unknown eigenfunction name: "// trim(this%name) &
      )
      nullify(this%get_derived_ef)
      return
    end select
  end subroutine set_function_pointer


  function get_entropy(settings, grid, eigenvector) result(entropy)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: entropy(size(grid%ef_grid))
    complex(dp) :: rho(size(grid%ef_grid)), T(size(grid%ef_grid))

    rho = get_base_eigenfunction(sv_rho1, settings, grid, eigenvector)
    T = get_base_eigenfunction(sv_T1, settings, grid, eigenvector)
    entropy = ( &
      T / rho0_on_ef_grid ** (2.0_dp / 3.0_dp) &
      - (2.0_dp / 3.0_dp) * ( &
        rho * T0_on_ef_grid / rho0_on_ef_grid ** (5.0_dp / 3.0_dp) &
      ) &
    )
  end function get_entropy


  function get_div_v(settings, grid, eigenvector) result(div_v)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: div_v(size(grid%ef_grid))
    complex(dp) :: dv1(size(grid%ef_grid))
    complex(dp) :: v2(size(grid%ef_grid))
    complex(dp) :: v3(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid))

    dv1 = get_base_eigenfunction(sv_v1, settings, grid, eigenvector, diff_order=1)
    v2 = get_base_eigenfunction(sv_v2, settings, grid, eigenvector)
    v3 = get_base_eigenfunction(sv_v3, settings, grid, eigenvector)
    ef_eps = grid%get_eps(grid%ef_grid)
    div_v = ic * (-dv1 / ef_eps + k2 * v2 / ef_eps + k3 * v3)
  end function get_div_v


  function get_curl_v_1(settings, grid, eigenvector) result(curl_v_1)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: curl_v_1(size(grid%ef_grid))
    complex(dp) :: v2(size(grid%ef_grid))
    complex(dp) :: v3(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid))

    v2 = get_base_eigenfunction(sv_v2, settings, grid, eigenvector)
    v3 = get_base_eigenfunction(sv_v3, settings, grid, eigenvector)
    ef_eps = grid%get_eps(grid%ef_grid)
    curl_v_1 = ic * (k2 * v3 / ef_eps - k3 * v2)
  end function get_curl_v_1


  function get_curl_v_2(settings, grid, eigenvector) result(curl_v_2)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: curl_v_2(size(grid%ef_grid))
    complex(dp) :: v1(size(grid%ef_grid))
    complex(dp) :: v3(size(grid%ef_grid))
    complex(dp) :: dv3(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid)), ef_deps

    v1 = get_base_eigenfunction(sv_v1, settings, grid, eigenvector)
    v3 = get_base_eigenfunction(sv_v3, settings, grid, eigenvector)
    dv3 = get_base_eigenfunction(sv_v3, settings, grid, eigenvector, diff_order=1)
    ef_eps = grid%get_eps(grid%ef_grid)
    ef_deps = grid%get_deps()
    curl_v_2 = ic * k3 * v1 - dv3 / ef_eps - ef_deps * v3 / ef_eps
  end function get_curl_v_2


  function get_curl_v_3(settings, grid, eigenvector) result(curl_v_3)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: curl_v_3(size(grid%ef_grid))
    complex(dp) :: v1(size(grid%ef_grid))
    complex(dp) :: v2(size(grid%ef_grid))
    complex(dp) :: dv2(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid)), ef_deps

    v1 = get_base_eigenfunction(sv_v1, settings, grid, eigenvector)
    v2 = get_base_eigenfunction(sv_v2, settings, grid, eigenvector)
    dv2 = get_base_eigenfunction(sv_v2, settings, grid, eigenvector, diff_order=1)
    ef_eps = grid%get_eps(grid%ef_grid)
    ef_deps = grid%get_deps()
    curl_v_3 = dv2 + (ef_deps * v2 - ic * k2 * v1) / ef_eps
  end function get_curl_v_3


  function get_B1(settings, grid, eigenvector) result(B1)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: B1(size(grid%ef_grid))
    complex(dp) :: a2(size(grid%ef_grid))
    complex(dp) :: a3(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid))

    a2 = get_base_eigenfunction(sv_a2, settings, grid, eigenvector)
    a3 = get_base_eigenfunction(sv_a3, settings, grid, eigenvector)
    ef_eps = grid%get_eps(grid%ef_grid)
    B1 = ic * (k2 * a3 / ef_eps - k3 * a2)
  end function get_B1


  function get_B2(settings, grid, eigenvector) result(B2)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: B2(size(grid%ef_grid))
    complex(dp) :: a1(size(grid%ef_grid))
    complex(dp) :: da3(size(grid%ef_grid))

    a1 = get_base_eigenfunction(sv_a1, settings, grid, eigenvector)
    da3 = get_base_eigenfunction(sv_a3, settings, grid, eigenvector, diff_order=1)
    B2 = ic * k3 * a1 - da3
  end function get_B2


  function get_B3(settings, grid, eigenvector) result(B3)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: B3(size(grid%ef_grid))
    complex(dp) :: a1(size(grid%ef_grid))
    complex(dp) :: da2(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid))

    a1 = get_base_eigenfunction(sv_a1, settings, grid, eigenvector)
    da2 = get_base_eigenfunction(sv_a2, settings, grid, eigenvector, diff_order=1)
    ef_eps = grid%get_eps(grid%ef_grid)
    B3 = (da2 - ic * k2 * a1) / ef_eps
  end function get_B3


  function get_div_B(settings, grid, eigenvector) result(div_B)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: div_B(size(grid%ef_grid))
    complex(dp) :: a1(size(grid%ef_grid))
    complex(dp) :: da2(size(grid%ef_grid))
    complex(dp) :: da3(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid))

    a1 = get_base_eigenfunction(sv_a1, settings, grid, eigenvector)
    da2 = get_base_eigenfunction(sv_a2, settings, grid, eigenvector, diff_order=1)
    da3 = get_base_eigenfunction(sv_a3, settings, grid, eigenvector, diff_order=1)
    ef_eps = grid%get_eps(grid%ef_grid)
    div_B = ( &
      ic * (k2 * da3 - k3 * da2) &
      + ic * k2 * (ic * k3 * a1 - da3) &
      + ic * k3 * (da2 - ic * k2 * a1) &
    ) / ef_eps
  end function get_div_B


  function get_curl_B_1(settings, grid, eigenvector) result(curl_B_1)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: curl_B_1(size(grid%ef_grid))
    complex(dp) :: a1(size(grid%ef_grid))
    complex(dp) :: da2(size(grid%ef_grid))
    complex(dp) :: da3(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid))

    a1 = get_base_eigenfunction(sv_a1, settings, grid, eigenvector)
    da2 = get_base_eigenfunction(sv_a2, settings, grid, eigenvector, diff_order=1)
    da3 = get_base_eigenfunction(sv_a3, settings, grid, eigenvector, diff_order=1)
    ef_eps = grid%get_eps(grid%ef_grid)
    curl_B_1 = ( &
      ic * k2 * (da2 - ic * k2 * a1) / ef_eps**2 - ic * k3 * (ic * k3 * a1 - da3) &
    )
  end function get_curl_B_1


  function get_curl_B_2(settings, grid, eigenvector) result(curl_B_2)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: curl_B_2(size(grid%ef_grid))
    complex(dp) :: a1(size(grid%ef_grid)), da1(size(grid%ef_grid))
    complex(dp) :: a2(size(grid%ef_grid)), da2(size(grid%ef_grid))
    complex(dp) :: dda2(size(grid%ef_grid))
    complex(dp) :: a3(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid)), ef_deps

    a1 = get_base_eigenfunction(sv_a1, settings, grid, eigenvector)
    da1 = get_base_eigenfunction(sv_a1, settings, grid, eigenvector, diff_order=1)
    a2 = get_base_eigenfunction(sv_a2, settings, grid, eigenvector)
    da2 = get_base_eigenfunction(sv_a2, settings, grid, eigenvector, diff_order=1)
    dda2 = get_base_eigenfunction(sv_a2, settings, grid, eigenvector, diff_order=2)
    a3 = get_base_eigenfunction(sv_a3, settings, grid, eigenvector)
    ef_eps = grid%get_eps(grid%ef_grid)
    ef_deps = grid%get_deps()
    curl_B_2 = ( &
      -k3 * (k2 * a3 / ef_eps - k3 * a2) &
      - (dda2 - k2 * da1) / ef_eps &
      + ef_deps * (da2 - ic * k2 * a1) / ef_eps ** 2 &
    )
  end function get_curl_B_2


  function get_curl_B_3(settings, grid, eigenvector) result(curl_B_3)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: curl_B_3(size(grid%ef_grid))
    complex(dp) :: a1(size(grid%ef_grid)), da1(size(grid%ef_grid))
    complex(dp) :: a2(size(grid%ef_grid)), a3(size(grid%ef_grid))
    complex(dp) :: da3(size(grid%ef_grid)), dda3(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid)), ef_deps

    a1 = get_base_eigenfunction(sv_a1, settings, grid, eigenvector)
    da1 = get_base_eigenfunction(sv_a1, settings, grid, eigenvector, diff_order=1)
    a2 = get_base_eigenfunction(sv_a2, settings, grid, eigenvector)
    a3 = get_base_eigenfunction(sv_a3, settings, grid, eigenvector)
    da3 = get_base_eigenfunction(sv_a3, settings, grid, eigenvector, diff_order=1)
    dda3 = get_base_eigenfunction(sv_a3, settings, grid, eigenvector, diff_order=2)
    ef_eps = grid%get_eps(grid%ef_grid)
    ef_deps = grid%get_deps()
    curl_B_3 = ( &
      k3 * da1 &
      - dda3 &
      + ef_deps * (ic * k3 * a1 - da3) / ef_eps &
      + k2 * (k2 * a3 / ef_eps - k3 * a2) / ef_eps &
    )
  end function get_curl_B_3


  function get_B_para(settings, grid, eigenvector) result(B_para)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: B_para(size(grid%ef_grid))
    complex(dp) :: B2(size(grid%ef_grid)), B3(size(grid%ef_grid))

    B2 = get_B2(settings, grid, eigenvector)
    B3 = get_B3(settings, grid, eigenvector)
    B_para = ( &
      (B02_on_ef_grid * B2 + B03_on_ef_grid * B3) &
      / sqrt(B02_on_ef_grid**2 + B03_on_ef_grid**2) &
    )
  end function get_B_para


  function get_B_perp(settings, grid, eigenvector) result(B_perp)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: B_perp(size(grid%ef_grid))
    complex(dp) :: B2(size(grid%ef_grid)), B3(size(grid%ef_grid))

    B2 = get_B2(settings, grid, eigenvector)
    B3 = get_B3(settings, grid, eigenvector)
    B_perp = ( &
      (B02_on_ef_grid * B3 - B03_on_ef_grid * B2) &
      / sqrt(B02_on_ef_grid**2 + B03_on_ef_grid**2) &
    )
  end function get_B_perp


  function get_curl_B_para(settings, grid, eigenvector) result(curl_B_para)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: curl_B_para(size(grid%ef_grid))
    complex(dp) :: curl_B_2(size(grid%ef_grid)), curl_B_3(size(grid%ef_grid))

    curl_B_2 = get_curl_B_2(settings, grid, eigenvector)
    curl_B_3 = get_curl_B_3(settings, grid, eigenvector)
    curl_B_para = ( &
      (B02_on_ef_grid * curl_B_2 + B03_on_ef_grid * curl_B_3) &
      / sqrt(B02_on_ef_grid**2 + B03_on_ef_grid**2) &
    )
  end function get_curl_B_para


  function get_curl_B_perp(settings, grid, eigenvector) result(curl_B_perp)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: curl_B_perp(size(grid%ef_grid))
    complex(dp) :: curl_B_2(size(grid%ef_grid)), curl_B_3(size(grid%ef_grid))

    curl_B_2 = get_curl_B_2(settings, grid, eigenvector)
    curl_B_3 = get_curl_B_3(settings, grid, eigenvector)
    curl_B_perp = ( &
      (B02_on_ef_grid * curl_B_3 - B03_on_ef_grid * curl_B_2) &
      / sqrt(B02_on_ef_grid**2 + B03_on_ef_grid**2) &
    )
  end function get_curl_B_perp


  function get_v_para(settings, grid, eigenvector) result(v_para)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: v_para(size(grid%ef_grid))
    complex(dp) :: v2(size(grid%ef_grid)), v3(size(grid%ef_grid))

    v2 = get_base_eigenfunction(sv_v2, settings, grid, eigenvector)
    v3 = get_base_eigenfunction(sv_v3, settings, grid, eigenvector)
    v_para = ( &
      (B02_on_ef_grid * v2 + B03_on_ef_grid * v3) &
      / sqrt(B02_on_ef_grid**2 + B03_on_ef_grid**2) &
    )
  end function get_v_para


  function get_v_perp(settings, grid, eigenvector) result(v_perp)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: v_perp(size(grid%ef_grid))
    complex(dp) :: v2(size(grid%ef_grid)), v3(size(grid%ef_grid))

    v2 = get_base_eigenfunction(sv_v2, settings, grid, eigenvector)
    v3 = get_base_eigenfunction(sv_v3, settings, grid, eigenvector)
    v_perp = ( &
      (B02_on_ef_grid * v3 - B03_on_ef_grid * v2) &
      / sqrt(B02_on_ef_grid**2 + B03_on_ef_grid**2) &
    )
  end function get_v_perp


  function get_curl_v_para(settings, grid, eigenvector) result(curl_v_para)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: curl_v_para(size(grid%ef_grid))
    complex(dp) :: curl_v_2(size(grid%ef_grid))
    complex(dp) :: curl_v_3(size(grid%ef_grid))

    curl_v_2 = get_curl_v_2(settings, grid, eigenvector)
    curl_v_3 = get_curl_v_3(settings, grid, eigenvector)
    curl_v_para = ( &
      (B02_on_ef_grid * curl_v_2 + B03_on_ef_grid * curl_v_3) &
      / sqrt(B02_on_ef_grid**2 + B03_on_ef_grid**2) &
    )
  end function get_curl_v_para


  function get_curl_v_perp(settings, grid, eigenvector) result(curl_v_perp)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: curl_v_perp(size(grid%ef_grid))
    complex(dp) :: curl_v_2(size(grid%ef_grid)), curl_v_3(size(grid%ef_grid))

    curl_v_2 = get_curl_v_2(settings, grid, eigenvector)
    curl_v_3 = get_curl_v_3(settings, grid, eigenvector)
    curl_v_perp = ( &
      (B02_on_ef_grid * curl_v_3 - B03_on_ef_grid * curl_v_2) &
      / sqrt(B02_on_ef_grid**2 + B03_on_ef_grid**2) &
    )
  end function get_curl_v_perp


  function get_dB1(settings, grid, eigenvector) result(dB1)
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: dB1(size(grid%ef_grid))
    complex(dp) :: da2(size(grid%ef_grid))
    complex(dp) :: a3(size(grid%ef_grid))
    complex(dp) :: da3(size(grid%ef_grid))
    real(dp) :: ef_eps(size(grid%ef_grid)), ef_deps

    da2 = get_base_eigenfunction(sv_a2, settings, grid, eigenvector, diff_order=1)
    a3 = get_base_eigenfunction(sv_a3, settings, grid, eigenvector)
    da3 = get_base_eigenfunction(sv_a3, settings, grid, eigenvector, diff_order=1)
    ef_eps = grid%get_eps(grid%ef_grid)
    ef_deps = grid%get_deps()

    dB1 = ic * (k2 * da3 / ef_eps - k3 * da2) - ic * k2 * ef_deps * a3 / ef_eps**2
  end function get_dB1


  function get_base_eigenfunction( &
    sv_component, settings, grid, eigenvector, diff_order &
  ) result(base_ef)
    type(sv_component_t), intent(in) :: sv_component
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    complex(dp), intent(in) :: eigenvector(:)
    integer, intent(in), optional :: diff_order

    complex(dp) :: base_ef(size(grid%ef_grid))
    integer :: derivative_order, sv_idx

    derivative_order = 0
    if (present(diff_order)) derivative_order = diff_order

    sv_idx = get_index(sv_component%get_name(), settings%state_vector%get_names())
    base_ef = assemble_eigenfunction( &
      settings=settings, &
      sv_component=sv_component, &
      grid=grid, &
      state_vector_index=sv_idx, &
      eigenvector=eigenvector, &
      derivative_order=derivative_order &
    )

    if (derivative_order /= 0) return
    ! only retransform for "true" base eigenfunctions, not their derivatives
    base_ef = retransform_eigenfunction( &
      ef_name=sv_component%get_name(), &
      ef_eps=grid%get_eps(grid%ef_grid), &
      eigenfunction=base_ef &
    )
  end function get_base_eigenfunction


  subroutine set_equilibrium_arrays_on_ef_grid(background, grid)
    use mod_function_utils, only: from_function

    type(background_t), intent(in) :: background
    type(grid_t), intent(in) :: grid

    if (.not. allocated(rho0_on_ef_grid)) then
      rho0_on_ef_grid = interpolate_array_on_ef_grid( &
        from_function(background%density%rho0, grid%gaussian_grid), grid &
      )
      T0_on_ef_grid = interpolate_array_on_ef_grid( &
        from_function(background%temperature%T0, grid%gaussian_grid), grid &
      )
      B02_on_ef_grid = interpolate_array_on_ef_grid( &
        from_function(background%magnetic%B02, grid%gaussian_grid), grid &
      )
      B03_on_ef_grid = interpolate_array_on_ef_grid( &
        from_function(background%magnetic%B03, grid%gaussian_grid), grid &
      )
    end if
  end subroutine set_equilibrium_arrays_on_ef_grid


  function interpolate_array_on_ef_grid(array, grid) result(array_on_ef_grid)
    use mod_interpolation, only: lookup_table_value

    real(dp), intent(in) :: array(:)
    type(grid_t), intent(in) :: grid
    real(dp), allocatable :: array_on_ef_grid(:)
    integer :: i

    allocate(array_on_ef_grid(size(grid%ef_grid)))
    do i = 1, size(grid%ef_grid)
      array_on_ef_grid(i) = lookup_table_value( &
        grid%ef_grid(i), grid%gaussian_grid, array, allow_outside=.true. &
      )
    end do
  end function interpolate_array_on_ef_grid


  subroutine deallocate_derived_ef_module_variables()
    if (allocated(rho0_on_ef_grid)) deallocate(rho0_on_ef_grid)
    if (allocated(T0_on_ef_grid)) deallocate(T0_on_ef_grid)
    if (allocated(B02_on_ef_grid)) deallocate(B02_on_ef_grid)
    if (allocated(B03_on_ef_grid)) deallocate(B03_on_ef_grid)
  end subroutine deallocate_derived_ef_module_variables


end module mod_derived_efs
