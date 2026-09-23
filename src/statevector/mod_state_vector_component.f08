module mod_state_vector_component
  use mod_global_variables, only: str_len_arr
  use mod_basis_function_names, only: QUADRATIC, CUBIC
  use mod_basis_functions, only: basis_function
  use mod_logging, only: logger, str
  implicit none

  private

  character(str_len_arr), parameter, private :: CUSTOM = "custom"

  type, public :: sv_component_t
    character(len=str_len_arr), private :: spline_name
    character(len=str_len_arr), private :: name
    logical, private :: is_initialised = .false.

    procedure(basis_function), pointer, private, nopass :: custom_spline
  contains
    procedure, public :: get_name
    procedure, public :: get_spline_function
    procedure, public :: get_basis_function_name
    procedure, public :: get_default_basis_function
    procedure, public :: set_basis_function
    procedure, public :: set_custom_spline
    procedure, public :: delete

    procedure, private :: spline
    procedure, private :: dspline
    procedure, private :: ddspline
  end type sv_component_t

  public :: new_sv_component

contains

  function new_sv_component(name) result(sv_comp)
    character(len=*), intent(in) :: name
    type(sv_component_t) :: sv_comp

    sv_comp%name = name
    sv_comp%spline_name = ""
    sv_comp%is_initialised = .true.
  end function new_sv_component


  pure function get_name(this) result(name)
    class(sv_component_t), intent(in) :: this
    character(:), allocatable :: name
    name = this%name
  end function get_name


  function get_basis_function_name(this) result(name)
    class(sv_component_t), intent(in) :: this
    character(:), allocatable :: name
    name = this%spline_name
  end function get_basis_function_name


  function get_default_basis_function(this) result(name)
    use mod_state_vector_names

    class(sv_component_t), intent(in) :: this
    character(:), allocatable :: name

    select case(this%name)
    case(sv_rho1_name, sv_v2_name, sv_v3_name, sv_T1_name, sv_a1_name)
      name = QUADRATIC
    case(sv_v1_name, sv_a2_name, sv_a3_name)
      name = CUBIC
    case default
      call logger%error( &
        "default basis function not defined for state vector component " &
        // trim(adjustl(this%name)) &
      )
    end select
  end function get_default_basis_function


  subroutine set_basis_function(this, spline_name)
    class(sv_component_t), intent(inout) :: this
    character(len=*), intent(in) :: spline_name

    select case(spline_name)
      case(QUADRATIC, CUBIC)
        this%spline_name = spline_name
      case default
        call logger%error("unknown basis function name: " // trim(adjustl(spline_name)))
    end select
  end subroutine set_basis_function


  subroutine set_custom_spline(this, func)
    class(sv_component_t), intent(inout) :: this
    procedure(basis_function) :: func

    this%spline_name = CUSTOM
    this%custom_spline => func
  end subroutine set_custom_spline


  subroutine get_spline_function(this, spline_order, spline_func)
    class(sv_component_t), intent(in) :: this
    integer, intent(in), optional :: spline_order
    procedure(basis_function), pointer, intent(out) :: spline_func
    integer :: order

    if (this%spline_name == CUSTOM) then
      spline_func => this%custom_spline
      return
    end if

    order = 0
    if (present(spline_order)) order = spline_order

    spline_func => null()
    select case(order)
    case(0)
      call this%spline(spline_func)
    case(1)
      call this%dspline(spline_func)
    case(2)
      call this%ddspline(spline_func)
    case default
      call logger%error( &
        "spline order = " // str(order) // " not implemented for spline " &
        // trim(adjustl(this%spline_name)) &
      )
      return
    end select
  end subroutine get_spline_function


  subroutine spline(this, func)
    use mod_basis_functions, only: hquad, hcubic

    class(sv_component_t), intent(in) :: this
    procedure(basis_function), pointer, intent(out) :: func

    func => null()
    select case(this%spline_name)
      case(QUADRATIC)
        func => hquad
      case(CUBIC)
        func => hcubic
      case default
        call logger%error( &
          trim(adjustl(this%spline_name)) &
          // " has no implemented basis function" &
        )
        return
    end select
  end subroutine spline


  subroutine dspline(this, func)
    use mod_basis_functions, only: dhquad, dhcubic

    class(sv_component_t), intent(in) :: this
    procedure(basis_function), pointer, intent(out) :: func

    func => null()
    select case(this%spline_name)
      case(QUADRATIC)
        func => dhquad
      case(CUBIC)
        func => dhcubic
      case default
        call logger%error( &
          trim(adjustl(this%spline_name)) &
          // " basis function has no defined derivative" &
        )
        return
    end select
  end subroutine dspline


  subroutine ddspline(this, func)
    use mod_basis_functions, only: ddhcubic

    class(sv_component_t), intent(in) :: this
    procedure(basis_function), pointer, intent(out) :: func

    func => null()
    select case(this%spline_name)
      case(CUBIC)
        func => ddhcubic
      case default
        call logger%error( &
          trim(adjustl(this%spline_name)) &
          // " basis function has no defined second derivative" &
        )
        return
    end select
  end subroutine ddspline


  pure subroutine delete(this)
    class(sv_component_t), intent(inout) :: this
    this%name = ""
    this%spline_name = ""
    this%is_initialised = .false.
    nullify(this%custom_spline)
  end subroutine delete

end module mod_state_vector_component
