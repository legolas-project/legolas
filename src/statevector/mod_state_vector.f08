module mod_state_vector
  use mod_global_variables, only: str_len_arr
  use mod_state_vector_component, only: sv_component_t
  use mod_logging, only: logger, str
  implicit none

  private

  type, private :: sv_comp_ptr_t
    type(sv_component_t), pointer :: ptr
  end type sv_comp_ptr_t

  type, public :: state_vector_t
    type(sv_comp_ptr_t), allocatable :: components(:)

  contains
    procedure, public :: assemble
    procedure, public :: set_basis_functions
    procedure, public :: get_names
    procedure, public :: get_basis_functions
    procedure, public :: get_components_from_basis_function
    procedure, public :: delete

    generic, public :: contains => contains_on_component, contains_on_name

    procedure, private :: set_default_basis_functions
    procedure, private :: is_compatible_with
    procedure, private :: contains_on_component
    procedure, private :: contains_on_name
  end type state_vector_t

  logical, save, private :: sv_components_initialised = .false.
  type(sv_component_t), public, protected, target :: sv_rho1
  type(sv_component_t), public, protected, target :: sv_v1
  type(sv_component_t), public, protected, target :: sv_v2
  type(sv_component_t), public, protected, target :: sv_v3
  type(sv_component_t), public, protected, target :: sv_T1
  type(sv_component_t), public, protected, target :: sv_a1
  type(sv_component_t), public, protected, target :: sv_a2
  type(sv_component_t), public, protected, target :: sv_a3

contains

  subroutine assemble(this, physics_type)
    class(state_vector_t), intent(inout) :: this
    character(len=*), intent(in) :: physics_type

    if (.not. sv_components_initialised) call initialise_sv_components()
    if (allocated(this%components)) deallocate(this%components)

    select case(physics_type)
      case("isothermal-1d")
        allocate(this%components(2))
        this%components(1)%ptr => sv_rho1
        this%components(2)%ptr => sv_v1
      case("hd")
        allocate(this%components(5))
        this%components(1)%ptr => sv_rho1
        this%components(2)%ptr => sv_v1
        this%components(3)%ptr => sv_v2
        this%components(4)%ptr => sv_v3
        this%components(5)%ptr => sv_T1
      case("hd-1d")
        allocate(this%components(3))
        this%components(1)%ptr => sv_rho1
        this%components(2)%ptr => sv_v1
        this%components(3)%ptr => sv_T1
      case default
        allocate(this%components(8))
        this%components(1)%ptr => sv_rho1
        this%components(2)%ptr => sv_v1
        this%components(3)%ptr => sv_v2
        this%components(4)%ptr => sv_v3
        this%components(5)%ptr => sv_T1
        this%components(6)%ptr => sv_a1
        this%components(7)%ptr => sv_a2
        this%components(8)%ptr => sv_a3
    end select
    call this%set_default_basis_functions()
  end subroutine assemble


  subroutine set_basis_functions(this, splines)
    class(state_vector_t), intent(inout) :: this
    character(len=*), intent(in) :: splines(:)
    integer :: i

    if (size(splines) == 0) then
      call this%set_default_basis_functions()
      return
    end if
    if (.not. this%is_compatible_with(splines)) return
    do i = 1, size(this%components)
      call this%components(i)%ptr%set_basis_function(splines(i))
    end do
  end subroutine set_basis_functions


  subroutine set_default_basis_functions(this)
    class(state_vector_t), intent(inout) :: this
    character(:), allocatable :: default_name
    integer :: i

    do i = 1, size(this%components)
      default_name = this%components(i)%ptr%get_default_basis_function()
      call this%components(i)%ptr%set_basis_function(default_name)
    end do
  end subroutine set_default_basis_functions


  pure function get_names(this) result(names)
    class(state_vector_t), intent(in) :: this
    character(len=:), allocatable :: names(:)
    integer :: i

    allocate(names(size(this%components)), mold=this%components(1)%ptr%get_name())
    do i = 1, size(this%components)
      names(i) = this%components(i)%ptr%get_name()
    end do
  end function get_names


  function get_basis_functions(this) result(names)
    class(state_vector_t), intent(in) :: this
    character(len=:), allocatable :: names(:)
    integer :: i

    allocate(names(size(this%components)), mold=this%components(1)%ptr%get_name())
    do i = 1, size(this%components)
      names(i) = this%components(i)%ptr%get_basis_function_name()
    end do
  end function get_basis_functions


  function get_components_from_basis_function(this, basis_function_name) result(comps)
    class(state_vector_t), intent(in) :: this
    character(len=*), intent(in) :: basis_function_name
    type(sv_component_t), allocatable :: comps(:)
    integer :: i
    logical, allocatable :: mask(:)

    allocate(comps(size(this%components)))
    allocate(mask(size(this%components)))
    do i = 1, size(comps)
      comps(i) = this%components(i)%ptr
      mask(i) = this%components(i)%ptr%get_basis_function_name() == basis_function_name
    end do
    comps = pack(comps, mask)
    deallocate(mask)
  end function get_components_from_basis_function


  logical function contains_on_component(this, component)
    class(state_vector_t), intent(in) :: this
    type(sv_component_t), intent(in) :: component
    contains_on_component = this%contains_on_name(component%get_name())
  end function contains_on_component


  logical function contains_on_name(this, name)
    class(state_vector_t), intent(in) :: this
    character(len=*), intent(in) :: name
    integer :: i

    contains_on_name = .false.
    do i = 1, size(this%components)
      if (this%components(i)%ptr%get_name() == name) then
        contains_on_name = .true.
        return
      end if
    end do
  end function contains_on_name


  logical function is_compatible_with(this, splines)
    class(state_vector_t), intent(in) :: this
    character(len=*), intent(in) :: splines(:)
    integer :: size_sv, size_splines

    size_sv = size(this%components)
    size_splines = size(splines)
    is_compatible_with = .true.
    if (size_sv == size_splines) return

    is_compatible_with = .false.
    call logger%error( &
      "state vector size (" // str(size_sv) // ") is not compatible with " &
      // "given number of basis functions (" // str(size_splines) // ")" &
    )
  end function is_compatible_with


  subroutine delete(this)
    class(state_vector_t), intent(inout) :: this
    integer :: i

    call sv_rho1%delete()
    call sv_v1%delete()
    call sv_v2%delete()
    call sv_v3%delete()
    call sv_T1%delete()
    call sv_a1%delete()
    call sv_a2%delete()
    call sv_a3%delete()
    sv_components_initialised = .false.
    if (allocated(this%components)) then
      do i = 1, size(this%components)
        nullify(this%components(i)%ptr)
      end do
      deallocate(this%components)
    end if
  end subroutine delete


  subroutine initialise_sv_components()
    use mod_state_vector_component, only: new_sv_component
    use mod_state_vector_names

    sv_rho1 = new_sv_component(sv_rho1_name)
    sv_v1 = new_sv_component(sv_v1_name)
    sv_v2 = new_sv_component(sv_v2_name)
    sv_v3 = new_sv_component(sv_v3_name)
    sv_T1 = new_sv_component(sv_T1_name)
    sv_a1 = new_sv_component(sv_a1_name)
    sv_a2 = new_sv_component(sv_a2_name)
    sv_a3 = new_sv_component(sv_a3_name)
    sv_components_initialised = .true.
    call logger%debug("initialised state vector module components")
  end subroutine initialise_sv_components


end module mod_state_vector
