module mod_matrix_elements
  use mod_global_variables, only: dp, NaN
  use mod_matrix_element_node, only: matrix_element_node_t, new_matrix_element_node
  use mod_basis_functions, only: basis_function
  use mod_state_vector, only: state_vector_t
  use mod_state_vector_component, only: sv_component_t
  use mod_get_indices, only: get_index
  use mod_logging, only: logger, str
  implicit none

  private

  type, public :: matrix_elements_t
    integer, private :: nb_elements
    type(matrix_element_node_t), pointer, private :: head
    type(matrix_element_node_t), pointer, private :: tail
    type(state_vector_t), pointer, private :: state_vector

  contains

    procedure, public :: add => add_node
    procedure, public :: get_node
    procedure, public :: get_elements
    procedure, public :: get_positions
    procedure, public :: get_nb_elements
    procedure, public :: delete
    procedure, private :: increment_nb_elements
  end type matrix_elements_t

  public ::  new_matrix_elements

contains

  function new_matrix_elements(state_vector) result(elements)
    type(state_vector_t), target, intent(in) :: state_vector
    type(matrix_elements_t) :: elements
    elements%nb_elements = 0
    elements%head => null()
    elements%tail => null()
    elements%state_vector => state_vector
  end function new_matrix_elements


  subroutine add_node(this, element, sv_comp1, sv_comp2, s1do, s2do)
    class(matrix_elements_t), intent(inout) :: this
    class(*), intent(in) :: element
    !> first state vector component
    type(sv_component_t), intent(in) :: sv_comp1
    !> second state vector component
    type(sv_component_t), intent(in) :: sv_comp2
    !> spline 1 derivative order, 1 = first derivative, 2 = second derivative, etc.
    integer, intent(in), optional :: s1do
    !> spline 2 derivative order, 1 = first derivative, 2 = second derivative, etc.
    integer, intent(in), optional :: s2do

    complex(dp) :: node_element
    integer :: position(2)
    procedure(basis_function), pointer :: spline1, spline2

    ! position checks
    position(1) = get_index( &
      name=sv_comp1%get_name(), array=this%state_vector%get_names() &
    )
    position(2) = get_index( &
      name=sv_comp2%get_name(), array=this%state_vector%get_names() &
    )
    if (.not. is_valid_position(position)) return

    node_element = cast_node_element_to_complex(element)
    if (is_NaN_element(node_element, sv_comp1, sv_comp2)) return
    if (is_inf_element(node_element, sv_comp1, sv_comp2)) return

    call sv_comp1%get_spline_function(spline_order=s1do, spline_func=spline1)
    call sv_comp2%get_spline_function(spline_order=s2do, spline_func=spline2)
    ! if head is not associated, then the list is empty and we create the first node
    if (.not. associated(this%head)) then
      allocate( &
        this%head, &
        source=new_matrix_element_node(node_element, position, spline1, spline2) &
      )
      this%tail => this%head
    else
      allocate( &
        this%tail%next, &
        source=new_matrix_element_node(node_element, position, spline1, spline2) &
      )
      this%tail => this%tail%next
    end if
    call this%increment_nb_elements()
  end subroutine add_node


  function get_node(this, inode) result(node)
    class(matrix_elements_t), intent(in) :: this
    integer, intent(in) :: inode
    type(matrix_element_node_t), pointer :: node
    type(matrix_element_node_t), pointer :: current_node
    integer :: i

    node => null()
    if (inode < 1 .or. inode > this%nb_elements) then
      call logger%error("get_node: inode out of range: " // str(inode))
      return
    end if
    current_node => this%head
    do i = 1, inode - 1
      current_node => current_node%next
    end do
    node => current_node
    nullify(current_node)
  end function get_node


  function get_elements(this) result(elements)
    class(matrix_elements_t), intent(in) :: this
    complex(dp) :: elements(this%nb_elements)
    type(matrix_element_node_t), pointer :: current_node
    integer :: i

    current_node => this%head
    do i = 1, this%nb_elements
      elements(i) = current_node%get_element()
      current_node => current_node%next
    end do
    nullify(current_node)
  end function get_elements


  function get_positions(this) result(positions)
    class(matrix_elements_t), intent(in) :: this
    integer :: positions(this%nb_elements, 2)
    type(matrix_element_node_t), pointer :: current_node
    integer :: i

    current_node => this%head
    do i = 1, this%nb_elements
      positions(i, :) = current_node%get_position()
      current_node => current_node%next
    end do
    nullify(current_node)
  end function get_positions


  pure integer function get_nb_elements(this)
    class(matrix_elements_t), intent(in) :: this
    get_nb_elements = this%nb_elements
  end function get_nb_elements


  pure function cast_node_element_to_complex(element) result(node_element)
    class(*), intent(in) :: element
    complex(dp) :: node_element

    select type(element)
      type is (complex(dp))
        node_element = element
      type is (real(dp))
        node_element = cmplx(element, 0.0_dp, kind=dp)
      type is (integer)
        node_element = cmplx(element, 0.0_dp, kind=dp)
      class default
        node_element = cmplx(NaN, NaN, kind=dp)
    end select
  end function cast_node_element_to_complex


  pure logical function is_valid_position(position)
    integer, intent(in) :: position(:)
    is_valid_position = all(position > 0)
  end function is_valid_position


  logical function is_NaN_element(element, sv_comp1, sv_comp2)
    use mod_check_values, only: is_NaN

    complex(dp), intent(in) :: element
    type(sv_component_t), intent(in) :: sv_comp1
    type(sv_component_t), intent(in) :: sv_comp2

    is_NaN_element = is_NaN(element)
    if (is_NaN_element) call logger%error( &
      "NaN matrix element encountered for [" &
      // trim(sv_comp1%get_name()) // ", " // trim(sv_comp2%get_name()) // "]" &
    )
  end function is_NaN_element


  logical function is_inf_element(element, sv_comp1, sv_comp2)
    use mod_check_values, only: is_infinite

    complex(dp), intent(in) :: element
    type(sv_component_t), intent(in) :: sv_comp1
    type(sv_component_t), intent(in) :: sv_comp2

    is_inf_element = is_infinite(element)
    if (is_inf_element) call logger%error( &
      "Infinity encountered in matrix element for [" &
      // trim(sv_comp1%get_name()) // ", " // trim(sv_comp2%get_name()) // "]" &
    )
  end function is_inf_element


  pure subroutine increment_nb_elements(this)
    class(matrix_elements_t), intent(inout) :: this
    this%nb_elements = this%nb_elements + 1
  end subroutine increment_nb_elements


  pure subroutine delete(this)
    class(matrix_elements_t), intent(inout) :: this
    type(matrix_element_node_t), pointer :: current_node, next_node

    if (.not. associated(this%head)) return

    current_node => this%head
    next_node => current_node%next
    do while (associated(current_node))
      call current_node%delete()
      deallocate(current_node)
      current_node => next_node
      if (associated(current_node)) next_node => current_node%next
    end do
    nullify(current_node)
    nullify(next_node)
    nullify(this%head)
    nullify(this%tail)
    nullify(this%state_vector)
    this%nb_elements = 0
  end subroutine delete

end module mod_matrix_elements
