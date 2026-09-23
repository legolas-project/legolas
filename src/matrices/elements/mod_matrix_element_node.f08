module mod_matrix_element_node
  use mod_global_variables, only: dp
  use mod_basis_functions, only: basis_function
  implicit none

  private

  type, public :: matrix_element_node_t
    complex(dp), private :: element
    integer, private :: position(2)
    procedure(basis_function), pointer, nopass, public :: spline1
    procedure(basis_function), pointer, nopass, public :: spline2
    type(matrix_element_node_t), pointer :: next

  contains

    procedure, public :: get_element
    procedure, public :: get_position
    procedure, public :: delete
  end type matrix_element_node_t

  public :: new_matrix_element_node

contains

  pure function new_matrix_element_node( &
    element, position, spline1, spline2 &
  ) result(node)
    complex(dp), intent(in) :: element
    integer, intent(in) :: position(2)
    procedure(basis_function), pointer, intent(in) :: spline1
    procedure(basis_function), pointer, intent(in) :: spline2
    type(matrix_element_node_t) :: node

    node%element = element
    node%position = position
    node%spline1 => spline1
    node%spline2 => spline2
    node%next => null()
  end function new_matrix_element_node


  pure complex(dp) function get_element(this)
    class(matrix_element_node_t), intent(in) :: this
    get_element = this%element
  end function get_element


  pure function get_position(this) result(position)
    class(matrix_element_node_t), intent(in) :: this
    integer :: position(2)
    position = this%position
  end function get_position


  pure subroutine delete(this)
    class(matrix_element_node_t), intent(inout) :: this
    nullify(this%spline1)
    nullify(this%spline2)
    nullify(this%next)
  end subroutine delete

end module mod_matrix_element_node
