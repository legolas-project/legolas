module mod_build_quadblock
  use mod_global_variables, only: dp
  use mod_dims, only: dims_t
  use mod_logging, only: logger, str
  use mod_matrix_element_node, only: matrix_element_node_t
  use mod_matrix_elements, only: matrix_elements_t
  implicit none

  private

  public :: add_to_quadblock

contains

  !> This routine builds the quadblock at one particular grid point in the Gaussian
  !! grid and for one particular Gaussian weight.
  !! For a 2x2 block at index \((i, j)\) in the top-left block we have
  !! \((2i, 2j)\) as index of the bottom-right corner of the 2x2 block. The other
  !! corners are then filled by subtracting one from an index.
  subroutine add_to_quadblock(quadblock, elements, x, x0, x1, weight, dims)
    complex(dp), intent(inout) :: quadblock(:, :)
    type(matrix_elements_t), intent(in) :: elements
    real(dp), intent(in) :: x
    real(dp), intent(in) :: x0
    real(dp), intent(in) :: x1
    real(dp), intent(in) :: weight
    type(dims_t), intent(in) :: dims

    type(matrix_element_node_t), pointer :: node
    complex(dp), allocatable :: new_quadblock(:, :)
    complex(dp) :: factor
    real(dp), allocatable :: spline1(:), spline2(:)
    integer :: idxs(2), position(2)
    integer :: inode, dim_subblock

    allocate(new_quadblock, mold=quadblock)
    dim_subblock = dims%get_dim_subblock()

    do inode = 1, elements%get_nb_elements()
      node => elements%get_node(inode)
      if (.not. node_has_spline(node)) return
      allocate(spline1, source=node%spline1(x, x0, x1))
      allocate(spline2, source=node%spline2(x, x0, x1))

      new_quadblock = (0.0_dp, 0.0_dp)
      idxs = 0
      factor = weight * node%get_element()
      position = node%get_position()

      ! subblock top-left corner
      idxs = 2 * position
      new_quadblock(idxs(1) - 1, idxs(2) - 1) = spline1(2) * factor * spline2(2)
      new_quadblock(idxs(1) - 1, idxs(2)) = spline1(2) * factor * spline2(4)
      new_quadblock(idxs(1), idxs(2) - 1) = spline1(4) * factor * spline2(2)
      new_quadblock(idxs(1), idxs(2)) = spline1(4) * factor * spline2(4)
      ! subblock top-right corner
      idxs = [2 * position(1), 2 * position(2) + dim_subblock]
      new_quadblock(idxs(1) - 1, idxs(2) - 1) = spline1(2) * factor * spline2(1)
      new_quadblock(idxs(1) - 1, idxs(2)) = spline1(2) * factor * spline2(3)
      new_quadblock(idxs(1), idxs(2) - 1) = spline1(4) * factor * spline2(1)
      new_quadblock(idxs(1), idxs(2)) = spline1(4) * factor * spline2(3)
      ! subblock bottom-left corner
      idxs = [2 * position(1) + dim_subblock, 2 * position(2)]
      new_quadblock(idxs(1) - 1, idxs(2) - 1) = spline1(1) * factor * spline2(2)
      new_quadblock(idxs(1) - 1, idxs(2)) = spline1(1) * factor * spline2(4)
      new_quadblock(idxs(1), idxs(2) - 1) = spline1(3) * factor * spline2(2)
      new_quadblock(idxs(1), idxs(2)) = spline1(3) * factor * spline2(4)
      ! subblock bottom-right corner
      idxs = 2 * position + dim_subblock
      new_quadblock(idxs(1) - 1, idxs(2) - 1) = spline1(1) * factor * spline2(1)
      new_quadblock(idxs(1) - 1, idxs(2)) = spline1(1) * factor * spline2(3)
      new_quadblock(idxs(1), idxs(2) - 1) = spline1(3) * factor * spline2(1)
      new_quadblock(idxs(1), idxs(2)) = spline1(3) * factor * spline2(3)

      quadblock = quadblock + new_quadblock

      deallocate(spline1)
      deallocate(spline2)
    end do

    nullify(node)
    deallocate(new_quadblock)
  end subroutine add_to_quadblock


  logical function node_has_spline(node)
    type(matrix_element_node_t), intent(in) :: node

    node_has_spline = associated(node%spline1) .and. associated(node%spline2)
    if (.not. node_has_spline) call logger%error( &
      "matrix element node has no spline procedure linked! Are basis function set?" &
    )
  end function node_has_spline

end module mod_build_quadblock
