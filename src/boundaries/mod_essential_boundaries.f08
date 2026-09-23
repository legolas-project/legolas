module mod_essential_boundaries
  use mod_global_variables, only: dp, str_len_arr
  use mod_check_values, only: is_zero
  use mod_equilibrium_params, only: k2, k3
  use mod_matrix_structure, only: matrix_t
  use mod_settings, only: settings_t
  use mod_state_vector_component, only: sv_component_t
  use mod_state_vector, only: sv_rho1, sv_v1, sv_v2, sv_v3, sv_T1, sv_a1, sv_a2, sv_a3
  use mod_basis_function_names, only: QUADRATIC, CUBIC
  use mod_logging, only: logger, str
  implicit none

  private

  character(len=5), parameter :: LEFT = "left"
  character(len=6), parameter :: RIGHT = "right"
  character(len=4), parameter :: EVEN = "even"
  character(len=3), parameter :: ODD = "odd"

  public :: apply_essential_boundaries_left
  public :: apply_essential_boundaries_right
  public :: get_block_indices

contains

  subroutine apply_essential_boundaries_left(matrix, settings)
    type(matrix_t), intent(inout) :: matrix
    type(settings_t), intent(in) :: settings
    type(sv_component_t), allocatable :: components(:)
    integer :: limits(2)

    call logger%debug( &
      "applying left essential boundary conditions for " &
      // matrix%get_label() // " matrix" &
    )

    ! left side quadblock limits are (1, 1) -> (dim_quadblock, dim_quadblock)
    limits = [1, settings%dims%get_dim_quadblock()]

    ! The quadratic basis functions have a zero (2nd entry) for the left side, which
    ! automatically zeroes out odd rows/columns. We explicitly handle this by
    ! introducing an element on the diagonal for those indices.
    components = settings%state_vector%get_components_from_basis_function(QUADRATIC)
    call zero_out_row_and_col( &
      matrix=matrix, &
      idxs=get_block_indices( &
        sv_components=components, settings=settings, edge=LEFT, force_parity=ODD &
      ), &
      limits=limits &
    )
    ! wall/regularity conditions: v1 must be zero
    components = [sv_v1]
    ! (k3 * a2 - k2 * a3) has to be zero
    select case(settings%equilibrium%get_boundary_type())
    case("wall")
      ! for "wall" force a2 = a3 = 0 regardless of k2 and k3
      components = [components, sv_a2, sv_a3]
    case("wall_weak")
      ! for "wall_weak" we only force a2 resp. a3 to zero if k3 resp. k2 is nonzero
      if (.not. is_zero(k2)) components = [components, sv_a3]
      if (.not. is_zero(k3)) components = [components, sv_a2]
    end select
    ! check T boundary conditions
    if (needs_T_bounds(settings)) components = [components, sv_T1]
    ! In the case of no-slip boundary conditions, then v2 and v3 should equal the
    ! wall's tangential velocities, here zero.
    if (needs_noslip_left(settings)) components = [components, sv_v2, sv_v3]
    call zero_out_row_and_col( &
      matrix=matrix, &
      idxs=get_block_indices( &
        sv_components=components, settings=settings, edge=LEFT &
      ), &
      limits=limits &
    )
    if (allocated(components)) deallocate(components)
  end subroutine apply_essential_boundaries_left


  subroutine apply_essential_boundaries_right(matrix, settings)
    type(matrix_t), intent(inout) :: matrix
    type(settings_t), intent(in) :: settings
    type(sv_component_t), allocatable :: components(:)
    integer :: ishift, limits(2)

    call logger%debug( &
      "applying right essential boundary conditions for " &
      // matrix%get_label() // " matrix" &
    )

    ! index shift, even number so we get the end of previous block
    ishift = matrix%matrix_dim - settings%dims%get_dim_quadblock()
    ! last block indices hence run from ishift + 1 to matrix dimension
    limits = [ishift + 1, matrix%matrix_dim]

    ! wall conditions: v1 must be zero
    components = [sv_v1]
    ! (k3 * a2 - k2 * a3) has to be zero
    select case(settings%equilibrium%get_boundary_type())
    case("wall")
      components = [components, sv_a2, sv_a3]
    case("wall_weak")
      if (.not. is_zero(k2)) components = [components, sv_a3]
      if (.not. is_zero(k3)) components = [components, sv_a2]
    end select
    ! check T boundary conditions
    if (needs_T_bounds(settings)) components = [components, sv_T1]
    ! check no-slip
    if (needs_noslip_right(settings)) components = [components, sv_v2, sv_v3]
    call zero_out_row_and_col( &
      matrix=matrix, &
      idxs=ishift + get_block_indices( &
        sv_components=components, settings=settings, edge=RIGHT &
      ), &
      limits=limits &
    )
    if (allocated(components)) deallocate(components)
  end subroutine apply_essential_boundaries_right


  function get_block_indices(sv_components, settings, edge, force_parity) result(idxs)
    !> array containing state vector components for which to retrieve indices
    type(sv_component_t), intent(in) :: sv_components(:)
    !> settings object
    type(settings_t), intent(in) :: settings
    !> edge for which to retrieve indices
    character(len=*), intent(in) :: edge
    !> parity is based on the basis functions unless forced through this argument
    character(len=*), intent(in), optional :: force_parity
    !> array containing the corresponding block indices
    integer, allocatable :: idxs(:)

    character(len=str_len_arr) :: comp_names(size(sv_components))
    integer :: i

    if (size(sv_components) == 0) return

    allocate(idxs(size(sv_components)))
    do i = 1, size(sv_components)
      idxs(i) = get_block_index_for_single_component( &
        sv_components(i), settings, edge, force_parity &
      )
    end do
    idxs = pack(idxs, idxs > 0)

    if (size(idxs) == 0) then
      do i = 1, size(sv_components)
        comp_names(i) = sv_components(i)%get_name()
      end do
      call logger%error( &
        "could not retrieve subblock indices for any variable in " &
        // str(comp_names) &
        // " for state vector " &
        // str(settings%state_vector%get_names()) &
      )
      return
    end if
  end function get_block_indices


  function get_block_index_for_single_component( &
    component, settings, edge, force_parity &
  ) result(idx)
    use mod_get_indices, only: get_index

    type(sv_component_t), intent(in) :: component
    type(settings_t), intent(in) :: settings
    character(len=*), intent(in) :: edge
    character(len=*), intent(in), optional :: force_parity
    integer :: idx
    logical :: is_odd

    idx = 2 * get_index(component%get_name(), settings%state_vector%get_names())
    if (idx == 0) return

    if (present(force_parity)) then
      is_odd = (force_parity == ODD)
    else
      is_odd = get_odd_parity_from_basis_function_name( &
        component%get_basis_function_name() &
      )
    end if

    if (is_odd) idx = idx - 1
    if (edge == RIGHT) idx = idx + settings%dims%get_dim_subblock()
  end function get_block_index_for_single_component


  !> Zeroes out the row and column corresponding to the given indices.
  !! Afterwards `diagonal_factor` is introduced in that row/column on the main diagonal.
  subroutine zero_out_row_and_col(matrix, idxs, limits)
    !> the matrix under consideration
    type(matrix_t), intent(inout) :: matrix
    !> indices of the row and column to zero out
    integer, intent(in) :: idxs(:)
    !> (start, end) limits of quadblock corresponding to (start, start):(end, end)
    integer, intent(in) :: limits(2)
    integer :: i, k, idx

    call logger%debug("zeroing out indices " // str(idxs))
    do i = 1, size(idxs)
      idx = idxs(i)
      do k = limits(1), limits(2)
        ! row at idx, columns within limits
        call matrix%rows(idx)%delete_node_from_row(column=k)
        ! column at idx, rows within limits
        call matrix%rows(k)%delete_node_from_row(column=idx)
      end do
      ! add diagonal factor to main diagonal
      call matrix%add_element(row=idx, column=idx, element=get_diagonal_factor(matrix))
    end do
  end subroutine zero_out_row_and_col


    !! zeroing out the corresponding row and column.
  !! Depends on the matrix that is used.
  function get_diagonal_factor(matrix) result(diagonal_factor)
    use mod_global_variables, only: NaN

    type(matrix_t), intent(in) :: matrix
    complex(dp) :: diagonal_factor

    if (matrix%get_label() == "B") then
      diagonal_factor = (1.0d0, 0.0d0)
    else if (matrix%get_label() == "A") then
      diagonal_factor = (0.0d0, 0.0d0)
    else
      diagonal_factor = NaN
      call logger%error( &
        "get_diagonal_factor: invalid or empty matrix label: " // matrix%get_label() &
      )
    end if
  end function get_diagonal_factor


  !> This relies on the behaviour of the basis functions. If a variable needs to be
  !! zero, then this is done by forcing the basis functions on that edge to zero.
  !! For both left and right edges the quadratic basis functions have a non-zero entry
  !! in their even rows/columns, whereas the cubic basis functions have a non-zero entry
  !! in their odd rows/columns.
  !! Concrete:
  !! - cubic, left: C2 is non-zero, zero out elements with spline(2)
  !! - cubic, right: C1 is non-zero, zero out elements with spline(1)
  !! - quad, left: Q4 is non-zero, zero out elements with spline(4)
  !! - quad, right: Q3 is non-zero, zero out elements with spline(3)
  !! See also ordening of a quadblock in `mod_build_quadblock`.
  logical function get_odd_parity_from_basis_function_name(name) result(is_odd)
    character(len=*), intent(in) :: name

    select case(name)
    case(QUADRATIC)
      is_odd = .false.
    case(CUBIC)
      is_odd = .true.
    case default
      call logger%error( &
        "unable to retrieve parity -- not implemented for basis function: " // name &
      )
      is_odd = .false.
    end select
  end function get_odd_parity_from_basis_function_name


  !> Checks if we need regularity conditions on temperature, this is the case
  !! if we have perpendicular thermal conduction.
  pure logical function needs_T_bounds(settings)
    type(settings_t), intent(in) :: settings

    needs_T_bounds = settings%physics%conduction%has_perpendicular_conduction()
  end function needs_T_bounds


  !> Check if we need a no-slip condition on the left-hand side (i.e. viscosity).
  !! Does not apply on-axis for cylindrical unless two coaxial wall are present.
  pure logical function needs_noslip_left(settings)
    type(settings_t), intent(in) :: settings
    needs_noslip_left = ( &
      settings%physics%viscosity%is_enabled() &
      .and. ( &
        settings%grid%coaxial .or. settings%grid%get_geometry() == "Cartesian" &
      ) &
    )
  end function needs_noslip_left


  !> Check if we need a no-slip condition on the right-hand side (i.e. viscosity).
  pure logical function needs_noslip_right(settings)
    type(settings_t), intent(in) :: settings
    needs_noslip_right = settings%physics%viscosity%is_enabled()
  end function needs_noslip_right


end module mod_essential_boundaries
