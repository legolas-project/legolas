module mod_natural_boundaries
  use mod_global_variables, only: dp, ic
  use mod_matrix_structure, only: matrix_t
  use mod_matrix_elements, only: matrix_elements_t, new_matrix_elements
  use mod_settings, only: settings_t
  use mod_grid, only: grid_t
  use mod_background, only: background_t
  use mod_physics, only: physics_t
  use mod_state_vector, only: sv_rho1, sv_v1, sv_v2, sv_v3, sv_T1, sv_a1, sv_a2, sv_a3
  use mod_equilibrium_params, only: k2, k3
  use mod_logging, only: logger
  implicit none
  private

  interface
    module subroutine add_natural_regular_terms(x, elements, settings, grid, background)
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
    end subroutine add_natural_regular_terms

    module subroutine add_natural_flow_terms(x, elements, settings, grid, background)
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
    end subroutine add_natural_flow_terms

    module subroutine add_natural_resistive_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_natural_resistive_terms

    module subroutine add_natural_conduction_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_natural_conduction_terms

    module subroutine add_natural_viscosity_terms( &
      x, elements, settings, grid, background &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
    end subroutine add_natural_viscosity_terms

    module subroutine add_natural_hall_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_natural_hall_terms

    module subroutine add_natural_hall_Bterms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_natural_hall_Bterms
  end interface

  public :: apply_natural_boundaries_left
  public :: apply_natural_boundaries_right

contains

  subroutine apply_natural_boundaries_left(matrix, settings, grid, background, physics)
    type(matrix_t), intent(inout) :: matrix
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics

    complex(dp), allocatable :: quadblock(:, :)
    integer :: i, j, dim_quadblock

    call logger%debug( &
      "applying left natural boundary conditions for " &
      // matrix%get_label() // " matrix" &
    )
    call fetch_boundary_quadblock( &
      x=grid%base_grid(1), &
      x0=grid%base_grid(1), &
      x1=grid%base_grid(2), &
      weight=-1.0_dp, &  ! -1 since we evaluate boundaries as Bounds[x1] - Bounds[x0]
      matrix=matrix, &
      settings=settings, &
      grid=grid, &
      background=background, &
      physics=physics, &
      quadblock=quadblock &
    )
    ! add quadblock to left edge
    dim_quadblock = settings%dims%get_dim_quadblock()
    do j = 1, dim_quadblock
      do i = 1, dim_quadblock
        call matrix%add_element(row=i, column=j, element=quadblock(i, j))
      end do
    end do
    deallocate(quadblock)
  end subroutine apply_natural_boundaries_left


  subroutine apply_natural_boundaries_right(matrix, settings, grid, background, physics)
    type(matrix_t), intent(inout) :: matrix
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics

    complex(dp), allocatable :: quadblock(:, :)
    integer :: i, j, dim_quadblock, ishift

    call logger%debug( &
      "applying right natural boundary conditions for " &
      // matrix%get_label() // " matrix" &
    )
    call fetch_boundary_quadblock( &
      x=grid%base_grid(settings%grid%get_gridpts()), &
      x0=grid%base_grid(settings%grid%get_gridpts() - 1), &
      x1=grid%base_grid(settings%grid%get_gridpts()), &
      weight=1.0_dp, &
      matrix=matrix, &
      settings=settings, &
      grid=grid, &
      background=background, &
      physics=physics, &
      quadblock=quadblock &
    )
    ! add quadblock to right edge. `ishift` represents the final index of the
    ! second-to-last quadblock. We add this to the iteration such that it starts
    ! from 1 + ishift, which is the starting index of the last quadblock.
    dim_quadblock = settings%dims%get_dim_quadblock()
    ishift = matrix%matrix_dim - dim_quadblock
    do j = 1, dim_quadblock
      do i = 1, dim_quadblock
        call matrix%add_element( &
          row=i + ishift, column=j + ishift, element=quadblock(i, j) &
        )
      end do
    end do
    deallocate(quadblock)
  end subroutine apply_natural_boundaries_right


  subroutine fetch_boundary_quadblock( &
    x, x0, x1, weight, matrix, settings, grid, background, physics, quadblock &
  )
    use mod_build_quadblock, only: add_to_quadblock

    real(dp), intent(in) :: x, x0, x1, weight
    type(matrix_t), intent(in) :: matrix
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics
    complex(dp), intent(out), allocatable :: quadblock(:, :)

    type(matrix_elements_t) :: elements
    integer :: dim_quadblock

    elements = new_matrix_elements(settings%state_vector)
    if (matrix%get_label() == "A") then
      call add_natural_regular_terms(x, elements, settings, grid, background)
      call add_natural_flow_terms(x, elements, settings, grid, background)
      call add_natural_resistive_terms(x, elements, settings, grid, background, physics)
      call add_natural_conduction_terms( &
        x, elements, settings, grid, background, physics &
      )
      call add_natural_viscosity_terms(x, elements, settings, grid, background)
      call add_natural_hall_terms(x, elements, settings, grid, background, physics)
    else if (matrix%get_label() == "B") then
      call add_natural_hall_Bterms(x, elements, settings, grid, background, physics)
    end if
    dim_quadblock = settings%dims%get_dim_quadblock()
    allocate(quadblock(dim_quadblock, dim_quadblock), source=(0.0_dp, 0.0_dp))
    call add_to_quadblock(quadblock, elements, x, x0, x1, weight, settings%dims)

    call elements%delete()
  end subroutine fetch_boundary_quadblock

end module mod_natural_boundaries
