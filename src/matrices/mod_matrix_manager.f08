module mod_matrix_manager
  use mod_global_variables, only: dp, ir, ic
  use mod_equilibrium_params, only: k2, k3
  use mod_logging, only: logger, str
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_physics, only: physics_t
  use mod_grid, only: grid_t
  use mod_matrix_elements, only: matrix_elements_t
  use mod_state_vector, only: sv_rho1, sv_v1, sv_v2, sv_v3, sv_T1, sv_a1, sv_a2, sv_a3
  implicit none

  interface
    module subroutine add_bmatrix_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_bmatrix_terms

    module subroutine add_regular_matrix_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_regular_matrix_terms

    module subroutine add_flow_matrix_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_flow_matrix_terms

    module subroutine add_resistive_matrix_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_resistive_matrix_terms

    module subroutine add_heatloss_matrix_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_heatloss_matrix_terms

    module subroutine add_conduction_matrix_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_conduction_matrix_terms

    module subroutine add_viscosity_matrix_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_viscosity_matrix_terms

    module subroutine add_hall_matrix_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_hall_matrix_terms

    module subroutine add_hall_bmatrix_terms( &
      x, elements, settings, grid, background, physics &
    )
      real(dp), intent(in) :: x
      type(matrix_elements_t), intent(inout) :: elements
      type(settings_t), intent(in) :: settings
      type(grid_t), intent(in) :: grid
      type(background_t), intent(in) :: background
      type(physics_t), intent(in) :: physics
    end subroutine add_hall_bmatrix_terms
  end interface

  private

  public  :: build_matrices

contains

  subroutine build_matrices(matrix_B, matrix_A, settings, grid, background, physics)
    use mod_global_variables, only: n_gauss, gaussian_weights
    use mod_matrix_structure, only: matrix_t
    use mod_build_quadblock, only: add_to_quadblock
    use mod_boundary_manager, only: apply_boundary_conditions
    use mod_matrix_elements, only: new_matrix_elements

    !> the B-matrix
    type(matrix_t), intent(inout) :: matrix_B
    !> the A-matrix
    type(matrix_t), intent(inout) :: matrix_A
    !> the settings object
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics

    !> quadblock for the A-matrix
    complex(dp), allocatable :: quadblock_A(:, :)
    !> quadblock for the B-matrix
    complex(dp), allocatable :: quadblock_B(:, :)
    !> left side of current interval
    real(dp)  :: x_left
    !> right side of current interval
    real(dp)  :: x_right
    !> current position in the Gaussian grid
    real(dp)  :: x_gauss
    !> current weight
    real(dp)  :: weight

    type(matrix_elements_t) :: elements_A, elements_B

    integer :: i, j, k, l, idx1, idx2
    integer :: quadblock_idx, gauss_idx, dim_quadblock

    logical :: has_flow, has_resistivity, has_cooling, has_heating, has_conduction
    logical :: has_viscosity, has_hall

    has_flow = settings%physics%flow%is_enabled()
    has_resistivity = settings%physics%resistivity%is_enabled()
    has_cooling = settings%physics%cooling%is_enabled()
    has_heating = settings%physics%heating%is_enabled()
    has_conduction = settings%physics%conduction%is_enabled()
    has_viscosity = settings%physics%viscosity%is_enabled()
    has_hall = settings%physics%hall%is_enabled()

    dim_quadblock = settings%dims%get_dim_quadblock()
    allocate(quadblock_A(dim_quadblock, dim_quadblock))
    allocate(quadblock_B, mold=quadblock_A)
    ! used to shift the quadblock along the main diagonal
    quadblock_idx = 0

    do i = 1, settings%grid%get_gridpts() - 1
      ! reset quadblocks
      quadblock_A = (0.0d0, 0.0d0)
      quadblock_B = (0.0d0, 0.0d0)

      ! get interval boundaries
      x_left = grid%base_grid(i)
      x_right = grid%base_grid(i + 1)

      ! loop over Gaussian points to calculate integral
      do j = 1, n_gauss
        ! current grid index of Gaussian grid
        gauss_idx = (i - 1) * n_gauss + j
        x_gauss = grid%gaussian_grid(gauss_idx)
        weight = gaussian_weights(j)

        elements_A = new_matrix_elements(settings%state_vector)
        elements_B = new_matrix_elements(settings%state_vector)

        ! get matrix elements
        call add_bmatrix_terms( &
          x_gauss, elements_B, settings, grid, background, physics &
        )
        call add_regular_matrix_terms( &
          x_gauss, elements_A, settings, grid, background, physics &
        )
        if (has_flow) call add_flow_matrix_terms( &
          x_gauss, elements_A, settings, grid, background, physics &
        )
        if (has_resistivity) call add_resistive_matrix_terms( &
          x_gauss, elements_A, settings, grid, background, physics &
        )
        if (has_cooling .or. has_heating) call add_heatloss_matrix_terms( &
          x_gauss, elements_A, settings, grid, background, physics &
        )
        if (has_conduction) call add_conduction_matrix_terms( &
          x_gauss, elements_A, settings, grid, background, physics &
        )
        if (has_viscosity) call add_viscosity_matrix_terms( &
          x_gauss, elements_A, settings, grid, background, physics &
        )
        if (has_hall) then
          call add_hall_matrix_terms( &
            x_gauss, elements_A, settings, grid, background, physics &
        )
          call add_hall_bmatrix_terms( &
            x_gauss, elements_B, settings, grid, background, physics &
        )
        end if

        call add_to_quadblock( &
          quadblock_A, elements_A, x_gauss, x_left, x_right, weight, settings%dims &
        )
        call elements_A%delete()
        call add_to_quadblock( &
          quadblock_B, elements_B, x_gauss, x_left, x_right, weight, settings%dims &
        )
        call elements_B%delete()
      end do

      ! dx from integral
      quadblock_B = quadblock_B * (x_right - x_left)
      quadblock_A = quadblock_A * (x_right - x_left)

      ! fill matrices
      !> @note
      !!     The quadblock is shifted along the main (tri)diagonal.
      !!     We add `dim_subblock` instead of `dim_quadblock` to the indices,
      !!     due to overlapping of the bottom-right part of the quadblock with the
      !!     top-left part of the next grid interval.
      !! @endnote
      do k = 1, dim_quadblock
        do l = 1, dim_quadblock
          idx1 = k + quadblock_idx
          idx2 = l + quadblock_idx
          call matrix_B%add_element(row=idx1, column=idx2, element=quadblock_B(k, l))
          call matrix_A%add_element(row=idx1, column=idx2, element=quadblock_A(k, l))
        end do
      end do
      quadblock_idx = quadblock_idx + settings%dims%get_dim_subblock()
    end do

    deallocate(quadblock_A, quadblock_B)
    call apply_boundary_conditions( &
      matrix_A, matrix_B, settings, grid, background, physics &
    )
  end subroutine build_matrices

end module mod_matrix_manager
