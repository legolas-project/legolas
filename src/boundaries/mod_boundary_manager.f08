module mod_boundary_manager
  use mod_global_variables, only: dp
  use mod_logging, only: logger, str
  use mod_matrix_structure, only: matrix_t
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_physics, only: physics_t
  use mod_grid, only: grid_t
  implicit none

  private

  public :: apply_boundary_conditions

contains

  subroutine apply_boundary_conditions( &
    matrix_A, matrix_B, settings, grid, background, physics &
  )
    use mod_natural_boundaries, only: apply_natural_boundaries_left, &
      apply_natural_boundaries_right
    use mod_essential_boundaries, only: apply_essential_boundaries_left, &
      apply_essential_boundaries_right

    !> the A-matrix with boundary conditions imposed on exit
    type(matrix_t), intent(inout) :: matrix_A
    !> the B-matrix with boundary conditions imposed on exit
    type(matrix_t), intent(inout) :: matrix_B
    !> the settings object
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics

    ! handle left side boundary conditions B-matrix
    call apply_natural_boundaries_left(matrix_B, settings, grid, background, physics)
    call apply_essential_boundaries_left(matrix_B, settings)
    ! handle left side boundary conditions A-matrix
    call apply_natural_boundaries_left(matrix_A, settings, grid, background, physics)
    call apply_essential_boundaries_left(matrix_A, settings)
    ! handle right side boundary conditions B-matrix
    call apply_natural_boundaries_right(matrix_B, settings, grid, background, physics)
    call apply_essential_boundaries_right(matrix_B, settings)
    ! handle right side boundary conditions A-matrix
    call apply_natural_boundaries_right(matrix_A, settings, grid, background, physics)
    call apply_essential_boundaries_right(matrix_A, settings)
  end subroutine apply_boundary_conditions
end module mod_boundary_manager
