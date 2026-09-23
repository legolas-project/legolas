module mod_base_efs
  use mod_global_variables, only: dp, str_len_arr
  use mod_settings, only: settings_t
  use mod_grid, only: grid_t
  use mod_get_indices, only: get_index
  use mod_ef_assembly, only: assemble_eigenfunction, retransform_eigenfunction
  use mod_state_vector_component, only: sv_component_t
  implicit none

  private

  type, public :: base_ef_t
    type(sv_component_t), pointer :: sv_component
    character(str_len_arr) :: name
    complex(dp), allocatable :: quantities(:, :)

  contains

    procedure, public :: initialise
    procedure, public :: assemble
    procedure, public :: delete
  end type base_ef_t

contains

  subroutine initialise(this, sv_component, ef_grid_size, nb_efs)
    class(base_ef_t), intent(inout) :: this
    type(sv_component_t), intent(in), target :: sv_component
    integer, intent(in) :: ef_grid_size
    integer, intent(in) :: nb_efs

    this%name = sv_component%get_name()
    this%sv_component => sv_component
    allocate(this%quantities(ef_grid_size, nb_efs))
  end subroutine initialise


  subroutine assemble(this, settings, grid, idxs_to_assemble, right_eigenvectors)
    class(base_ef_t), intent(inout) :: this
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    integer, intent(in) :: idxs_to_assemble(:)
    complex(dp), intent(in) :: right_eigenvectors(:, :)

    integer :: i, idx, state_vector_idx
    complex(dp) :: assembled_ef(size(grid%ef_grid))

    state_vector_idx = get_index(this%name, settings%get_state_vector())
    do i = 1, size(idxs_to_assemble)
      idx = idxs_to_assemble(i)
      assembled_ef = assemble_eigenfunction( &
        settings=settings, &
        sv_component=this%sv_component, &
        grid=grid, &
        state_vector_index=state_vector_idx, &
        eigenvector=right_eigenvectors(:, idx) &
      )
      this%quantities(:, i) = retransform_eigenfunction( &
        ef_name=this%name, &
        ef_eps=grid%get_eps(grid%ef_grid), &
        eigenfunction=assembled_ef &
      )
    end do
  end subroutine assemble


  pure subroutine delete(this)
    class(base_ef_t), intent(inout) :: this
    nullify(this%sv_component)
    if(allocated(this%quantities)) deallocate(this%quantities)
  end subroutine delete

end module mod_base_efs
