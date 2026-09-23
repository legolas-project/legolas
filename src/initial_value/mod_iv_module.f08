module mod_iv_module
  use mod_global_variables, only: dp
  use mod_logging, only: logger, str
  use mod_matrix_structure, only: matrix_t
  use mod_settings, only: settings_t
  use mod_grid, only: grid_t
  use mod_iv_state_vector, only: iv_state_vector_t, init_and_bind
  use mod_iv_solver, only: solve
  use mod_iv_initial_conditions, only: initial_conditions_t
  implicit none

  private

  type, public :: iv_module_t
    logical, private :: is_initialised
    type(settings_t), pointer, private :: settings
    type(grid_t), pointer, private :: grid
    type(iv_state_vector_t), public :: state_vec
    type(initial_conditions_t), pointer, private :: iv_initial_conditions

    real, allocatable :: iv_grid(:) 

    complex(dp), allocatable :: snapshots(:,:)
    real(dp), allocatable :: snap_times(:)
  contains

  procedure, public :: initialise
  procedure, public :: solve_ivp

  procedure, public :: postprocess_snapshot

  end type iv_module_t

  public :: new_iv_module

contains

  function new_iv_module(settings, grid) result(iv_module)
    type(settings_t), target, intent(in) :: settings
    type(grid_t), target, intent(in) :: grid

    type(iv_module_t) :: iv_module

    iv_module%settings => settings
    iv_module%grid => grid
    iv_module%is_initialised = .false.
  end function new_iv_module


  subroutine initialise(self, initial_conditions)
    class(iv_module_t), intent(inout) :: self
    class(initial_conditions_t), intent(inout) :: initial_conditions

    if (self%is_initialised) return

    ! Initialise and setup the state vector
    self%state_vec = init_and_bind(self%settings%state_vector)
    call self%state_vec%initialise_components(self%settings%get_physics_type(), initial_conditions)

    ! Assemble the initial value array in block format
    call self%state_vec%assemble_iv_array(self%settings%grid%get_gridpts(), self%grid%base_grid)

    ! Setup snapshots array
    allocate(self%snapshots(self%state_vec%stride * self%settings%grid%get_gridpts(), &
     self%settings%iv%get_n_snapshots()))

    allocate(self%snap_times(self%settings%iv%get_n_snapshots()))

    self%is_initialised = .true.
  end subroutine initialise


  subroutine solve_ivp(self, matrix_A, matrix_B)
    class(iv_module_t), intent(inout) :: self
    type(matrix_t), intent(in) :: matrix_A
    type(matrix_t), intent(in) :: matrix_B

    call solve(matrix_A, matrix_B, self%state_vec%x0, self%settings, self%snapshots, self%snap_times)
    
  end subroutine solve_ivp

  !> Postprocess a given snapshot.
  !! Reassemble the profile and sotre in each component
  subroutine postprocess_snapshot(self, i_snap)
    class(iv_module_t), intent(inout) :: self
    integer, intent(in) :: i_snap

    if (i_snap > self%settings%iv%get_n_snapshots()) then
      call logger%error("requested snapshot is out of bounds")
    end if

    ! 1. Update the state vector
    self%state_vec%x0 = self%snapshots(:, i_snap)

    ! 2. Re-compute c1, c2 from x0
    call self%state_vec%disassemble_iv_array(self%settings%grid%get_gridpts())

    ! 3. Reconstruct
    call self%state_vec%reassemble_from_block( &
        self%settings, self%grid)

  end subroutine postprocess_snapshot


end module mod_iv_module