module mod_eigenfunctions
  use mod_global_variables, only: dp
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_grid, only: grid_t
  use mod_base_efs, only: base_ef_t
  use mod_derived_efs, only: derived_ef_t, deallocate_derived_ef_module_variables
  use mod_derived_ef_names, only: create_and_set_derived_state_vector
  implicit none

  private

  type, public :: eigenfunctions_t
    type(settings_t), pointer, private :: settings
    type(grid_t), pointer, private :: grid
    type(background_t), pointer, private :: background
    type(base_ef_t), allocatable :: base_efs(:)
    type(derived_ef_t), allocatable :: derived_efs(:)
    logical, allocatable :: ef_written_flags(:)
    integer, allocatable :: ef_written_idxs(:)

  contains

    procedure, public :: initialise
    procedure, public :: assemble
    procedure, public :: delete

    procedure, private :: select_eigenfunctions_to_save
  end type eigenfunctions_t

  public :: new_eigenfunctions

contains

  function new_eigenfunctions(settings, grid, background) result(eigenfunctions)
    type(settings_t), target, intent(inout) :: settings
    type(grid_t), target, intent(in) :: grid
    type(background_t), target, intent(in) :: background
    type(eigenfunctions_t) :: eigenfunctions
    eigenfunctions%settings => settings
    eigenfunctions%grid => grid
    eigenfunctions%background => background
  end function new_eigenfunctions


  subroutine initialise(this, omega)
    class(eigenfunctions_t), intent(inout) :: this
    complex(dp), intent(in) :: omega(:)
    character(len=:), allocatable :: derived_state_vector(:)
    integer :: i, nb_efs

    call this%select_eigenfunctions_to_save(omega)
    nb_efs = size(this%ef_written_idxs)

    allocate(this%base_efs(size(this%settings%state_vector%components)))
    do i = 1, size(this%base_efs)
      call this%base_efs(i)%initialise( &
        sv_component=this%settings%state_vector%components(i)%ptr, &
        ef_grid_size=size(this%grid%ef_grid), &
        nb_efs=nb_efs &
      )
    end do

    if (.not. this%settings%io%write_derived_eigenfunctions) return

    derived_state_vector = create_and_set_derived_state_vector( &
      this%settings, this%background &
    )
    allocate(this%derived_efs(size(derived_state_vector)))
    do i = 1, size(this%derived_efs)
      call this%derived_efs(i)%initialise( &
        name=derived_state_vector(i), &
        ef_grid_size=size(this%grid%ef_grid), &
        nb_efs=nb_efs &
      )
    end do
    deallocate(derived_state_vector)
  end subroutine initialise


  subroutine assemble(this, right_eigenvectors)
    class(eigenfunctions_t), intent(inout) :: this
    complex(dp), intent(in) :: right_eigenvectors(:, :)
    integer :: i
    do i = 1, size(this%base_efs)
      call this%base_efs(i)%assemble( &
        settings=this%settings, &
        grid=this%grid, &
        idxs_to_assemble=this%ef_written_idxs, &
        right_eigenvectors=right_eigenvectors &
      )
    end do

    if (.not. this%settings%io%write_derived_eigenfunctions) return
    do i = 1, size(this%derived_efs)
      call this%derived_efs(i)%assemble( &
        settings=this%settings, &
        grid=this%grid, &
        background=this%background, &
        idxs_to_assemble=this%ef_written_idxs, &
        right_eigenvectors=right_eigenvectors &
      )
    end do
    call deallocate_derived_ef_module_variables()
  end subroutine assemble


  pure subroutine delete(this)
    class(eigenfunctions_t), intent(inout) :: this
    integer :: i
    if (allocated(this%base_efs)) then
      do i = 1, size(this%base_efs)
        call this%base_efs(i)%delete()
      end do
      deallocate(this%base_efs)
    end if
    if (allocated(this%derived_efs)) then
      do i = 1, size(this%derived_efs)
        call this%derived_efs(i)%delete()
      end do
      deallocate(this%derived_efs)
    end if
    if (allocated(this%ef_written_flags)) deallocate(this%ef_written_flags)
    if (allocated(this%ef_written_idxs)) deallocate(this%ef_written_idxs)
    nullify(this%settings)
    nullify(this%background)
    nullify(this%grid)
  end subroutine delete


  pure subroutine select_eigenfunctions_to_save(this, omega)
    class(eigenfunctions_t), intent(inout) :: this
    complex(dp), intent(in) :: omega(:)
    integer :: i

    allocate(this%ef_written_flags(size(omega)))
    if (this%settings%io%write_ef_subset) then
      this%ef_written_flags = eigenvalue_is_inside_subset_radius( &
        eigenvalue=omega, &
        radius=this%settings%io%ef_subset_radius, &
        center=this%settings%io%ef_subset_center &
      )
    else
      this%ef_written_flags = .true.
    end if
    ! extract indices of those eigenvalues that have their eigenfunctions written
    allocate(this%ef_written_idxs(count(this%ef_written_flags)))
    this%ef_written_idxs = pack([(i, i=1, size(omega))], this%ef_written_flags)
  end subroutine select_eigenfunctions_to_save


  elemental logical function eigenvalue_is_inside_subset_radius( &
    eigenvalue, radius, center &
  )
    complex(dp), intent(in) :: eigenvalue
    real(dp), intent(in) :: radius
    complex(dp), intent(in) :: center
    real(dp) :: distance_from_subset_center

    distance_from_subset_center = sqrt( &
      (aimag(eigenvalue) - aimag(center)) ** 2 &
      + (real(eigenvalue) - real(center)) ** 2 &
    )
    eigenvalue_is_inside_subset_radius = ( &
      distance_from_subset_center <= radius &
    )
  end function eigenvalue_is_inside_subset_radius

end module mod_eigenfunctions
