module mod_suite_utils
  use mod_global_variables, only: dp
  use mod_settings, only: settings_t, new_settings
  use mod_background, only: background_t, new_background
  use mod_physics, only: physics_t, new_physics
  use mod_grid, only: grid_t, new_grid
  use mod_function_utils, only: from_function
  use mod_logging, only: logger
  implicit none

  real(dp), parameter :: TOL = 1.0d-12
  integer, parameter :: TEST_LOG_LVL = 1

contains

  subroutine set_name(name)
    character(len=*), intent(in)  :: name

    write(*, *)
    write(*, "('" // " [Test]: " // name // "')")
  end subroutine set_name

  subroutine reset_globals()
    use mod_global_variables, only: initialise_globals
    use mod_equilibrium_params, only: init_equilibrium_params, k2, k3

    call initialise_globals()
    call init_equilibrium_params()
    call logger%set_logging_level(TEST_LOG_LVL)  ! also print warnings
    k2 = 1.0d0
    k3 = 2.5d0
  end subroutine reset_globals


  function get_settings() result(settings)
    type(settings_t) :: settings
    settings = new_settings()
    call settings%set_state_vector("mhd")
  end function get_settings


  function get_background() result(background)
    type(background_t) :: background
    background = new_background()
  end function get_background


  function get_physics(settings, background) result(physics)
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background
    type(physics_t) :: physics
    physics = new_physics(settings, background)
  end function get_physics


  function get_grid(settings) result(grid)
    type(settings_t), intent(in) :: settings
    type(grid_t) :: grid
    grid = new_grid(settings)
  end function get_grid


  subroutine get_test_matrices(settings, grid, background, physics, matrix_A, matrix_B)
    use mod_matrix_structure, only: matrix_t, new_matrix
    use mod_matrix_manager, only: build_matrices

    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics
    type(matrix_t), intent(out) :: matrix_A
    type(matrix_t), intent(out) :: matrix_B

    matrix_A = new_matrix(nb_rows=settings%dims%get_dim_matrix(), label="A")
    matrix_B = new_matrix(nb_rows=settings%dims%get_dim_matrix(), label="B")
    call build_matrices(matrix_B, matrix_A, settings, grid, background, physics)
  end subroutine get_test_matrices


  real(dp) function zero(x)
    real(dp), intent(in) :: x
    zero = 0.0d0
  end function zero


  real(dp) function one(x)
    real(dp), intent(in) :: x
    one = 1.0d0
  end function one


  function create_test_grid(settings, pts, geometry, grid_start, grid_end) result(grid)
    type(settings_t), intent(inout) :: settings
    integer, intent(in), optional :: pts
    character(len=*), intent(in), optional :: geometry
    real(dp), intent(in), optional  :: grid_start, grid_end
    type(grid_t) :: grid
    real(dp) :: x_start, x_end
    integer :: gridpts
    character(:), allocatable :: grid_geometry

    if (present(pts)) then
      gridpts = pts
    else
      gridpts = settings%grid%get_gridpts()
    end if
    if (present(geometry)) then
      grid_geometry = geometry
    else
      grid_geometry = settings%grid%get_geometry()
    end if
    if (present(grid_start)) then
      x_start = grid_start
    else
      x_start = 0.0d0
    end if
    if (present(grid_end)) then
      x_end = grid_end
    else
      x_end = 1.0d0
    end if
    call settings%grid%set_geometry(grid_geometry)
    call settings%grid%set_grid_boundaries(x_start, x_end)
    call settings%grid%set_gridpts(gridpts)
    call settings%update_block_dimensions()

    grid = new_grid(settings)
    call grid%initialise()
  end function create_test_grid


  subroutine set_default_units(settings)
    type(settings_t), intent(inout) :: settings
    call settings%units%set_units_from_temperature( &
      unit_temperature=1.0d6, &
      unit_magneticfield=5.0d0, &
      unit_length=1.0d10, &
      He_abundance=0.0d0 &
    )
  end subroutine set_default_units


  function linspace(x0, x1, xvals) result(xarray)
    real(dp), intent(in)  :: x0, x1
    integer, intent(in)   :: xvals
    real(dp), allocatable :: xarray(:)
    real(dp)  :: dx
    integer   :: i

    allocate(xarray(xvals))
    dx = (x1 - x0) / (xvals - 1)
    do i = 1, xvals
      xarray(i) = x0 + (i - 1) * dx
    end do
  end function linspace


  !> Generate random number between a and b.
  function random_uniform(a, b) result(random_nb)
    real(dp), intent(in) :: a
    real(dp), intent(in) :: b
    real(dp) :: random_nb

    call random_number(random_nb)
    random_nb = (b - a) * random_nb + a
  end function random_uniform


  subroutine sort_complex_array(array)
    complex(dp), intent(inout)  :: array(:)
    complex(dp) :: temp
    integer     :: i, minidx

    ! sort array using selection sort, based on real part
    do i = 1, size(array) - 1
      minidx = minloc(real(array(i:)), 1) + i - 1
      if (real(array(i)) > real(array(minidx))) then
        temp = array(i)
        array(i) = array(minidx)
        array(minidx) = temp
      end if
    end do
  end subroutine sort_complex_array


  function create_identity_matrix(nbrows, diagonal_value) result(idmat)
    integer, intent(in) :: nbrows
    complex(dp), intent(in) :: diagonal_value
    complex(dp) :: idmat(nbrows, nbrows)
    integer :: i

    idmat = (0.0d0, 0.0d0)
    do i = 1, nbrows
      idmat(i, i) = diagonal_value
    end do
  end function create_identity_matrix


  subroutine create_banded_array(subdiags, superdiags, mat)
    integer, intent(in) :: subdiags, superdiags
    complex(dp), intent(out) :: mat(8, 8)
    integer :: i

    if (superdiags > 2 .or. subdiags > 2) then
      write(*, *) "can only create banded array up to 2 sub/super diagonals"
      stop
    end if

    mat = (0.0d0, 0.0d0)
    ! diagonal
    do i = 1, 8
      mat(i, i) = (1.0d0, 2.0d0) * i
    end do
    ! diagonals 1
    do i = 1, 7
      if (superdiags >= 1) mat(i, i + 1) = -2.0d0 * i + (3.0d0, 5.0d0)
      if (subdiags >= 1) mat(i + 1, i) =  cmplx(1.5d0, 0.5d0, kind=dp) * i
    end do
    ! dieagonals 2
    do i = 1, 6
      if (superdiags >= 2) mat(i, i + 2) = cmplx(8.0d0 - 2.0d0 * i, 1.5d0 * i, kind=dp)
      if (subdiags >= 2) mat(i + 2, i) = cmplx(1.0d0, -2.5d0 * i, kind=dp)
    end do
  end subroutine create_banded_array


  !> Checks if a given bandmatrix equals a linked-list matrix structure
  logical function matrix_equals_band(matrix, band)
    use mod_matrix_structure, only: matrix_t
    use mod_matrix_node, only: node_t
    use mod_banded_matrix, only: banded_matrix_t
    use mod_check_values, only: is_equal

    type(matrix_t), intent(in) :: matrix
    type(banded_matrix_t), intent(in) :: band
    integer :: irow, inode
    type(node_t), pointer :: current_node
    complex(dp) :: value_list, value_band

    matrix_equals_band = .true.
    ! check dimensions
    if (matrix%matrix_dim /= band%m .or. matrix%matrix_dim /= band%n) then
      write(*, *) "incompatible matrix - bandmatrix dimensions"
      write(*, *) "matrix dimensions", matrix%matrix_dim, matrix%matrix_dim
      write(*, *) "bandmatrix dimensions", band%m, band%n
      matrix_equals_band = .false.
      return
    end if
    ! check number of elements
    if (matrix%get_total_nb_elements() /= band%get_total_nb_nonzero_elements()) then
      write(*, *) "unequal number of nonzero elements in matrix and band"
      matrix_equals_band = .false.
      return
    end if
    ! iterate over linked list and retrieve element from band, check if equal
    do irow = 1, matrix%matrix_dim
      current_node => matrix%rows(irow)%head
      do inode = 1, matrix%rows(irow)%nb_elements
        value_list = current_node%get_node_element()
        value_band = band%get_element(irow, current_node%column)
        current_node => current_node%next
        if (.not. is_equal(value_list, value_band)) then
          write(*, *) "unequal matrix and bandmatrix!"
          write(*, *) "row index: ", irow
          write(*, *) "column index ", current_node%column
          write(*, *) "value bandmatrix at index: ", value_band
          write(*, *) "value linked-list matrix at index: ", value_list
          matrix_equals_band = .false.
          return
        end if
      end do
    end do
  end function matrix_equals_band


end module mod_suite_utils
