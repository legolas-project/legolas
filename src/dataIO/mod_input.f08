! =============================================================================
!> Module to handle parfile reading.
!! Contains subroutines to retrieve the parfile based on the commandline arguments
!! and to read the parfile, setting the global variables.
module mod_input
  use mod_global_variables, only: dp, str_len, str_len_arr, NaN
  use mod_logging, only: logger
  use mod_settings, only: settings_t
  implicit none

  private

  integer, parameter :: unit_par = 101

  public :: read_parfile
  public :: get_parfile

contains


  subroutine read_parfile(parfile, settings)
    character(len=*), intent(in) :: parfile
    type(settings_t), intent(inout) :: settings

    if (parfile == "") then
      call logger%error("no parfile supplied!")
      return
    end if

    open(unit_par, file=trim(parfile), status="old")
      ! rewind after reading so namelists can appear out of order
      call read_gridlist(unit_par, settings)
      rewind(unit_par)
      call read_savelist(unit_par, settings)
      rewind(unit_par)
      call read_solvelist(unit_par, settings)
      rewind(unit_par)
      call read_physicslist(unit_par, settings)
      rewind(unit_par)
      call read_equilibriumlist(unit_par, settings)
      rewind(unit_par)
      call read_paramlist(unit_par)
      rewind(unit_par)
      call read_unitlist(unit_par, settings)
    close(unit_par)

    ! make sure to update dimensions after reading in the parfile
    call settings%update_block_dimensions()
    if (settings%solvers%get_solver() == "none") call settings%io%set_all_io_to_false()
  end subroutine read_parfile


  subroutine read_gridlist(unit, settings)
    integer, intent(in) :: unit
    type(settings_t), intent(inout) :: settings
    integer :: iostat
    character(str_len) :: iomsg

    character(len=str_len) :: geometry
    integer :: gridpoints
    real(dp) :: x_start, x_end
    logical :: coaxial
    logical :: force_r0
    logical :: symmetric_grid

    namelist /gridlist/ &
      geometry, gridpoints, x_start, x_end, coaxial, force_r0, symmetric_grid

    ! defaults
    geometry = settings%grid%get_geometry()
    gridpoints = settings%grid%get_gridpts()
    x_start = settings%grid%get_grid_start()
    x_end = settings%grid%get_grid_end()
    coaxial = settings%grid%coaxial
    force_r0 = settings%grid%force_r0
    symmetric_grid = settings%grid%symmetric_grid

    read(unit, nml=gridlist, iostat=iostat, iomsg=iomsg)
    call parse_io_info(iostat, iomsg)

    call settings%grid%set_geometry(geometry)
    call settings%grid%set_gridpts(gridpoints)
    call settings%grid%set_grid_boundaries(grid_start=x_start, grid_end=x_end)
    settings%grid%coaxial = coaxial
    settings%grid%force_r0 = force_r0
    settings%grid%symmetric_grid = symmetric_grid
  end subroutine read_gridlist


  subroutine read_savelist(unit, settings)
    integer, intent(in) :: unit
    type(settings_t), intent(inout) :: settings
    integer :: iostat
    character(str_len) :: iomsg

    logical :: write_matrices, write_eigenvectors, write_residuals, write_background
    logical :: write_eigenfunctions, write_derived_eigenfunctions
    logical :: write_eigenfunction_subset
    logical :: show_results
    integer :: logging_level
    real(dp) :: eigenfunction_subset_radius
    complex(dp) :: eigenfunction_subset_center
    character(len=str_len) :: basename_datfile, output_folder

    namelist /savelist/ &
      write_matrices, write_eigenvectors, write_residuals, write_background, &
      write_eigenfunctions, write_derived_eigenfunctions, write_eigenfunction_subset, &
      show_results, basename_datfile, output_folder, logging_level, &
      eigenfunction_subset_radius, eigenfunction_subset_center

    ! get defaults
    write_matrices = settings%io%write_matrices
    write_eigenvectors = settings%io%write_eigenvectors
    write_residuals = settings%io%write_residuals
    write_background = settings%io%write_background
    write_eigenfunctions = settings%io%write_eigenfunctions
    write_derived_eigenfunctions = settings%io%write_derived_eigenfunctions
    write_eigenfunction_subset = settings%io%write_ef_subset
    show_results = settings%io%show_results
    basename_datfile = settings%io%get_basename_datfile()
    output_folder = settings%io%get_output_folder()
    logging_level = logger%get_logging_level()
    eigenfunction_subset_radius = settings%io%ef_subset_radius
    eigenfunction_subset_center = settings%io%ef_subset_center

    read(unit, nml=savelist, iostat=iostat, iomsg=iomsg)
    call parse_io_info(iostat, iomsg)

    call logger%set_logging_level(logging_level)

    settings%io%write_matrices = write_matrices
    settings%io%write_eigenvectors = write_eigenvectors
    settings%io%write_residuals = write_residuals
    settings%io%write_background = write_background
    settings%io%write_eigenfunctions = write_eigenfunctions
    settings%io%write_derived_eigenfunctions = write_derived_eigenfunctions
    settings%io%write_ef_subset = write_eigenfunction_subset
    if (write_eigenfunction_subset) call check_eigenfunction_subset_params( &
      center=eigenfunction_subset_center, radius=eigenfunction_subset_radius &
    )
    settings%io%ef_subset_radius = eigenfunction_subset_radius
    settings%io%ef_subset_center = eigenfunction_subset_center
    settings%io%show_results = show_results
    call settings%io%set_basename_datfile(basename_datfile)
    call settings%io%set_output_folder(output_folder)
  end subroutine read_savelist


  subroutine read_solvelist(unit, settings)
    integer, intent(in) :: unit
    type(settings_t), intent(inout) :: settings
    integer :: iostat
    character(str_len) :: iomsg

    character(len=str_len) :: solver, arpack_mode
    character(len=2) :: which_eigenvalues
    integer :: number_of_eigenvalues, maxiter, ncv
    real(dp) :: tolerance
    complex(dp) :: sigma

    namelist /solvelist/ &
      solver, arpack_mode, which_eigenvalues, number_of_eigenvalues, &
      maxiter, ncv, tolerance, sigma

    ! get defaults
    solver = settings%solvers%get_solver()
    arpack_mode = settings%solvers%get_arpack_mode()
    which_eigenvalues = settings%solvers%which_eigenvalues
    number_of_eigenvalues = settings%solvers%number_of_eigenvalues
    maxiter = settings%solvers%maxiter
    ncv = settings%solvers%ncv
    tolerance = settings%solvers%tolerance
    sigma = settings%solvers%sigma

    read(unit, nml=solvelist, iostat=iostat, iomsg=iomsg)
    call parse_io_info(iostat, iomsg)

    call settings%solvers%set_solver(solver)
    call settings%solvers%set_arpack_mode(arpack_mode)
    settings%solvers%number_of_eigenvalues = number_of_eigenvalues
    settings%solvers%which_eigenvalues = which_eigenvalues
    settings%solvers%maxiter = maxiter
    settings%solvers%ncv = ncv
    settings%solvers%tolerance = tolerance
    settings%solvers%sigma = sigma
  end subroutine read_solvelist


  subroutine read_physicslist(unit, settings)
    integer, intent(in) :: unit
    type(settings_t), intent(inout) :: settings
    integer :: iostat
    character(str_len) :: iomsg

    character(len=str_len) :: physics_type
    character(len=str_len_arr), allocatable :: basis_functions(:)
    logical :: flow, incompressible, radiative_cooling, heating, &
      force_thermal_balance, external_gravity, &
      parallel_conduction, perpendicular_conduction, resistivity, use_eta_dropoff, &
      viscosity, viscous_heating, hall_mhd, hall_dropoff, &
      elec_inertia, inertia_dropoff
    integer :: ncool
    character(len=str_len) :: cooling_curve
    real(dp) :: mhd_gamma
    real(dp) :: fixed_tc_para_value, fixed_tc_perp_value, fixed_resistivity_value
    real(dp) :: viscosity_value
    real(dp) :: electron_fraction
    real(dp) :: dropoff_edge_dist, dropoff_width

    namelist /physicslist/ &
      physics_type, basis_functions, mhd_gamma, flow, incompressible, &
      radiative_cooling, heating, force_thermal_balance, &
      external_gravity, parallel_conduction, perpendicular_conduction, &
      resistivity, use_eta_dropoff, viscosity, viscous_heating, hall_mhd, &
      hall_dropoff, elec_inertia, inertia_dropoff, ncool, &
      cooling_curve, fixed_resistivity_value, fixed_tc_para_value, &
      fixed_tc_perp_value, dropoff_edge_dist, dropoff_width, viscosity_value, &
      electron_fraction

    ! get defaults
    physics_type = settings%get_physics_type()
    allocate(basis_functions(16))
    basis_functions = ""
    mhd_gamma = settings%physics%get_gamma()
    incompressible = settings%physics%is_incompressible
    dropoff_edge_dist = settings%physics%dropoff_edge_dist
    dropoff_width = settings%physics%dropoff_width

    flow = settings%physics%flow%is_enabled()

    radiative_cooling = settings%physics%cooling%is_enabled()
    heating = settings%physics%heating%is_enabled()
    force_thermal_balance = settings%physics%heating%force_thermal_balance
    ncool = settings%physics%cooling%get_interpolation_points()
    cooling_curve = settings%physics%cooling%get_cooling_curve()

    external_gravity = settings%physics%gravity%is_enabled()

    parallel_conduction = settings%physics%conduction%has_parallel_conduction()
    perpendicular_conduction = ( &
      settings%physics%conduction%has_perpendicular_conduction() &
    )
    fixed_tc_para_value = settings%physics%conduction%get_fixed_tc_para()
    fixed_tc_perp_value = settings%physics%conduction%get_fixed_tc_perp()

    resistivity = settings%physics%resistivity%is_enabled()
    fixed_resistivity_value = settings%physics%resistivity%get_fixed_resistivity()
    use_eta_dropoff = settings%physics%resistivity%use_dropoff

    viscosity = settings%physics%viscosity%is_enabled()
    viscosity_value = settings%physics%viscosity%get_viscosity_value()
    viscous_heating = settings%physics%viscosity%has_viscous_heating()

    hall_mhd = settings%physics%hall%is_enabled()
    hall_dropoff = settings%physics%hall%use_dropoff
    elec_inertia = settings%physics%hall%has_electron_inertia()
    inertia_dropoff = settings%physics%hall%use_inertia_dropoff
    electron_fraction = settings%physics%hall%get_electron_fraction()

    read(unit, nml=physicslist, iostat=iostat, iomsg=iomsg)
    call parse_io_info(iostat, iomsg)


    call settings%set_state_vector(physics_type)
    basis_functions = pack(basis_functions, basis_functions /= "")
    call settings%state_vector%set_basis_functions(basis_functions)
    call settings%physics%set_gamma(mhd_gamma)
    if (incompressible) call settings%physics%set_incompressible()
    if (flow) call settings%physics%flow%enable()
    if (radiative_cooling) call settings%physics%enable_cooling(cooling_curve, ncool)
    settings%physics%heating%force_thermal_balance = force_thermal_balance
    if (heating) call settings%physics%enable_heating()
    if (external_gravity) call settings%physics%enable_gravity()
    if (parallel_conduction) then
      call settings%physics%enable_parallel_conduction(fixed_tc_para_value)
    end if
    if (perpendicular_conduction) then
      call settings%physics%enable_perpendicular_conduction(fixed_tc_perp_value)
    end if
    if (resistivity) call settings%physics%enable_resistivity(fixed_resistivity_value)
    settings%physics%resistivity%use_dropoff = use_eta_dropoff
    if (viscosity) call settings%physics%enable_viscosity( &
      viscosity_value, viscous_heating &
    )
    if (hall_mhd) call settings%physics%enable_hall(elec_inertia, electron_fraction)
    settings%physics%hall%use_dropoff = hall_dropoff
    settings%physics%hall%use_inertia_dropoff = inertia_dropoff
    settings%physics%dropoff_edge_dist = dropoff_edge_dist
    settings%physics%dropoff_width = dropoff_width
  end subroutine read_physicslist


  subroutine read_equilibriumlist(unit, settings)
    integer, intent(in) :: unit
    type(settings_t), intent(inout) :: settings
    integer :: iostat
    character(str_len) :: iomsg

    character(len=str_len) :: equilibrium_type, boundary_type
    logical :: use_defaults

    namelist /equilibriumlist/ equilibrium_type, boundary_type, use_defaults

    ! get defaults
    equilibrium_type = settings%equilibrium%get_equilibrium_type()
    boundary_type = settings%equilibrium%get_boundary_type()
    use_defaults = settings%equilibrium%use_defaults

    read(unit, nml=equilibriumlist, iostat=iostat, iomsg=iomsg)
    call parse_io_info(iostat, iomsg)

    call settings%equilibrium%set_equilibrium_type(equilibrium_type)
    call settings%equilibrium%set_boundary_type(boundary_type)
    settings%equilibrium%use_defaults = use_defaults
  end subroutine read_equilibriumlist


  subroutine read_paramlist(unit)
    use mod_equilibrium_params

    integer, intent(in) :: unit
    integer :: iostat
    character(str_len) :: iomsg

    namelist /paramlist/  &
      k2, k3, cte_rho0, cte_T0, cte_B01, cte_B02, cte_B03, cte_v02, cte_v03, &
      cte_p0, p1, p2, p3, p4, p5, p6, p7, p8, &
      alpha, beta, delta, theta, tau, lambda, nu, &
      r0, rc, rj, Bth0, Bz0, V, j0, g, eq_bool, input_file, n_input

    call init_equilibrium_params()
    read(unit, nml=paramlist, iostat=iostat, iomsg=iomsg)
    call parse_io_info(iostat, iomsg)
  end subroutine read_paramlist


  subroutine read_unitlist(unit, settings)
    use mod_check_values, only: is_NaN

    integer, intent(in) :: unit
    type(settings_t), intent(inout) :: settings
    integer :: iostat
    character(str_len) :: iomsg

    real(dp) :: unit_numberdensity
    real(dp) :: unit_density, unit_temperature, unit_magneticfield, unit_length
    real(dp) :: mean_molecular_weight

    namelist /unitslist/ &
      unit_density, unit_numberdensity, unit_temperature, unit_magneticfield, &
      unit_length, mean_molecular_weight

    ! defaults
    unit_numberdensity = NaN
    unit_density = NaN
    unit_temperature = settings%units%get_unit_temperature()
    unit_magneticfield = settings%units%get_unit_magneticfield()
    unit_length = settings%units%get_unit_length()
    mean_molecular_weight = settings%units%get_mean_molecular_weight()

    read(unit, nml=unitslist, iostat=iostat, iomsg=iomsg)
    call parse_io_info(iostat, iomsg)

    if (.not. is_NaN(unit_numberdensity)) then
      call settings%units%set_units_from_numberdensity( &
        unit_length=unit_length, &
        unit_temperature=unit_temperature, &
        unit_numberdensity=unit_numberdensity, &
        mean_molecular_weight=mean_molecular_weight &
      )
    else if (.not. is_NaN(unit_density)) then
      call settings%units%set_units_from_density( &
        unit_length=unit_length, &
        unit_magneticfield=unit_magneticfield, &
        unit_density=unit_density, &
        mean_molecular_weight=mean_molecular_weight &
      )
    else
      call settings%units%set_units_from_temperature( &
        unit_length=unit_length, &
        unit_magneticfield=unit_magneticfield, &
        unit_temperature=unit_temperature, &
        mean_molecular_weight=mean_molecular_weight &
      )
    end if
  end subroutine read_unitlist


  !> Called when the eigenfunction subset selection is enabled, this checks if the
  !! global variables are properly set.
  subroutine check_eigenfunction_subset_params(center, radius)
    use mod_check_values, only: is_NaN

    complex(dp), intent(in) :: center
    real(dp), intent(in) :: radius

    if (is_NaN(center)) then
      call logger%error("eigenfunction_subset_center must be set!")
      return
    end if
    if (is_NaN(radius)) then
      call logger%error("eigenfunction_subset_radius must be set!")
      return
    end if
  end subroutine check_eigenfunction_subset_params


  subroutine parse_io_info(iostat, iomsg)
    integer, intent(in) :: iostat
    character(len=*), intent(in) :: iomsg

    if (iostat > 0) call logger%error(trim(adjustl(iomsg)))
  end subroutine parse_io_info


  ! LCOV_EXCL_START <this routine is excluded from coverage>
  !> Parses the command line arguments and retrieves the parfile passed.
  !! @warning Throws an error if
  !!
  !! - command line arguments can not be parsed.
  !! - the parfile is not found. @endwarning
  !! @note If no parfile is passed, the code uses a default configuration. @endnote
  subroutine get_parfile(filename_par)
    !> the name of the parfile
    character(len=*), intent(out) :: filename_par

    integer :: num_args, i
    character(len=5*str_len), allocatable :: args(:)
    logical :: file_exists

    num_args = command_argument_count()
    allocate(args(num_args))
    do i = 1, num_args
      call get_command_argument(i, args(i))
    end do

    filename_par = ""

    do i = 1, num_args, 2
      select case(args(i))
      case('-i')
        filename_par = args(i+1)
      case default
        call logger%error("unable to read in command line arguments")
      end select
    end do

    ! If no parfile supplied, flag an error
    if (filename_par == "") then
      call logger%error( &
        "no parfile supplied, please provide the file path with the -i flag" &
      )
    end if

    !! Check if supplied file exists
    inquire(file=trim(filename_par), exist=file_exists)
    if (.not. file_exists) then
      call logger%error(("parfile not found: " // trim(filename_par)))
    end if
  end subroutine get_parfile
  ! LCOV_EXCL_STOP

end module mod_input
