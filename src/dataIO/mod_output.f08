module mod_output
  use mod_global_variables, only: dp, str_len, str_len_arr
  use mod_settings, only: settings_t
  use mod_grid, only: grid_t
  use mod_background, only: background_t
  use mod_physics, only: physics_t
  use mod_matrix_structure, only: matrix_t
  use mod_logging, only: logger, str
  use mod_eigenfunctions, only: eigenfunctions_t
  use mod_iv_module, only: iv_module_t
  implicit none

  private

  ! IO units -- do not use 0/5/6/7, these are system-reserved
  !> filehandler IO unit for the main data file
  integer, parameter  :: dat_fh = 10
  !> filehandler IO unit for the log file
  integer, parameter  :: log_fh = 20
  !> datfile name
  character(:), allocatable :: datfile_path

  public :: datfile_path
  public :: create_datfile

contains


  subroutine open_file(file_unit, filename)
    integer, intent(in) :: file_unit
    character(len=*), intent(in)  :: filename
    open( &
      unit=file_unit, &
      file=filename, &
      access="stream", &
      status="unknown", &
      action="write" &
    )
  end subroutine open_file


  function get_datfile_path(settings, extension) result(filename)
    type(settings_t), intent(in) :: settings
    character(len=*), intent(in) :: extension
    !> the filename that is created
    character(len=:), allocatable :: filename

    filename = trim( &
      settings%io%get_output_folder() &
      // "/" &
      // settings%io%get_basename_datfile() &
      // trim(adjustl(extension)) &
    )
  end function get_datfile_path


  subroutine create_datfile( &
    settings, &
    grid, &
    background, &
    physics, &
    eigenvalues, &
    matrix_A, &
    matrix_B, &
    eigenvectors, &
    eigenfunctions, &
    iv_module &
  )
    use mod_version, only: LEGOLAS_VERSION

    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics
    complex(dp), intent(in) :: eigenvalues(:)
    type(matrix_t), intent(in) :: matrix_A
    type(matrix_t), intent(in) :: matrix_B
    complex(dp), intent(in) :: eigenvectors(:, :)
    type(eigenfunctions_t), intent(in) :: eigenfunctions
    type(iv_module_t), intent(inout) :: iv_module

    datfile_path = get_datfile_path(settings=settings, extension=".dat")
    call open_file(file_unit=dat_fh, filename=datfile_path)

    write(dat_fh) "legolas_version", LEGOLAS_VERSION
    write(dat_fh) str_len, str_len_arr

    call write_header(settings)

    write(dat_fh) size(eigenvalues), eigenvalues
    write(dat_fh) grid%base_grid, grid%gaussian_grid
    if (settings%io%write_background) then
      call write_background_data(settings, grid, background, physics)
    end if
    if (settings%io%write_eigenfunctions) then
      call write_base_eigenfunction_data(grid, eigenfunctions)
    end if
    if (settings%io%write_derived_eigenfunctions) then
      call write_derived_eigenfunction_data(settings, eigenfunctions)
    end if
    if (settings%io%write_eigenvectors) call write_eigenvector_data(eigenvectors)
    if (settings%io%write_residuals) then
      call write_residual_data(eigenvalues, matrix_A, matrix_B, eigenvectors)
    end if
    if (settings%io%write_matrices) call write_matrix_data(matrix_A, matrix_B)
    if (settings%io%write_iv_snapshots .and. settings%iv%enabled) then
      call write_iv_snapshots(settings, iv_module)
    end if

    close(dat_fh)
  end subroutine create_datfile


  subroutine write_header(settings)
    type(settings_t), intent(in) :: settings

    call write_physics_type_info(settings)
    call write_grid_info(settings)
    call write_io_info(settings)
    call write_solver_info(settings)
    call write_equilibrium_info(settings)
    call write_units_info(settings)
    call write_physics_info(settings)
    call write_parameters(settings)
    call write_background_names(settings)
  end subroutine write_header


  subroutine write_physics_type_info(settings)
    type(settings_t), intent(in) :: settings
    character(len=:), allocatable :: state_vector(:)
    character(len=:), allocatable :: basis_functions(:)

    allocate(state_vector, source=settings%get_state_vector())
    allocate(basis_functions, source=settings%state_vector%get_basis_functions())

    write(dat_fh) settings%get_nb_eqs()
    write(dat_fh) len(settings%get_physics_type()), settings%get_physics_type()
    write(dat_fh) len(state_vector(1)), size(state_vector), state_vector
    write(dat_fh) len(basis_functions(1)), size(basis_functions), basis_functions
    write(dat_fh) settings%dims%get_dim_integralblock()
    write(dat_fh) settings%dims%get_dim_subblock()
    write(dat_fh) settings%dims%get_dim_quadblock()
    write(dat_fh) settings%dims%get_dim_matrix()

    if (allocated(state_vector)) deallocate(state_vector)
    if (allocated(basis_functions)) deallocate(basis_functions)
  end subroutine write_physics_type_info


  subroutine write_grid_info(settings)
    use mod_global_variables, only: n_gauss, gaussian_nodes, gaussian_weights

    type(settings_t), intent(in) :: settings
    character(len=:), allocatable :: geometry

    geometry = settings%grid%get_geometry()
    write(dat_fh) len(geometry), geometry
    write(dat_fh) settings%grid%get_gridpts()
    write(dat_fh) settings%grid%get_gauss_gridpts()
    write(dat_fh) settings%grid%get_ef_gridpts()
    write(dat_fh) n_gauss, gaussian_nodes, gaussian_weights
    write(dat_fh) settings%grid%get_grid_start()
    write(dat_fh) settings%grid%get_grid_end()

    if (allocated(geometry)) deallocate(geometry)
  end subroutine write_grid_info


  subroutine write_io_info(settings)
    type(settings_t), intent(in) :: settings

    write(dat_fh) settings%io%write_matrices
    write(dat_fh) settings%io%write_eigenvectors
    write(dat_fh) settings%io%write_residuals
    write(dat_fh) settings%io%write_eigenfunctions
    write(dat_fh) settings%io%write_derived_eigenfunctions
    write(dat_fh) settings%io%write_ef_subset
    write(dat_fh) settings%io%ef_subset_radius
    write(dat_fh) settings%io%ef_subset_center
    write(dat_fh) settings%io%write_iv_snapshots .and. settings%iv%enabled
  end subroutine write_io_info


  subroutine write_solver_info(settings)
    type(settings_t), intent(in) :: settings
    character(len=:), allocatable :: solver, arpack_mode

    allocate(solver, source=settings%solvers%get_solver())
    allocate(arpack_mode, source=settings%solvers%get_arpack_mode())
    write(dat_fh) len(solver), solver
    write(dat_fh) len(arpack_mode), arpack_mode
    write(dat_fh) settings%solvers%number_of_eigenvalues
    write(dat_fh) len(settings%solvers%which_eigenvalues), &
      settings%solvers%which_eigenvalues
    write(dat_fh) settings%solvers%ncv
    write(dat_fh) settings%solvers%maxiter
    write(dat_fh) settings%solvers%sigma
    write(dat_fh) settings%solvers%tolerance

    if (allocated(solver)) deallocate(solver)
    if (allocated(arpack_mode)) deallocate(arpack_mode)
  end subroutine write_solver_info


  subroutine write_equilibrium_info(settings)
    type(settings_t), intent(in) :: settings
    character(len=:), allocatable :: equilibrium_type
    character(len=:), allocatable :: boundary_type

    allocate(equilibrium_type, source=settings%equilibrium%get_equilibrium_type())
    allocate(boundary_type, source=settings%equilibrium%get_boundary_type())
    write(dat_fh) len(equilibrium_type), equilibrium_type
    write(dat_fh) len(boundary_type), boundary_type

    if (allocated(equilibrium_type)) deallocate(equilibrium_type)
    if (allocated(boundary_type)) deallocate(boundary_type)
  end subroutine write_equilibrium_info


  subroutine write_units_info(settings)
    type(settings_t), intent(in) :: settings
    ! number of units written to the datfile
    integer, parameter :: n_units = 15

    write(dat_fh) n_units
    write(dat_fh) settings%units%in_cgs()
    write(dat_fh) len("unit_length"), "unit_length", settings%units%get_unit_length()
    write(dat_fh) len("unit_time"), "unit_time", settings%units%get_unit_time()
    write(dat_fh) len("unit_density"), "unit_density", settings%units%get_unit_density()
    write(dat_fh) len("unit_velocity"), "unit_velocity", &
      settings%units%get_unit_velocity()
    write(dat_fh) len("unit_temperature"), "unit_temperature", &
      settings%units%get_unit_temperature()
    write(dat_fh) len("unit_pressure"), "unit_pressure", &
      settings%units%get_unit_pressure()
    write(dat_fh) len("unit_magneticfield"), "unit_magneticfield", &
      settings%units%get_unit_magneticfield()
    write(dat_fh) len("unit_numberdensity"), "unit_numberdensity", &
      settings%units%get_unit_numberdensity()
    write(dat_fh) len("unit_mass"), "unit_mass", settings%units%get_unit_mass()
    write(dat_fh) len("He_abundance"), "He_abundance", &
      settings%units%get_He_abundance()
    write(dat_fh) len("units_a"), "units_a", &
      settings%units%get_units_parameter_a()
    write(dat_fh) len("units_b"), "units_b", &
      settings%units%get_units_parameter_b()
    write(dat_fh) len("unit_resistivity"), "unit_resistivity", &
      settings%units%get_unit_resistivity()
    write(dat_fh) len("unit_lambdaT"), "unit_lambdaT", &
      settings%units%get_unit_lambdaT()
    write(dat_fh) len("unit_conduction"), "unit_conduction", &
      settings%units%get_unit_conduction()
  end subroutine write_units_info


  subroutine write_physics_info(settings)
    type(settings_t), intent(in) :: settings
    character(len=:), allocatable :: cooling_curve

    allocate(cooling_curve, source=settings%physics%cooling%get_cooling_curve())

    write(dat_fh) settings%physics%get_gamma()
    write(dat_fh) settings%physics%is_incompressible
    ! flow info
    write(dat_fh) settings%physics%flow%is_enabled()
    ! cooling info
    write(dat_fh) settings%physics%cooling%is_enabled()
    write(dat_fh) len(cooling_curve), cooling_curve
    write(dat_fh) settings%physics%cooling%get_interpolation_points()
    ! gravity info
    write(dat_fh) settings%physics%gravity%is_enabled()
    ! resistivity info
    write(dat_fh) settings%physics%resistivity%is_enabled()
    write(dat_fh) settings%physics%resistivity%has_fixed_resistivity()
    ! viscosity info
    write(dat_fh) settings%physics%viscosity%is_enabled()
    write(dat_fh) settings%physics%viscosity%has_viscous_heating()
    ! conduction info
    write(dat_fh) settings%physics%conduction%is_enabled()
    write(dat_fh) settings%physics%conduction%has_parallel_conduction()
    write(dat_fh) settings%physics%conduction%has_fixed_tc_para()
    write(dat_fh) settings%physics%conduction%has_perpendicular_conduction()
    write(dat_fh) settings%physics%conduction%has_fixed_tc_perp()
    ! Hall info
    write(dat_fh) settings%physics%hall%is_enabled()
    write(dat_fh) settings%physics%hall%is_using_substitution()
    write(dat_fh) settings%physics%hall%has_electron_inertia()

    if (allocated(cooling_curve)) deallocate(cooling_curve)
  end subroutine write_physics_info


  subroutine write_parameters(settings)
    use mod_equilibrium_params

    type(settings_t), intent(in) :: settings
    integer, parameter :: nb_params = 35
    character(len=str_len_arr) :: parameter_names(nb_params)
    real(dp) :: parameters(nb_params)
    integer :: i

    parameter_names = [ &
      character(len=str_len_arr) :: &
      "k2", "k3", "cte_rho0", "cte_T0", "cte_p0", &
      "cte_B01", "cte_B02", "cte_B03", "Bth0", "Bz0", &
      "cte_v02", "cte_v03", &
      "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", &
      "alpha", "beta", "delta", "theta", "tau", "lambda", "nu", &
      "r0", "rc", "rj", "V", "j0", "g", &
      "electronfraction", &
      "viscosity_value" &
    ]
    parameters = [ &
      k2, k3, cte_rho0, cte_T0, cte_p0, &
      cte_B01, cte_B02, cte_B03, Bth0, Bz0, &
      cte_v02, cte_v03, &
      p1, p2, p3, p4, p5, p6, p7, p8, &
      alpha, beta, delta, theta, tau, lambda, nu, &
      r0, rc, rj, V, j0, g, &
      settings%physics%hall%get_electron_fraction(), &
      settings%physics%viscosity%get_viscosity_value() &
    ]
    write(dat_fh) nb_params, len(parameter_names(1))
    do i = 1, nb_params
      write(dat_fh) parameter_names(i), parameters(i)
    end do
  end subroutine write_parameters


  subroutine write_background_names(settings)
    type(settings_t), intent(in) :: settings
    integer, parameter :: nb_names = 44
    character(len=str_len_arr) :: equilibrium_names(nb_names)

    equilibrium_names = [ &
      character(len=str_len_arr) :: &
      "rho0", "drho0", &
      "T0", "dT0", "ddT0", &
      "B01", "B02", "B03", "dB02", "dB03", "ddB02", "ddB03", "B0", &
      "v01", "v02", "v03", "dv01", "dv02", "dv03", "ddv01", "ddv02", "ddv03", &
      "L0", "dLdT", "dLdrho", &
      "lambdaT", "dlambdadT", &
      "H0", "dHdT", "dHdrho", &
      "kappa_para", "kappa_perp", &
      "dkappa_para_dT", "dkappa_para_dr", &
      "dkappa_perp_drho", "dkappa_perp_dT", "dkappa_perp_dB2", "dkappa_perp_dr", &
      "eta", "detadT", "detadr", &
      "gravity", &
      "Hall", "inertia" &
    ]
    if (.not. settings%io%write_background) then
      write(dat_fh) 0, 0
      return
    end if
    write(dat_fh) nb_names, len(equilibrium_names(1))
    write(dat_fh) equilibrium_names
  end subroutine write_background_names


  subroutine write_background_data(settings, grid, background, physics)
    use mod_function_utils, only: from_function

    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics

    write(dat_fh) from_function(background%density%rho0, grid%gaussian_grid)
    write(dat_fh) from_function(background%density%drho0, grid%gaussian_grid)
    write(dat_fh) from_function(background%temperature%T0, grid%gaussian_grid)
    write(dat_fh) from_function(background%temperature%dT0, grid%gaussian_grid)
    write(dat_fh) from_function(background%temperature%ddT0, grid%gaussian_grid)
    write(dat_fh) from_function(background%magnetic%B01, grid%gaussian_grid)
    write(dat_fh) from_function(background%magnetic%B02, grid%gaussian_grid)
    write(dat_fh) from_function(background%magnetic%B03, grid%gaussian_grid)
    write(dat_fh) from_function(background%magnetic%dB02, grid%gaussian_grid)
    write(dat_fh) from_function(background%magnetic%db03, grid%gaussian_grid)
    write(dat_fh) from_function(background%magnetic%ddB02, grid%gaussian_grid)
    write(dat_fh) from_function(background%magnetic%ddb03, grid%gaussian_grid)
    write(dat_fh) background%magnetic%get_B0(grid%gaussian_grid)
    write(dat_fh) from_function(background%velocity%v01, grid%gaussian_grid)
    write(dat_fh) from_function(background%velocity%v02, grid%gaussian_grid)
    write(dat_fh) from_function(background%velocity%v03, grid%gaussian_grid)
    write(dat_fh) from_function(background%velocity%dv01, grid%gaussian_grid)
    write(dat_fh) from_function(background%velocity%dv02, grid%gaussian_grid)
    write(dat_fh) from_function(background%velocity%dv03, grid%gaussian_grid)
    write(dat_fh) from_function(background%velocity%ddv01, grid%gaussian_grid)
    write(dat_fh) from_function(background%velocity%ddv02, grid%gaussian_grid)
    write(dat_fh) from_function(background%velocity%ddv03, grid%gaussian_grid)

    write(dat_fh) physics%heatloss%get_L0(grid%gaussian_grid)
    write(dat_fh) physics%heatloss%get_dLdT(grid%gaussian_grid)
    write(dat_fh) physics%heatloss%get_dLdrho(grid%gaussian_grid)
    write(dat_fh) from_function(physics%heatloss%cooling%lambdaT, grid%gaussian_grid)
    write(dat_fh) from_function(physics%heatloss%cooling%dlambdadT, grid%gaussian_grid)
    write(dat_fh) from_function(physics%heatloss%heating%H, grid%gaussian_grid)
    write(dat_fh) from_function(physics%heatloss%heating%dHdT, grid%gaussian_grid)
    write(dat_fh) from_function(physics%heatloss%heating%dHdrho, grid%gaussian_grid)

    write(dat_fh) from_function(physics%conduction%tcpara, grid%gaussian_grid)
    write(dat_fh) from_function(physics%conduction%tcperp, grid%gaussian_grid)
    write(dat_fh) from_function(physics%conduction%dtcparadT, grid%gaussian_grid)
    write(dat_fh) physics%conduction%get_dtcparadr(grid%gaussian_grid)
    write(dat_fh) from_function(physics%conduction%dtcperpdrho, grid%gaussian_grid)
    write(dat_fh) from_function(physics%conduction%dtcperpdT, grid%gaussian_grid)
    write(dat_fh) from_function(physics%conduction%dtcperpdB2, grid%gaussian_grid)
    write(dat_fh) physics%conduction%get_dtcperpdr(grid%gaussian_grid)

    write(dat_fh) from_function(physics%resistivity%eta, grid%gaussian_grid)
    write(dat_fh) from_function(physics%resistivity%detadT, grid%gaussian_grid)
    write(dat_fh) from_function(physics%resistivity%detadr, grid%gaussian_grid)
    write(dat_fh) from_function(physics%gravity%g0, grid%gaussian_grid)
    write(dat_fh) from_function(physics%hall%hallfactor, grid%gaussian_grid)
    write(dat_fh) from_function(physics%hall%inertiafactor, grid%gaussian_grid)
  end subroutine write_background_data


  subroutine write_base_eigenfunction_data(grid, eigenfunctions)
    type(grid_t), intent(in) :: grid
    type(eigenfunctions_t), intent(in) :: eigenfunctions
    integer :: i

    call logger%info("writing eigenfunctions...")
    write(dat_fh) size(grid%ef_grid), grid%ef_grid
    write(dat_fh) size(eigenfunctions%ef_written_flags), eigenfunctions%ef_written_flags
    write(dat_fh) size(eigenfunctions%ef_written_idxs), eigenfunctions%ef_written_idxs
    do i = 1, size(eigenfunctions%base_efs)
      write(dat_fh) eigenfunctions%base_efs(i)%quantities
    end do
  end subroutine write_base_eigenfunction_data


  subroutine write_derived_eigenfunction_data(settings, eigenfunctions)
    type(settings_t), intent(in) :: settings
    type(eigenfunctions_t), intent(in) :: eigenfunctions
    character(len=:), allocatable :: derived_state_vector(:)
    integer :: i

    allocate(derived_state_vector, source=settings%get_derived_state_vector())

    call logger%info("writing derived eigenfunctions...")
    write(dat_fh) size(derived_state_vector), len(derived_state_vector(1))
    write(dat_fh) derived_state_vector
    do i = 1, size(eigenfunctions%derived_efs)
      write(dat_fh) eigenfunctions%derived_efs(i)%quantities
    end do
    if (allocated(derived_state_vector)) deallocate(derived_state_vector)
  end subroutine write_derived_eigenfunction_data


  subroutine write_eigenvector_data(eigenvectors)
    complex(dp), intent(in) :: eigenvectors(:, :)

    call logger%info("writing eigenvectors...")
    write(dat_fh) size(eigenvectors, 1), size(eigenvectors, 2), eigenvectors
  end subroutine write_eigenvector_data


  subroutine write_residual_data(eigenvalues, matrix_A, matrix_B, eigenvectors)
    use mod_transform_matrix, only: matrix_to_banded
    use mod_banded_matrix, only: banded_matrix_t

    complex(dp), intent(in) :: eigenvalues(:)
    type(matrix_t), intent(in) :: matrix_A
    type(matrix_t), intent(in) :: matrix_B
    complex(dp), intent(in) :: eigenvectors(:, :)
    real(dp), allocatable :: residuals(:)
    integer :: i, subdiags, superdiags
    type(banded_matrix_t) :: matrix_A_banded, matrix_B_banded

    call logger%info("computing residuals...")
    call matrix_A%get_nb_diagonals(ku=superdiags, kl=subdiags)
    call matrix_to_banded(matrix_A, subdiags, superdiags, banded=matrix_A_banded)
    call matrix_B%get_nb_diagonals(ku=superdiags, kl=subdiags)
    call matrix_to_banded(matrix_B, subdiags, superdiags, banded=matrix_B_banded)
    allocate(residuals(size(eigenvalues)))
    do i = 1, size(residuals)
      residuals(i) = get_residual( &
        matrix_A_banded, matrix_B_banded, eigenvalues(i), eigenvectors(:, i) &
      )
    end do
    call logger%info("writing residuals...")
    write(dat_fh) size(residuals), residuals
    deallocate(residuals)
    call matrix_A_banded%destroy()
    call matrix_B_banded%destroy()
  end subroutine write_residual_data


  subroutine write_matrix_data(matrix_A, matrix_B)
    use mod_matrix_node, only: node_t

    type(matrix_t), intent(in) :: matrix_A
    type(matrix_t), intent(in) :: matrix_B
    type(node_t), pointer :: current_node
    integer :: irow, inode

    ! write total number of nonzero elements
    write(dat_fh) matrix_B%get_total_nb_elements()
    write(dat_fh) matrix_A%get_total_nb_elements()

    ! write matrix B
    do irow = 1, matrix_B%matrix_dim
      current_node => matrix_B%rows(irow)%head
      do inode = 1, matrix_B%rows(irow)%nb_elements
        write(dat_fh) irow, current_node%column
        ! B is real, so write only real values
        write(dat_fh) real(current_node%get_node_element())
        current_node => current_node%next
      end do
    end do
    ! write matrix A
    do irow = 1, matrix_A%matrix_dim
      current_node => matrix_A%rows(irow)%head
      do inode = 1, matrix_A%rows(irow)%nb_elements
        write(dat_fh) irow, current_node%column
        write(dat_fh) current_node%get_node_element()
        current_node => current_node%next
      end do
    end do
    nullify(current_node)
  end subroutine write_matrix_data


  real(dp) function get_residual(amat_band, bmat_band, eigenvalue, eigenvector)
    use mod_check_values, only: is_zero
    use mod_banded_matrix, only: banded_matrix_t
    use mod_banded_operations, only: multiply

    !> the matrix A in banded form
    type(banded_matrix_t), intent(in) :: amat_band
    !> the matrix B in banded form
    type(banded_matrix_t), intent(in) :: bmat_band
    !> the eigenvalue
    complex(dp), intent(in) :: eigenvalue
    !> the eigenvector
    complex(dp), intent(in) :: eigenvector(:)

    integer :: N
    real(dp) :: dznrm2

    if (is_zero(eigenvalue)) then
      get_residual = 0.0_dp
      return
    end if
    N = size(eigenvector)

    get_residual = ( &
      dznrm2( &
        N, &
        ( &
          multiply(amat_band, eigenvector) &
          - eigenvalue * multiply(bmat_band, eigenvector) &
        ), &
        1 &
      ) &
      / dznrm2(N, eigenvalue * eigenvector, 1) &
    )
  end function get_residual


  subroutine write_iv_snapshots(settings, iv_module)
    type(settings_t), intent(in) :: settings
    type(iv_module_t), intent(inout) :: iv_module
    integer :: i_snap, i_comp
    integer :: n_snap, n_points
    
    call logger%info("writing IV snapshots")
    n_snap   = settings%iv%get_n_snapshots()
    n_points = settings%grid%get_ef_gridpts()

    ! Metadata
    write(dat_fh) n_snap
    write(dat_fh) n_points
    write(dat_fh) iv_module%state_vec%num_components
  
    ! Write snapshot times
    write(dat_fh) iv_module%snap_times

    ! Reassemble on the fly
    do i_snap = 1, n_snap
      call iv_module%postprocess_snapshot(i_snap)

      do i_comp = 1, iv_module%state_vec%num_components
        write(dat_fh) real(iv_module%state_vec%components(i_comp)%ptr%profile, kind = dp)
      end do
    end do

  end subroutine write_iv_snapshots
  

end module mod_output
