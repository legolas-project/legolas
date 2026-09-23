module mod_console
  use mod_global_variables, only: str_len
  use mod_settings, only: settings_t
  use mod_logging, only: logger, str, exp_fmt
  implicit none

  private

  public :: print_logo
  public :: print_startup_info
  public :: print_console_info
  public :: print_whitespace

contains

  ! LCOV_EXCL_START (we don't test printing statements)

  !> Prints the Legolas logo to the console.
  !! The logo is wrapped in 1 whitespace at the top and
  !! two at the bottom. Only for logging level 'warning' (1) and above
  subroutine print_logo()
    use mod_version, only: LEGOLAS_VERSION
    use mod_painting, only: paint_string

    !> array containing the different logo lines
    character(len=str_len) :: logo(11)
    !> whitespace prepended to logo
    character(len=3)       :: spaces_logo = ""
    !> whitespace prepended to versioning
    character(len=57)      :: spaces_v = ""
    integer :: i

    if (logger%get_logging_level() <= 1) return

    logo(1) = " __       ________  ________   _______   __          ___     __________ "
    logo(2) = "|  |     |   ____ \|   ____ \ /   _   \ |  |        /   \   |   ______ \"
    logo(3) = "|  |     |  |    \/|  |    \/|   / \   ||  |       /  _  \  |  |      \/"
    logo(4) = "|  |     |  |__    |  |      |  |   |  ||  |      /  / \  \ |  \_______ "
    logo(5) = "|  |     |   __/   |  | ____ |  |   |  ||  |     /  /   \  \\_______   \"
    logo(6) = "|  |     |  |      |  | \_  ||  |   |  ||  |    /  /     \  \       |  |"
    logo(7) = "|  |_____|  |____/\|  |___| ||   \_/   ||  |___/  /  /\___\  \      |  |"
    logo(8) = "|_______/|________/|________| \_______/ |________/   \_______/      |  |"
    logo(9) = "                                                        /\__________/  |"
    logo(10) ="Large Eigensystem Generator for One-dimensional pLASmas \______________/"
    logo(11) = spaces_v // "v. " // trim(adjustl(LEGOLAS_VERSION))

    call print_whitespace(1)
    do i = 1, size(logo)
      write(*, *) paint_string(spaces_logo // trim(logo(i)), "cyan")
    end do
    call print_whitespace(2)
  end subroutine print_logo


  subroutine print_startup_info(settings)
    type(settings_t), intent(in) :: settings

    call logger%info("the physics type is " // settings%get_physics_type())
    call logger%info("the state vector is " // str(settings%get_state_vector()))
    call logger%info( &
      "the basis functions are " // str(settings%state_vector%get_basis_functions()) &
    )
  end subroutine print_startup_info


  ! LCOV_EXCL_START <we don't print info to console when testing>
  !> Prints various console messages showing geometry, grid parameters,
  !! equilibrium parameters etc. Only for logging level "info" or above.
  subroutine print_console_info(settings)
    type(settings_t), intent(in) :: settings

    if (logger%get_logging_level() <= 1) return

    call logger%info("---------------------------------------------")
    call logger%disable_prefix()

    call log_grid_info(settings)
    call log_equilibrium_info(settings)
    call log_physics_info(settings)
    call log_solver_info(settings)
    call log_io_info(settings)

    call logger%info("---------------------------------------------")
    call logger%enable_prefix()
    call print_whitespace(1)
  end subroutine print_console_info


  !> Prints an empty line to the console.
  !! Only if logging level is 'warning' or above.
  subroutine print_whitespace(lines)
    !> amount of empty lines to print
    integer, intent(in) :: lines
    integer :: i

    if (logger%get_logging_level() > 1) then
      do i = 1, lines
        write(*, *) ""
      end do
    end if
  end subroutine print_whitespace


  subroutine log_grid_info(settings)
    type(settings_t), intent(in) :: settings
    integer :: dims

    call logger%info("              << Grid settings >>")
    call logger%info("geometry         : " // settings%grid%get_geometry())
    call logger%info("grid start       : " // str(settings%grid%get_grid_start()))
    call logger%info("grid end         : " // str(settings%grid%get_grid_end()))
    call logger%info("points base grid : " // str(settings%grid%get_gridpts()))
    dims = settings%dims%get_dim_matrix()
    call logger%info("matrix dimensions: " // str(dims) // " x " // str(dims))
  end subroutine log_grid_info


  subroutine log_equilibrium_info(settings)
    use mod_equilibrium_params, only: k2, k3
    type(settings_t), intent(in) :: settings

    call logger%info("          << Equilibrium settings >>")
    call logger%info("equilibrium : " // settings%equilibrium%get_equilibrium_type())
    call logger%info("k2          : " // str(k2))
    call logger%info("k3          : " // str(k3))
    call logger%info("use defaults: " // str(settings%equilibrium%use_defaults))
  end subroutine log_equilibrium_info


  subroutine log_physics_info(settings)
    type(settings_t), intent(in) :: settings

    call logger%info("            << Physics settings >>")
    call logger%info("physics type : " // settings%get_physics_type())
    call logger%info("gamma        : " // str(settings%physics%get_gamma()))
    call log_flow_info()
    call log_gravity_info()
    call log_cooling_info()
    call log_heating_info()
    call log_parallel_conduction_info()
    call log_perpendicular_conduction_info()
    call log_resistivity_info()
    call log_viscosity_info()
    call log_hall_info()

    contains

    subroutine log_flow_info()
      logical :: flow
      flow = settings%physics%flow%is_enabled()
      if (.not. flow) return
      call logger%info("flow                     : " // str(flow))
    end subroutine log_flow_info

    subroutine log_gravity_info()
      logical :: gravity
      gravity = settings%physics%gravity%is_enabled()
      if (.not. gravity) return
      call logger%info("external gravity         : " // str(gravity))
    end subroutine log_gravity_info

    subroutine log_cooling_info()
      logical :: cooling
      cooling = settings%physics%cooling%is_enabled()
      if (.not. cooling) return
      call logger%info("radiative cooling        : " // str(cooling))
      call logger%info("  cooling curve          : " // &
        trim(adjustl(settings%physics%cooling%get_cooling_curve())) &
      )
    end subroutine log_cooling_info

    subroutine log_heating_info()
      logical :: heating
      heating = settings%physics%heating%is_enabled()
      if (.not. heating) return
      call logger%info("heating                  : " // str(heating))
      call logger%info("  forcing thermal balance: " // &
        str(settings%physics%heating%force_thermal_balance) &
      )
    end subroutine log_heating_info

    subroutine log_parallel_conduction_info()
      logical :: tc_para
      tc_para = settings%physics%conduction%has_parallel_conduction()
      if (.not. tc_para) return
      call logger%info("parallel conduction      : " // str(tc_para))
      if (settings%physics%conduction%has_fixed_tc_para()) then
        call logger%info("  fixed value            : " // &
          str(settings%physics%conduction%get_fixed_tc_para(), fmt=exp_fmt) &
        )
      end if
    end subroutine log_parallel_conduction_info

    subroutine log_perpendicular_conduction_info()
      logical :: tc_perp
      tc_perp = settings%physics%conduction%has_perpendicular_conduction()
      if (.not. tc_perp) return
      call logger%info("perpendicular conduction : " // str(tc_perp))
      if (settings%physics%conduction%has_fixed_tc_perp()) then
        call logger%info("  fixed value            : " // &
          str(settings%physics%conduction%get_fixed_tc_perp(), fmt=exp_fmt) &
        )
      end if
    end subroutine log_perpendicular_conduction_info

    subroutine log_resistivity_info()
      logical :: resistivity
      resistivity = settings%physics%resistivity%is_enabled()
      if (.not. resistivity) return
      call logger%info("resistivity              : " // str(resistivity))
      if (settings%physics%resistivity%has_fixed_resistivity()) then
        call logger%info("  fixed eta value        : " // &
          str(settings%physics%resistivity%get_fixed_resistivity(), fmt=exp_fmt) &
        )
      end if
    end subroutine log_resistivity_info

    subroutine log_viscosity_info()
      logical :: viscosity
      viscosity = settings%physics%viscosity%is_enabled()
      if (.not. viscosity) return
      call logger%info("viscosity                : " // str(viscosity))
      call logger%info( &
        "  viscosity value        : " // &
        str(settings%physics%viscosity%get_viscosity_value()) &
      )
      call logger%info( &
        "  viscous heating        : " // &
        str(settings%physics%viscosity%has_viscous_heating()) &
      )
    end subroutine log_viscosity_info

    subroutine log_hall_info()
      logical :: hall
      hall = settings%physics%hall%is_enabled()
      if (.not. hall) return
      call logger%info("hall mhd                 : " // str(hall))
      call logger%info( &
        "  by substitution        : " // &
        str(settings%physics%hall%is_using_substitution()) &
      )
      call logger%info( &
        "  electron fraction      : " // &
        str(settings%physics%hall%get_electron_fraction()) &
      )
      call logger%info( &
        "  electron inertia       : " // &
        str(settings%physics%hall%has_electron_inertia()) &
      )
    end subroutine log_hall_info
  end subroutine log_physics_info


  subroutine log_solver_info(settings)
    type(settings_t), intent(in) :: settings

    call logger%info("            << Solver settings >>")
    call logger%info("solver      : " // settings%solvers%get_solver())
    select case(settings%solvers%get_solver())
    case("arnoldi")
      call logger%info("ARPACK mode : " // settings%solvers%get_arpack_mode())
      call logger%info( &
        "# eigenvalues : " // str(settings%solvers%number_of_eigenvalues) &
      )
      call logger%info("which eigenvalues : " // settings%solvers%which_eigenvalues)
      call logger%info("maxiter : " // str(settings%solvers%maxiter))
      call logger%info("ncv     : " // str(settings%solvers%ncv))
      if (settings%solvers%get_arpack_mode() == "shift-invert") then
        call logger%info("sigma : " // str(settings%solvers%sigma))
      end if
      call logger%info("tolerance : " // str(settings%solvers%tolerance, exp_fmt))
    case("inverse-iteration")
      call logger%info("sigma     : " // str(settings%solvers%sigma))
      call logger%info("maxiter   : " // str(settings%solvers%maxiter))
      call logger%info("tolerance : " // str(settings%solvers%tolerance, exp_fmt))
    end select
  end subroutine log_solver_info


  subroutine log_io_info(settings)
    type(settings_t), intent(in) :: settings

    call logger%info("            << DataIO settings >>")
    call logger%info("datfile name         : " // settings%io%get_basename_datfile())
    call logger%info("output folder        : " // settings%io%get_output_folder())
    call logger%info("write background     : " // str(settings%io%write_background))
    call logger%info("write matrices       : " // str(settings%io%write_matrices))
    call logger%info("write eigenfunctions : " // str(settings%io%write_eigenfunctions))
    call logger%info( &
      "write derived eigenfunctions : " &
      // str(settings%io%write_derived_eigenfunctions) &
    )
    if (settings%io%write_eigenvectors) then
      call logger%info("write eigenvectors   : " // str(settings%io%write_eigenvectors))
    end if
    if (settings%io%write_residuals) then
      call logger%info("write residuals      : " // str(settings%io%write_residuals))
    end if
    if (.not. settings%io%write_ef_subset) return
    call logger%info( &
      "write eigenfunction subset : " // str(settings%io%write_ef_subset) &
    )
    call logger%info("    subset center : " // str(settings%io%ef_subset_center))
    call logger%info("    subset radius : " // str(settings%io%ef_subset_radius))
  end subroutine log_io_info

end module mod_console
