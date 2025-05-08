! =============================================================================
!> Parent module governing all equilibrium types and submodules.
!! This module contains all equilibrium types and the initial
!! declarations of the module subroutines. Every equilibrium submodule
!! extends this module, implementing one of the module subroutines declared here.
!! All "main" equilibrium configurations are set in the submodules. The ones that
!! depend on "main" arrays, like radiative cooling, are set here through calls to their
!! respective modules.
!! @note    All use statements specified here at the main module scope
!!          are automatically accessible in every submodule that extends this one.
module mod_equilibrium
  use mod_global_variables, only: dp
  use mod_physical_constants, only: dpi
  use mod_equilibrium_params, only: k2, k3, input_file
  use mod_logging, only: logger, str, exp_fmt
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_physics, only: physics_t
  use mod_grid, only: grid_t
  implicit none

  private

  !> pointer for the submodule, initialised to null
  procedure(), pointer :: set_equilibrium_values => null()

  !> interface to the different equilibrium submodules
  interface
    module subroutine adiabatic_homo_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine adiabatic_homo_eq
    module subroutine constant_current_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine constant_current_eq
    module subroutine coronal_flux_tube_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine coronal_flux_tube_eq
    module subroutine discrete_alfven_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine discrete_alfven_eq
    module subroutine flow_driven_instabilities_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine flow_driven_instabilities_eq
    module subroutine gold_hoyle_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine gold_hoyle_eq
    module subroutine gravito_acoustic_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine gravito_acoustic_eq
    module subroutine gravito_mhd_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine gravito_mhd_eq
    module subroutine interchange_modes_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine interchange_modes_eq
    module subroutine internal_kink_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine internal_kink_eq
    module subroutine isothermal_atmosphere_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine isothermal_atmosphere_eq
    module subroutine KHI_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine
    module subroutine kh_cd_instability_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine kh_cd_instability_eq
    module subroutine magnetothermal_instability_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine magnetothermal_instability_eq
    module subroutine MRI_accretion_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine MRI_accretion_eq
    module subroutine photospheric_flux_tube_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine photospheric_flux_tube_eq
    module subroutine resistive_homo_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine resistive_homo_eq
    module subroutine resistive_tearing_modes_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine resistive_tearing_modes_eq
    module subroutine resistive_tearing_modes_flow_eq( &
      settings, grid, background, physics &
    )
      type(grid_t), intent(inout) :: grid
      type(settings_t), intent(inout) :: settings
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine resistive_tearing_modes_flow_eq
    module subroutine resonant_absorption_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine resonant_absorption_eq
    module subroutine rotating_plasma_cyl_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine rotating_plasma_cyl_eq
    module subroutine RTI_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine RTI_eq
    module subroutine RTI_KHI_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine RTI_KHI_eq
    module subroutine RTI_theta_pinch_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine RTI_theta_pinch_eq
    module subroutine suydam_cluster_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine suydam_cluster_eq
    module subroutine couette_flow_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine couette_flow_eq
    module subroutine taylor_couette_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine taylor_couette_eq
    module subroutine harris_sheet_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine harris_sheet_eq
    module subroutine tc_pinch_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine tc_pinch_eq
    module subroutine numerical_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine numerical_eq
    module subroutine user_defined_eq(settings, grid, background, physics)
      type(settings_t), intent(inout) :: settings
      type(grid_t), intent(inout) :: grid
      type(background_t), intent(inout) :: background
      type(physics_t), intent(inout) :: physics
    end subroutine user_defined_eq
  end interface

  public :: set_equilibrium

contains


  !> Calls the routine to set the equilibrium pointer, then calls the correct
  !! submodule. Performs some sanity checks (negative values, NaNs etc.) when
  !! the equilibrium is set, then calls additional physics modules if needed.
  !! @warning Throws appropriate errors if the equilibrium configuration is
  !!          not balanced, contains NaN or if density/temperature contains
  !!          negative values.
  subroutine set_equilibrium(settings, grid, background, physics)
    type(settings_t), intent(inout) :: settings
    type(grid_t), intent(inout) :: grid
    type(background_t), intent(inout) :: background
    type(physics_t), intent(inout) :: physics

    ! Set equilibrium submodule to use
    call set_equilibrium_pointer(settings)
    ! Call submodule
    call set_equilibrium_values(settings, grid, background, physics)
    call grid%initialise()

    if (settings%physics%cooling%is_enabled()) then
      call physics%heatloss%cooling%initialise()
    end if
    call physics%hall%validate_scale_ratio(grid%gaussian_grid)
  end subroutine set_equilibrium


  !> Selects the submodule based on the specified equilibrium
  !! in the parfile. Works on a case-select basis.
  !! @warning   Throws an error if the equilibrium type is not recognised.
  subroutine set_equilibrium_pointer(settings)
    type(settings_t), intent(in) :: settings

    select case(settings%equilibrium%get_equilibrium_type())
    case("adiabatic_homo")
      set_equilibrium_values => adiabatic_homo_eq
    case("constant_current_tokamak")
      set_equilibrium_values => constant_current_eq
    case("coronal_flux_tube")
      set_equilibrium_values => coronal_flux_tube_eq
    case("discrete_alfven")
      set_equilibrium_values => discrete_alfven_eq
    case("gold_hoyle")
      set_equilibrium_values => gold_hoyle_eq
    case("gravito_acoustic")
      set_equilibrium_values => gravito_acoustic_eq
    case("gravito_mhd")
      set_equilibrium_values => gravito_mhd_eq
    case("interchange_modes")
      set_equilibrium_values => interchange_modes_eq
    case("internal_kink")
      set_equilibrium_values => internal_kink_eq
    case("isothermal_atmosphere")
      set_equilibrium_values => isothermal_atmosphere_eq
    case("kelvin_helmholtz")
      set_equilibrium_values => KHI_eq
    case("kelvin_helmholtz_cd")
      set_equilibrium_values => kh_cd_instability_eq
    case("magnetothermal_instabilities")
      set_equilibrium_values => magnetothermal_instability_eq
    case("MRI_accretion")
      set_equilibrium_values => MRI_accretion_eq
    case("photospheric_flux_tube")
      set_equilibrium_values => photospheric_flux_tube_eq
    case("rayleigh_taylor")
      set_equilibrium_values => RTI_eq
    case("resistive_homo")
      set_equilibrium_values => resistive_homo_eq
    case("resistive_tearing")
      set_equilibrium_values => resistive_tearing_modes_eq
    case("resistive_tearing_flow")
      set_equilibrium_values => resistive_tearing_modes_flow_eq
    case("resonant_absorption")
      set_equilibrium_values => resonant_absorption_eq
    case("rotating_plasma_cylinder")
      set_equilibrium_values => rotating_plasma_cyl_eq
    case("RTI_KHI")
      set_equilibrium_values => RTI_KHI_eq
    case("RTI_theta_pinch")
      set_equilibrium_values => RTI_theta_pinch_eq
    case("suydam_cluster")
      set_equilibrium_values => suydam_cluster_eq
    case("couette_flow")
      set_equilibrium_values => couette_flow_eq
    case("taylor_couette")
      set_equilibrium_values => taylor_couette_eq
    case("harris_sheet")
      set_equilibrium_values => harris_sheet_eq
    case("tc_pinch")
      set_equilibrium_values => tc_pinch_eq
    case("numerical")
      set_equilibrium_values => numerical_eq
    case("user_defined")
      set_equilibrium_values => user_defined_eq
    case default
      call logger%error( &
        "equilibrium not recognised: " &
        // trim(settings%equilibrium%get_equilibrium_type()) &
      )
    end select
  end subroutine set_equilibrium_pointer

end module mod_equilibrium
