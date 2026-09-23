submodule (mod_natural_boundaries) smod_natural_bounds_flow
  implicit none

contains

  module procedure add_natural_flow_terms
    real(dp)  :: rho
    real(dp)  :: v01

    if (.not. settings%physics%flow%is_enabled()) return

    rho = background%density%rho0(x)
    v01 = background%velocity%v01(x)

    ! ==================== Cubic * Cubic ====================
    call elements%add(-ic * rho * v01, sv_v1, sv_v1)
    ! ==================== Quadratic * Quadratic ====================
    call elements%add(-ic * rho * v01, sv_v3, sv_v3)
    call elements%add(-ic * rho * v01, sv_T1, sv_T1)
  end procedure add_natural_flow_terms

end submodule smod_natural_bounds_flow
