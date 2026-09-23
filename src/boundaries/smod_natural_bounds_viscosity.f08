submodule (mod_natural_boundaries) smod_natural_bounds_viscosity
  implicit none

contains

  module procedure add_natural_viscosity_terms
    real(dp)  :: eps, deps
    real(dp)  :: mu
    real(dp)  :: dv01, dv03
    real(dp) :: gamma_1
    logical :: viscous_heating, is_compressible

    if (.not. settings%physics%viscosity%is_enabled()) return

    gamma_1 = settings%physics%get_gamma_1()
    viscous_heating = settings%physics%viscosity%has_viscous_heating()
    is_compressible = .not. settings%physics%is_incompressible

    eps = grid%get_eps(x)
    deps = grid%get_deps()
    mu = settings%physics%viscosity%get_viscosity_value()
    dv01 = background%velocity%dv01(x)
    dv03 = background%velocity%dv03(x)

    ! ==================== Cubic * Cubic ====================
    call elements%add(-ic * mu * deps / eps, sv_v1, sv_v1)
    ! ==================== Cubic * dCubic ====================
    call elements%add(4.0d0 * ic * mu / 3.0d0, sv_v1, sv_v1, s2do=1)
    ! ==================== Cubic * Quadratic ====================
    call elements%add(-ic * mu * k2 / 3.0d0, sv_v1, sv_v2)
    call elements%add(-ic * mu * k3 / 3.0d0, sv_v1, sv_v3)
    ! ==================== Quadratic * dQuadratic ====================
    call elements%add(ic * mu * eps, sv_v2, sv_v2, s2do=1)
    call elements%add(ic * mu, sv_v3, sv_v3, s2do=1)
    ! ==================== Quadratic * Quadratic ====================
    call elements%add(-ic * mu * deps / eps, sv_v3, sv_v3)

    if (viscous_heating .and. is_compressible) then
      ! ==================== Quadratic * Quadratic ====================
      call elements%add(2.0d0 * ic * gamma_1 * mu * dv03, sv_T1, sv_v3)
      ! ==================== Quadratic * Cubic ====================
      call elements%add(2.0d0 * gamma_1 * mu * dv01, sv_T1, sv_v1)
    end if
  end procedure add_natural_viscosity_terms

end submodule
