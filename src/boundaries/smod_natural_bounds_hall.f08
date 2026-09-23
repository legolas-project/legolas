submodule (mod_natural_boundaries) smod_natural_bounds_hall
  implicit none

contains

  module procedure add_natural_hall_Bterms
    real(dp)  :: eps
    real(dp)  :: rho
    real(dp)  :: eta_e

    if (.not. settings%physics%hall%is_enabled()) return
    if (.not. settings%physics%hall%has_electron_inertia()) return

    eps = grid%get_eps(x)
    rho = background%density%rho0(x)
    eta_e = physics%hall%inertiafactor(x)

    ! ==================== Cubic * Quadratic ====================
    call elements%add(eta_e * k2 / rho, sv_a2, sv_a1)
    call elements%add(eta_e * k3 * eps / rho, sv_a3, sv_a1)
    ! ==================== Cubic * dCubic ====================
    call elements%add(-eta_e / rho, sv_a2, sv_a2, s2do=1)
    call elements%add(-eta_e * eps / rho, sv_a3, sv_a3, s2do=1)
  end procedure add_natural_hall_Bterms

  module procedure add_natural_hall_terms
    real(dp)  :: eps, deps
    real(dp)  :: rho, T0, B01, B02, B03
    real(dp)  :: eta_H, mu, efrac

    if (.not. settings%physics%hall%is_enabled()) return
    ! Hall by substitution of the momentum equation, only contains viscosity terms
    if (.not. settings%physics%viscosity%is_enabled()) return

    eps = grid%get_eps(x)
    deps = grid%get_deps()

    rho = background%density%rho0(x)
    T0 = background%temperature%T0(x)
    B01 = background%magnetic%B01(x)
    B02 = background%magnetic%B02(x)
    B03 = background%magnetic%B03(x)

    eta_H = physics%hall%hallfactor(x)
    mu = settings%physics%viscosity%get_viscosity_value()
    efrac = settings%physics%hall%get_electron_fraction()

    ! ==================== Quadratic * Cubic ====================
    call elements%add(-eta_H * ic * mu * deps / (eps * rho), sv_a1, sv_v1)
    ! ==================== Quadratic * dCubic ====================
    call elements%add(4.0d0 * eta_H * ic * mu / (3.0d0 * rho), sv_a1, sv_v1, s2do=1)
    ! ==================== Cubic * dQuadratic ====================
    call elements%add(eta_H * ic * mu * eps / rho, sv_a2, sv_v2, s2do=1)
    call elements%add(eta_H * ic * mu / rho, sv_a3, sv_v3, s2do=1)
    ! ==================== Cubic * Quadratic ====================
    call elements%add(-eta_H * ic * mu * deps / (eps * rho), sv_a3, sv_v3)
  end procedure add_natural_hall_terms

end submodule
