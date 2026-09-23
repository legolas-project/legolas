submodule (mod_natural_boundaries) smod_natural_bounds_resistive
  implicit none

contains

  module procedure add_natural_resistive_terms
    real(dp)  :: eps, deps
    real(dp)  :: eta
    real(dp)  :: B02, dB02, drB02
    real(dp)  :: B03, dB03
    real(dp) :: gamma_1

    if (.not. settings%physics%resistivity%is_enabled()) return

    gamma_1 = settings%physics%get_gamma_1()

    eps = grid%get_eps(x)
    deps = grid%get_deps()
    eta = physics%resistivity%eta(x)
    B02 = background%magnetic%B02(x)
    dB02 = background%magnetic%dB02(x)
    B03 = background%magnetic%B03(x)
    dB03 = background%magnetic%dB03(x)

    drB02 = deps * B02 + eps * dB02

    ! ==================== Quadratic * Quadratic ====================
    call elements%add( &
      2.0d0 * ic * gamma_1 * eta * (k3 * drB02 - k2 * dB03), sv_T1, sv_a1 &
    )
    ! ==================== Quadratic * dCubic ====================
    call elements%add(2.0d0 * ic * gamma_1 * eta * dB03, sv_T1, sv_a2, s2do=1)
    call elements%add(-2.0d0 * ic * gamma_1 * eta * drB02, sv_T1, sv_a3, s2do=1)
    ! ==================== Cubic * Quadratic ====================
    call elements%add(-ic * eta * k2, sv_a2, sv_a1)
    call elements%add(-ic * eta * eps * k3, sv_a3, sv_a1)
    ! ==================== Cubic * dCubic ====================
    call elements%add(ic * eta, sv_a2, sv_a2, s2do=1)
    call elements%add(ic * eta * eps, sv_a3, sv_a3, s2do=1)
  end procedure add_natural_resistive_terms

end submodule smod_natural_bounds_resistive
