submodule (mod_natural_boundaries) smod_natural_bounds_regular
  implicit none

contains

  module procedure add_natural_regular_terms
    real(dp)  :: eps
    real(dp)  :: rho, T0
    real(dp)  :: B01, B02, B03
    real(dp)  :: eps_Gop_min

    eps = grid%get_eps(x)
    rho = background%density%rho0(x)
    T0 = background%temperature%T0(x)
    B01 = background%magnetic%B01(x)
    B02 = background%magnetic%B02(x)
    B03 = background%magnetic%B03(x)
    eps_Gop_min = eps * k3 * B02 - k2 * B03 ! this is eps * Gop_min regularised at r=0

    ! ==================== Cubic * Quadratic ====================
    call elements%add(T0, sv_v1, sv_rho1)
    call elements%add(rho, sv_v1, sv_T1)
    call elements%add(eps_Gop_min, sv_v1, sv_a1)
    ! ==================== Cubic * dCubic ====================
    call elements%add(B03, sv_v1, sv_a2, s2do=1)
    call elements%add(-eps * B02, sv_v1, sv_a3, s2do=1)
    ! ==================== Quadratic * Quadratic ====================
    call elements%add(ic * eps * k3 * B01, sv_v2, sv_a1)
    call elements%add(-ic * k2 * B01, sv_v3, sv_a1)
    ! ==================== Quadratic * dCubic ====================
    call elements%add(-ic * eps * B01, sv_v2, sv_a3, s2do=1)
    call elements%add(ic * B01, sv_v3, sv_a2, s2do=1)
  end procedure add_natural_regular_terms

end submodule smod_natural_bounds_regular
