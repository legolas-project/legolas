submodule (mod_matrix_manager) smod_regular_matrix
  implicit none

contains

  module procedure add_bmatrix_terms
    real(dp)  :: rho, eps

    rho = background%density%rho0(x)
    eps = grid%get_eps(x)

    ! ==================== Quadratic * Quadratic ====================
    call elements%add(1.0_dp, sv_rho1, sv_rho1)
    call elements%add(eps * rho, sv_v2, sv_v2)
    call elements%add(rho, sv_v3, sv_v3)
    call elements%add(rho, sv_T1, sv_T1)
    call elements%add(eps, sv_a1, sv_a1)
    ! ==================== Cubic * Cubic ====================
    call elements%add(rho, sv_v1, sv_v1)
    call elements%add(1.0_dp, sv_a2, sv_a2)
    call elements%add(eps, sv_a3, sv_a3)
  end procedure add_bmatrix_terms


  module procedure add_regular_matrix_terms
    real(dp) :: eps, deps
    real(dp) :: rho, drho
    real(dp) :: T0, dT0
    real(dp) :: g0
    real(dp) :: B01, B02, dB02, drB02, B03, dB03
    real(dp) :: Fop_plus, Gop_plus, Gop_min, WVop
    real(dp) :: gamma_1

    gamma_1 = settings%physics%get_gamma_1()

    ! grid variables
    eps = grid%get_eps(x)
    deps = grid%get_deps()
    ! density variables
    rho = background%density%rho0(x)
    drho = background%density%drho0(x)
    ! temperature variables
    T0 = background%temperature%T0(x)
    dT0 = background%temperature%dT0(x)
    g0 = physics%gravity%g0(x)
    ! magnetic field variables
    B01 = background%magnetic%B01(x)
    B02 = background%magnetic%B02(x)
    dB02 = background%magnetic%dB02(x)
    B03 = background%magnetic%B03(x)
    dB03 = background%magnetic%dB03(x)
    drB02 = deps * B02 + eps * dB02
    ! operators
    Fop_plus = k2 * B02 / eps + k3 * B03
    Gop_plus = k3 * B02 + k2 * B03 / eps
    Gop_min = k3 * B02 - k2 * B03 / eps
    WVop = k2**2 / eps + eps * k3**2


    ! ==================== Quadratic * Cubic ====================
    call elements%add(-drho, sv_rho1, sv_v1)
    call elements%add(k3 * (drB02 - ic * k2 * B01) / eps, sv_v2, sv_a2)
    call elements%add(k2 * (ic * k2 * B01 - drB02) / eps, sv_v2, sv_a3)
    call elements%add(k3 * (dB03 - ic * k3 * B01), sv_v3, sv_a2)
    call elements%add(k2 * (ic * k3 * B01 - dB03), sv_v3, sv_a3)
    if (.not. settings%physics%is_incompressible) call elements%add( &
      -dT0 * rho, sv_T1, sv_v1 &
    )

    ! ==================== Quadratic * dCubic ====================
    call elements%add(-rho, sv_rho1, sv_v1, s2do=1)
    call elements%add(k2 * B03 / eps, sv_v2, sv_a2, s2do=1)
    call elements%add(eps * k3 * B03, sv_v2, sv_a3, s2do=1)
    call elements%add(-(k2 * B02 + ic * deps * B01) / eps, sv_v3, sv_a2, s2do=1)
    call elements%add(-eps * k3 * B02, sv_v3, sv_a3, s2do=1)
    call elements%add(-gamma_1 * T0 * rho, sv_T1, sv_v1, s2do=1)

    ! ==================== Quadratic * Quadratic ====================
    call elements%add(rho * k2, sv_rho1, sv_v2)
    call elements%add(rho * k3, sv_rho1, sv_v3)
    call elements%add(k2 * T0 / eps, sv_v2, sv_rho1)
    call elements%add(k2 * rho / eps, sv_v2, sv_T1)
    call elements%add(-WVop * B03, sv_v2, sv_a1)
    call elements%add(k3 * T0, sv_v3, sv_rho1)
    call elements%add(k3 * rho, sv_v3, sv_T1)
    call elements%add(ic * deps * k2 * B01 / eps + B02 * WVop, sv_v3, sv_a1)
    call elements%add(gamma_1 * k2 * rho * T0, sv_T1, sv_v2)
    call elements%add(gamma_1 * k3 * rho * T0, sv_T1, sv_v3)
    call elements%add(-eps * B03, sv_a1, sv_v2)
    call elements%add(B02, sv_a1, sv_v3)

    ! ==================== Cubic * Quadratic ====================
    if (.not. settings%get_physics_type() == "isothermal-1d") call elements%add( &
      -deps * T0 / eps, sv_v1, sv_rho1 &
    )
    if (settings%physics%gravity%is_enabled()) call elements%add(g0, sv_v1, sv_rho1)
    call elements%add(-deps * rho / eps, sv_v1, sv_T1)
    call elements%add(deps * Gop_plus, sv_v1, sv_a1)
    call elements%add(ic * B01, sv_a2, sv_v3)
    call elements%add(-ic * eps * B01, sv_a3, sv_v2)

    ! ==================== dCubic * Quadratic ====================
    call elements%add(-T0, sv_v1, sv_rho1, s1do=1)
    call elements%add(-rho, sv_v1, sv_T1, s1do=1)
    call elements%add(-eps * Gop_min, sv_v1, sv_a1, s1do=1)

    ! ==================== Cubic * Cubic ====================
    call elements%add(-k3 * Fop_plus, sv_v1, sv_a2)
    call elements%add(k2 * Fop_plus, sv_v1, sv_a3)
    call elements%add(-B03, sv_a2, sv_v1)
    call elements%add(B02, sv_a3, sv_v1)

    ! ==================== Cubic * dCubic ====================
    call elements%add(-deps * B03 / eps, sv_v1, sv_a2, s2do=1)
    call elements%add(-deps * B02, sv_v1, sv_a3, s2do=1)

    ! ==================== dCubic * dCubic ====================
    call elements%add(-B03, sv_v1, sv_a2, s1do=1, s2do=1)
    call elements%add(eps * B02, sv_v1, sv_a3, s1do=1, s2do=1)

    ! ==================== dQuadratic * Quadratic ====================
    call elements%add(-ic * eps * k3 * B01, sv_v2, sv_a1, s1do=1)
    call elements%add(ic * k2 * B01, sv_v3, sv_a1, s1do=1)

    ! ==================== dQuadratic * dCubic ====================
    call elements%add(ic * eps * B01, sv_v2, sv_a3, s1do=1, s2do=1)
    call elements%add(-ic * B01, sv_v3, sv_a2, s1do=1, s2do=1)
  end procedure add_regular_matrix_terms

end submodule smod_regular_matrix
