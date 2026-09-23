submodule (mod_matrix_manager) smod_hall_matrix
  implicit none

contains

  module procedure add_hall_bmatrix_terms
    real(dp)  :: eps, deps
    real(dp)  :: rho, drho
    real(dp)  :: eta_H, eta_e
    real(dp)  :: WVop

    eps = grid%get_eps(x)
    deps = grid%get_deps()
    rho = background%density%rho0(x)
    drho = background%density%drho0(x)
    eta_H = physics%hall%hallfactor(x)
    eta_e = physics%hall%inertiafactor(x)
    WVop = k2**2 / eps + eps * k3**2

    ! ==================== Quadratic * Cubic ====================
    call elements%add(eta_H, sv_a1, sv_v1)
    ! ==================== Cubic * Quadratic ====================
    call elements%add(eta_H * eps, sv_a2, sv_v2)
    call elements%add(eta_H, sv_a3, sv_v3)

    if (settings%physics%hall%has_electron_inertia()) then
      ! ==================== Quadratic * Quadratic ====================
      call elements%add(eta_e * WVop / rho, sv_a1, sv_a1)
      ! ==================== Quadratic * dCubic ====================
      call elements%add(-eta_e * k2 / (eps * rho), sv_a1, sv_a2, s2do=1)
      call elements%add(-eta_e * eps * k3 / rho, sv_a1, sv_a3, s2do=1)
      ! ==================== Cubic * Quadratic ====================
      call elements%add( &
        -eta_e * k2 * (deps / (eps * rho) - drho / rho**2), sv_a2, sv_a1 &
      )
      call elements%add(eta_e * drho * eps * k3 / rho**2, sv_a3, sv_a1)
      ! ==================== dCubic * Quadratic ====================
      call elements%add(-eta_e * k2 / rho, sv_a2, sv_a1, s1do=1)
      call elements%add(-eta_e * eps * k3 / rho, sv_a3, sv_a1, s1do=1)
      ! ==================== Cubic * Cubic ====================
      call elements%add(eta_e * k3**2 / rho, sv_a2, sv_a2)
      call elements%add(-eta_e * k2 * k3 / rho, sv_a2, sv_a3)
      call elements%add(-eta_e * k2 * k3 / (eps * rho), sv_a3, sv_a2)
      call elements%add(eta_e * k2**2 / (eps * rho), sv_a3, sv_a3)
      ! ==================== Cubic * dCubic ====================
      call elements%add( &
        eta_e * (deps / (eps * rho) - drho / rho**2), sv_a2, sv_a2, s2do=1 &
      )
      call elements%add(-eta_e * eps * drho / rho**2, sv_a3, sv_a3, s2do=1)
      ! ==================== dCubic * dCubic ====================
      call elements%add(eta_e / rho, sv_a2, sv_a2, s1do=1, s2do=1)
      call elements%add(eta_e * eps / rho, sv_a3, sv_a3, s1do=1, s2do=1)
    end if
  end procedure add_hall_bmatrix_terms


  module procedure add_hall_matrix_terms
    real(dp)  :: eps, deps
    real(dp)  :: v01, v02, v03, dv01, dv02, dv03, ddv01, ddv02, ddv03
    real(dp)  :: rho, drho, T0, dT0, B01, B02, B03, dB02, dB03
    real(dp)  :: drB02, dB02_r, Fop_plus, Gop_plus, Gop_min, WVop
    real(dp)  :: eta_H, mu, efrac
    logical :: has_viscosity

    eps = grid%get_eps(x)
    deps = grid%get_deps()

    v01 = background%velocity%v01(x)
    v02 = background%velocity%v02(x)
    v03 = background%velocity%v03(x)
    dv01 = background%velocity%dv01(x)
    dv02 = background%velocity%dv02(x)
    dv03 = background%velocity%dv03(x)
    ddv01 = background%velocity%ddv01(x)
    ddv02 = background%velocity%ddv02(x)
    ddv03 = background%velocity%ddv03(x)

    rho = background%density%rho0(x)
    drho = background%density%drho0(x)
    T0 = background%temperature%T0(x)
    dT0 = background%temperature%dT0(x)

    B01 = background%magnetic%B01(x)
    B02 = background%magnetic%B02(x)
    dB02 = background%magnetic%dB02(x)
    B03 = background%magnetic%B03(x)
    dB03 = background%magnetic%dB03(x)

    ! Calculate derivatives eps*B02, B02/eps
    drB02 = deps * B02 + eps * dB02
    dB02_r = dB02 / eps - B02 * deps / eps**2

    Fop_plus = k2 * B02 / eps + k3 * B03
    Gop_plus = k3 * B02 + k2 * B03 / eps
    Gop_min = k3 * B02 - k2 * B03 / eps
    WVop = k2**2 / eps + eps * k3**2

    eta_H = physics%hall%hallfactor(x)
    mu = settings%physics%viscosity%get_viscosity_value()
    efrac = settings%physics%hall%get_electron_fraction()

    has_viscosity = settings%physics%viscosity%is_enabled()

    ! Hall by substitution of the momentum equation
    ! ==================== Quadratic * Cubic ====================
    call elements%add(eta_H * (k2 * v02 / eps + k3 * v03), sv_a1, sv_v1)
    ! ==================== Quadratic * Quadratic ====================
    call elements%add(-eta_H * (1.0d0 - efrac) * dT0 / rho, sv_a1, sv_rho1)
    call elements%add(-2.0d0 * eta_H * deps * v02, sv_a1, sv_v2)
    call elements%add(eta_H * (1.0d0 - efrac) * drho / rho, sv_a1, sv_T1)
    ! ==================== Cubic * Cubic ====================
    call elements%add(-eta_H * (dv02 - v02 * deps / eps), sv_a2, sv_v1)
    call elements%add(-eta_H * dv03, sv_a3, sv_v1)
    ! ==================== Cubic * Quadratic ====================
    call elements%add(eta_H * (k2 * v02 + eps * k3 * v03), sv_a2, sv_v2)
    call elements%add(eta_H * (k2 * v02 / eps + k3 * v03), sv_a3, sv_v3)

    ! elements below are only relevant when viscosity is included
    if (.not. has_viscosity) return

    ! ==================== Quadratic * Cubic ====================
    call elements%add( &
      -eta_H * ic * mu * ( &
        (drho / rho + 1.0d0 / eps) * deps / eps + (k2 / eps)**2 + k3**2 &
      ) / rho, &
      sv_a1, &
      sv_v1 &
    )
    ! ==================== Quadratic * dCubic ====================
    call elements%add( &
      eta_H * ic * mu * (4.0d0 * drho / rho - deps / eps) / (3.0d0 * rho), &
      sv_a1, &
      sv_v1, &
      s2do=1 &
    )
    ! ==================== dQuadratic * Cubic ====================
    call elements%add(eta_H * ic * mu * deps / (eps * rho), sv_a1, sv_v1, s1do=1)
    ! ==================== dQuadratic * dCubic ====================
    call elements%add( &
      -4.0d0 * eta_H * ic * mu / (3.0d0 * rho), sv_a1, sv_v1, s1do=1, s2do=1 &
    )
    ! ==================== Quadratic * Quadratic ====================
    call elements%add( &
      eta_H * mu * 4.0d0 * (ddv01 + deps * (dv01 - v01 / eps) / eps) &
      / (3.0d0 * rho**2), &
      sv_a1, &
      sv_rho1 &
    )
    call elements%add( &
      7.0d0 * eta_H * ic * mu * deps * k2 / (3.0d0 * eps * rho), sv_a1, sv_v2 &
    )
    call elements%add(eta_H * ic * mu * k3 * deps / (3.0d0 * eps * rho), sv_a1, sv_v3)
    ! ==================== Quadratic * dQuadratic ====================
    call elements%add(-eta_H * ic * mu * k2 / (3.0d0 * rho), sv_a1, sv_v2, s2do=1)
    call elements%add(-eta_H * ic * mu * k3 / (3.0d0 * rho), sv_a1, sv_v3, s2do=1)
    ! ==================== Cubic * Cubic ====================
    call elements%add( &
      2.0d0 * eta_H * ic * mu * k2 * deps / (eps**2 * rho), sv_a2, sv_v1 &
    )
    ! ==================== Cubic * dCubic ====================
    call elements%add(eta_H * ic * mu * k2 / (3.0d0 * eps * rho), sv_a2, sv_v1, s2do=1)
    call elements%add(eta_H * ic * mu * k3 / (3.0d0 * rho), sv_a3, sv_v1, s2do=1)
    ! ==================== Cubic * Quadratic ====================
    call elements%add( &
      -ic * eta_H * mu * (ddv02 + deps * (dv02 - v02 / eps) / eps) / rho**2, &
      sv_a2, &
      sv_rho1 &
    )
    call elements%add( &
      -eta_H * ic * mu * ( &
        4.0d0 * k2**2 / (3.0d0 * eps) + eps * k3**2 + deps / eps &
      ) / rho, &
      sv_a2, &
      sv_v2 &
    )
    call elements%add(-eta_H * ic * mu * k2 * k3 / (3.0d0 * eps * rho), sv_a2, sv_v3)
    call elements%add( &
      -ic * eta_H * mu * (ddv03 + deps * dv03 / eps) / rho**2, sv_a3, sv_rho1 &
    )
    call elements%add(-eta_H * ic * mu * k2 * k3 / (3.0d0 * rho), sv_a3, sv_v2)
    call elements%add( &
      -eta_H * ic * mu * ( &
        (k2 / eps)**2 + 4.0d0 * k3**2 / 3.0d0 + drho * deps / (eps * rho) &
      ) / rho, &
      sv_a3, &
      sv_v3 &
    )
    ! ==================== Cubic * dQuadratic ====================
    call elements%add(eta_H * ic * mu * eps * drho / rho**2, sv_a2, sv_v2, s2do=1)
    call elements%add(eta_H * ic * mu * drho / rho**2, sv_a3, sv_v3, s2do=1)
    ! ==================== dCubic * dQuadratic ====================
    call elements%add(-eta_H * ic * mu * eps / rho, sv_a2, sv_v2, s1do=1, s2do=1)
    call elements%add(-eta_H * ic * mu / rho, sv_a3, sv_v3, s1do=1, s2do=1)
    ! ==================== dCubic * Quadratic ====================
    call elements%add(eta_H * ic * mu * deps / (eps * rho), sv_a3, sv_v3, s1do=1)
  end procedure add_hall_matrix_terms

end submodule smod_hall_matrix
