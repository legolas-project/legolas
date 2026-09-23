submodule (mod_matrix_manager) smod_conduction_matrix
  implicit none

contains

  module procedure add_conduction_matrix_terms
    real(dp) :: eps, deps
    real(dp) :: dT0
    real(dp) :: WVop
    real(dp) :: kappa_perp, dkappa_perp_drho, dkappa_perp_dT
    real(dp) :: gamma_1

    if (settings%physics%is_incompressible) return

    gamma_1 = settings%physics%get_gamma_1()
    eps = grid%get_eps(x)
    deps = grid%get_deps()
    dT0 = background%temperature%dT0(x)
    kappa_perp = physics%conduction%tcperp(x)
    dkappa_perp_drho = physics%conduction%dtcperpdrho(x)
    dkappa_perp_dT = physics%conduction%dtcperpdT(x)
    ! operators
    WVop = k2**2 / eps + eps * k3**2

    ! ==================== Quadratic * Quadratic ====================
    call elements%add(-gamma_1 * ic * WVop * kappa_perp / eps, sv_T1, sv_T1)
    ! ==================== dQuadratic * Quadratic ====================
    call elements%add(-ic * gamma_1 * dT0 * dkappa_perp_drho, sv_T1, sv_rho1, s1do=1)
    call elements%add( &
      gamma_1 * (ic * deps * kappa_perp / eps - ic * dT0 * dkappa_perp_dT), &
      sv_T1, &
      sv_T1, &
      s1do=1 &
    )
    ! ==================== dQuadratic * dQuadratic ====================
    call elements%add(-ic * gamma_1 * kappa_perp, sv_T1, sv_T1, s1do=1, s2do=1)

    if (settings%has_bfield()) then
      call add_conduction_matrix_terms_bfield( &
        x, settings, grid, background, physics, elements &
      )
    end if
  end procedure add_conduction_matrix_terms


  subroutine add_conduction_matrix_terms_bfield( &
    x, settings, grid, background, physics, elements &
  )
    real(dp), intent(in) :: x
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    type(background_t), intent(in) :: background
    type(physics_t), intent(in) :: physics
    type(matrix_elements_t), intent(inout) :: elements

    real(dp)  :: eps, deps
    real(dp)  :: dT0, ddT0
    real(dp)  :: B0, B01, B02, B03, dB02, dB03
    real(dp)  :: diffKp, Kp, Kp_plus, Kp_plusplus
    real(dp)  :: Fop_plus, dFop_plus, Gop_min
    real(dp)  :: kappa_para, dkappa_para_dT
    real(dp)  :: kappa_perp, dkappa_perp_drho, dkappa_perp_dT, dkappa_perp_dB2
    real(dp) :: gamma_1
    complex(dp) :: Fop_B01

    gamma_1 = settings%physics%get_gamma_1()
    ! grid variables
    eps = grid%get_eps(x)
    deps = grid%get_deps()
    ! temperature variables
    dT0 = background%temperature%dT0(x)
    ddT0 = background%temperature%ddT0(x)
    ! magnetic field variables
    B0 = background%magnetic%get_B0(x)
    B01 = background%magnetic%B01(x)
    B02 = background%magnetic%B02(x)
    dB02 = background%magnetic%dB02(x)
    B03 = background%magnetic%B03(x)
    dB03 = background%magnetic%dB03(x)
    ! parallel thermal conduction variables
    kappa_para = physics%conduction%tcpara(x)
    dkappa_para_dT = physics%conduction%dtcparadT(x)
    ! perpendicular thermal conduction variables
    kappa_perp = physics%conduction%tcperp(x)
    dkappa_perp_drho = physics%conduction%dtcperpdrho(x)
    dkappa_perp_dT = physics%conduction%dtcperpdT(x)
    dkappa_perp_dB2 = physics%conduction%dtcperpdB2(x)
    ! prefactors
    Kp = physics%conduction%get_tcprefactor(x)
    diffKp = physics%conduction%get_dtcprefactordr(x)
    Kp_plus = Kp + dkappa_perp_dB2
    Kp_plusplus = dkappa_perp_dB2 - (B01**2 * Kp_plus / B0**2)
    ! operators
    Fop_plus = k2 * B02 / eps + k3 * B03
    dFop_plus = (k2 / eps) * (dB02 - deps * B02 / eps) + k3 * dB03
    Gop_min = k3 * B02 - k2 * B03 / eps

    ! B01 modified F+ operator
    Fop_B01 = deps * ic * B01 / eps + Fop_plus

    ! ==================== Quadratic * Quadratic ====================
    call elements%add( &
      gamma_1 * dT0 * (B01 / B0**2) * dkappa_perp_drho * Fop_B01, sv_T1, sv_rho1 &
    )
    call elements%add( &
      gamma_1 * ( &
        B01 * Fop_plus * diffKp &
        - B01 * dT0 * (dkappa_para_dT - dkappa_perp_dT) * Fop_B01 / B0**2 &
        + Kp * ( &
          B01 * deps * (2.0d0 * deps * ic * B01 / eps + 3.0d0 * Fop_plus) / eps &
          + B01 * dFop_plus &
          - ic * Fop_plus**2 &
        ) &
      ), &
      sv_T1, &
      sv_T1 &
    )
    call elements%add( &
      2.0d0 * gamma_1 * eps * dT0 * Gop_min * Kp_plus * B01 * Fop_B01 / B0**2, &
      sv_T1, &
      sv_a1 &
    )
    ! ==================== dQuadratic * Quadratic ====================
    call elements%add( &
      ic * gamma_1 * dT0 * dkappa_perp_drho * B01**2 / B0**2, sv_T1, sv_rho1, s1do=1 &
    )
    call elements%add( &
      gamma_1 * ( &
        B01 * Kp * (2.0d0 * deps * ic * B01 / eps + 3.0d0 * Fop_plus) &
        - ic * dT0 * ( &
          B01**2 * dkappa_para_dT / B0**2 - dkappa_perp_dT * B01**2 / B0**2 &
        ) &
      ), &
      sv_T1, &
      sv_T1, &
      s1do=1 &
    )
    call elements%add( &
      -2.0d0 * ic * gamma_1 * eps * dT0 * Gop_min * Kp_plusplus, sv_T1, sv_a1, s1do=1 &
    )
    ! ==================== Quadratic * dQuadratic ====================
    call elements%add( &
      -2.0d0 * ic * gamma_1 * deps * B01**2 * Kp / eps, sv_T1, sv_T1, s2do=1 &
    )
    ! ==================== dQuadratic * dQuadratic ====================
    call elements%add( &
      -ic * gamma_1 * 2.0d0 * B01**2 * Kp, sv_T1, sv_T1, s1do=1, s2do=1 &
    )
    ! ==================== Quadratic * Cubic ====================
    call elements%add( &
      gamma_1 * k3 * ( &
        Kp * (B01 * ddT0 + ic * dT0 * Fop_plus) &
        + B01 * dT0 * (diffKp - 2.0d0 * ic * B01 * Kp_plus * Fop_B01 / B0**2) &
      ), &
      sv_T1, &
      sv_a2 &
    )
    call elements%add( &
      -gamma_1 * k2 * ( &
        Kp * (B01 * ddT0 + ic * dT0 * Fop_plus) &
        + B01 * dT0 * (diffKp - 2.0d0 * ic * B01 * Kp_plus * Fop_B01 / B0**2) &
      ), &
      sv_T1, &
      sv_a3 &
    )
    ! ==================== Quadratic * dCubic ====================
    call elements%add( &
      gamma_1 * dT0 * B01 * (2.0d0 * B03 * Kp_plus * Fop_B01 / B0**2 - k3 * Kp), &
      sv_T1, &
      sv_a2, &
      s2do=1 &
    )
    call elements%add( &
      gamma_1 * dT0 * B01 * ( &
        -2.0d0 * eps * B02 * Kp_plus * Fop_B01 / B0**2 + k2 * Kp &
      ), &
      sv_T1, &
      sv_a3, &
      s2do=1 &
    )
    ! ==================== dQuadratic * Cubic ====================
    call elements%add( &
      -2.0d0 * gamma_1 * dT0 * B01 * k3 * Kp_plusplus, sv_T1, sv_a2, s1do=1 &
    )
    call elements%add( &
      2.0d0 * gamma_1 * dT0 * B01 * k2 * Kp_plusplus, sv_T1, sv_a3, s1do=1 &
    )
    ! ==================== dQuadratic * dCubic ====================
    call elements%add( &
      -2.0d0 * ic * gamma_1 * dT0 * B03 * Kp_plusplus, sv_T1, sv_a2, s1do=1, s2do=1 &
    )
    call elements%add( &
      2.0d0 * ic * gamma_1 * dT0 * eps * B02 * Kp_plusplus, &
      sv_T1, &
      sv_a3, &
      s1do=1, &
      s2do=1 &
    )
  end subroutine add_conduction_matrix_terms_bfield

end submodule smod_conduction_matrix
