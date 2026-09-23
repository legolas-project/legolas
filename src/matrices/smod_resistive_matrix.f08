submodule (mod_matrix_manager) smod_resistive_matrix
  implicit none

contains

  module procedure add_resistive_matrix_terms
    real(dp)  :: eps, deps
    real(dp)  :: B02, dB02, drB02, ddB02
    real(dp)  :: B03, dB03, ddB03
    real(dp)  :: eta, detadT, deta
    real(dp)  :: WVop, Rop_pos, Rop_neg
    real(dp) :: gamma_1

    gamma_1 = settings%physics%get_gamma_1()

    ! grid variables
    eps = grid%get_eps(x)
    deps = grid%get_deps()
    ! magnetic field variables
    B02 = background%magnetic%B02(x)
    dB02 = background%magnetic%dB02(x)
    drB02 = deps * B02 + eps * dB02
    ddB02 = background%magnetic%ddB02(x)
    B03 = background%magnetic%B03(x)
    dB03 = background%magnetic%dB03(x)
    ddB03 = background%magnetic%ddB03(x)
    ! resistivity variables
    eta = physics%resistivity%eta(x)
    detadT = physics%resistivity%detadT(x)
    ! total derivative eta = deta_dr + dT0_dr * deta_dT
    deta = ( &
      physics%resistivity%detadr(x) &
      + (background%temperature%dT0(x) * detadT) &
    )

    WVop = k2**2 / eps + eps * k3**2
    Rop_pos = deps * eta / eps + deta
    Rop_neg = deps * eta / eps - deta

    ! ==================== Quadratic * Quadratic ====================
    call elements%add(-ic * eta * WVop, sv_a1, sv_a1)

    ! ==================== Quadratic * dCubic ====================
    call elements%add(ic * eta * k2 / eps, sv_a1, sv_a2, s2do=1)
    call elements%add(ic * eta * eps * k3, sv_a1, sv_a3, s2do=1)

    ! ==================== Cubic * Quadratic ====================
    call elements%add(ic * dB03 * detadT, sv_a2, sv_T1)
    call elements%add(ic * k2 * Rop_pos, sv_a2, sv_a1)
    call elements%add(-ic * drB02 * detadT / eps, sv_a3, sv_T1)
    call elements%add(ic * deta * eps * k3, sv_a3, sv_a1)

    ! ==================== dCubic * Quadratic ====================
    call elements%add(ic * eta * k2, sv_a2, sv_a1, s1do=1)
    call elements%add(ic * eta * eps * k3, sv_a3, sv_a1, s1do=1)

    ! ==================== Cubic * Cubic ====================
    call elements%add(-ic * eta * k3**2, sv_a2, sv_a2)
    call elements%add(ic * eta * k2 * k3, sv_a2, sv_a3)
    call elements%add(ic * eta * k2 * k3 / eps, sv_a3, sv_a2)
    call elements%add(-ic * eta * k2**2 / eps, sv_a3, sv_a3)

    ! ==================== Cubic * dCubic ====================
    call elements%add(-ic * Rop_pos, sv_a2, sv_a2, s2do=1)
    call elements%add(-ic * deta * eps, sv_a3, sv_a3, s2do=1)

    ! ==================== dCubic * dCubic ====================
    call elements%add(-ic * eta, sv_a2, sv_a2, s1do=1, s2do=1)
    call elements%add(-ic * eta * eps, sv_a3, sv_a3, s1do=1, s2do=1)

    if (.not. settings%physics%is_incompressible) then
      call add_compressible_resistive_terms()
    end if
  contains

    subroutine add_compressible_resistive_terms()
      ! ==================== Quadratic * Quadratic ====================
      call elements%add( &
        ic * gamma_1 * detadT * ((drB02 / eps)**2 + dB03**2), sv_T1, sv_T1 &
      )
      call elements%add( &
        2.0d0 * ic * gamma_1 * ( &
          k2 * (dB03 * Rop_pos + eta * ddB03) &
          + k3 * (drB02 * Rop_neg - eta * (2.0d0 * deps * dB02 + eps * ddB02)) &
        ), &
        sv_T1, &
        sv_a1 &
      )
      ! ==================== dQuadratic * Quadratic ====================
      call elements%add( &
        -2.0d0 * ic * gamma_1 * eta * (k3 * drB02 - k2 * dB03), &
        sv_T1, &
        sv_a1, &
        s1do=1 &
      )
      ! ==================== Quadratic * Cubic ====================
      call elements%add( &
        -2.0d0 * ic * gamma_1 * eta * (drB02 * k2 * k3 / eps**2 + k3**2 * dB03), &
        sv_T1, &
        sv_a2 &
      )
      call elements%add( &
        2.0d0 * ic * gamma_1 * eta * (drB02 * k2**2 / eps**2 + k2 * k3 * dB03), &
        sv_T1, &
        sv_a3 &
      )
      ! ==================== Quadratic * dCubic ====================
      call elements%add( &
        -2.0d0 * ic * gamma_1 * (dB03 * Rop_pos + ddB03 * eta), sv_T1, sv_a2, s2do=1 &
      )
      call elements%add( &
        -2.0d0 * ic * gamma_1 * ( &
          drB02 * Rop_neg - eta * (2.0d0 * deps * dB02 + eps * ddB02) &
        ), &
        sv_T1, &
        sv_a3, &
        s2do=1 &
      )
      ! ==================== dQuadratic * dCubic ====================
      call elements%add( &
        -2.0d0 * ic * gamma_1 * eta * dB03, sv_T1, sv_a2, s1do=1, s2do=1 &
      )
      call elements%add( &
        2.0d0 * ic * gamma_1 * drB02 * eta, sv_T1, sv_a3, s1do=1, s2do=1 &
      )
    end subroutine add_compressible_resistive_terms

  end procedure add_resistive_matrix_terms

end submodule smod_resistive_matrix
