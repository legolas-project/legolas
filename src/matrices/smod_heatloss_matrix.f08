submodule (mod_matrix_manager) smod_heatloss_matrix
  implicit none

contains

  module procedure add_heatloss_matrix_terms
    real(dp) :: rho
    real(dp) :: Lrho, LT, L0
    real(dp) :: gamma_1

    if (settings%physics%is_incompressible) return

    gamma_1 = settings%physics%get_gamma_1()
    rho = background%density%rho0(x)
    Lrho = physics%heatloss%get_dLdrho(x)
    LT = physics%heatloss%get_dLdT(x)
    L0 = physics%heatloss%get_L0(x)

    ! ==================== Quadratic * Quadratic ====================
    call elements%add(-ic * gamma_1 * (L0 + rho * Lrho), sv_T1, sv_rho1)
    call elements%add(-ic * gamma_1 * rho * LT, sv_T1, sv_T1)
  end procedure add_heatloss_matrix_terms

end submodule smod_heatloss_matrix
