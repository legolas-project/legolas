submodule (mod_matrix_manager) smod_flow_matrix
  implicit none

contains

  module procedure add_flow_matrix_terms
    real(dp)  :: eps, deps
    real(dp)  :: rho, drho
    real(dp)  :: T0, dT0
    real(dp)  :: v01, dv01, drv01
    real(dp)  :: v02, dv02, drv02
    real(dp)  :: v03, dv03
    real(dp)  :: Vop
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
    ! flow variables
    v01 = background%velocity%v01(x)
    dv01 = background%velocity%dv01(x)
    drv01 = deps * v01 + eps * dv01
    v02 = background%velocity%v02(x)
    dv02 = background%velocity%dv02(x)
    drv02 = deps * v02 + eps * dv02
    v03 = background%velocity%v03(x)
    dv03 = background%velocity%dv03(x)
    Vop = k2 * v02 / eps + k3 * v03

    ! ==================== Quadratic * Quadratic ====================
    call elements%add(Vop - ic * dv01, sv_rho1, sv_rho1)
    call elements%add( &
      -drv02 * ic * v01 / eps, sv_v2, sv_rho1)
    call elements%add( &
      rho * (eps * Vop - ic * deps * v01), sv_v2, sv_v2)
    call elements%add(-ic * v01 * dv03, sv_v3, sv_rho1)
    call elements%add( &
      rho * (Vop + ic * dv01) + (deps * rho / eps + drho) * ic * v01, sv_v3, sv_v3 &
    )
    call elements%add(eps * Vop, sv_a1, sv_a1)

    ! ==================== Quadratic * dQuadratic ====================
    call elements%add(-ic * v01, sv_rho1, sv_rho1, s2do=1)
    call elements%add(-ic * eps * rho * v01, sv_v2, sv_v2, s2do=1)

    ! ==================== Cubic * Quadratic ====================
    call elements%add(v01 * dv01 - deps * v02**2 / eps, sv_v1, sv_rho1)
    call elements%add(-2.0d0 * deps * rho * v02, sv_v1, sv_v2)
    call elements%add(ic * v01 * k2, sv_a2, sv_a1)
    call elements%add(eps * k3 * ic * v01, sv_a3, sv_a1)

    ! ==================== Cubic * Cubic ====================
    call elements%add(rho * Vop + (deps * rho / eps + drho) * ic * v01, sv_v1, sv_v1)
    call elements%add(k3 * v03, sv_a2, sv_a2)
    call elements%add(-k2 * v03, sv_a2, sv_a3)
    call elements%add(-k3 * v02, sv_a3, sv_a2)
    call elements%add(k2 * v02, sv_a3, sv_a3)

    ! ==================== dCubic * Cubic ====================
    call elements%add(ic * rho * v01, sv_v1, sv_v1, s1do=1)

    ! ==================== Quadratic * Cubic ====================
    call elements%add(-drv02 * rho / eps, sv_v2, sv_v1)
    call elements%add(-rho * dv03, sv_v3, sv_v1)

    ! ==================== dQuadratic * Quadratic ====================
    call elements%add(ic * rho * v01, sv_v3, sv_v3, s1do=1)

    ! ==================== Quadratic * dCubic ====================
    call elements%add(-v02, sv_a1, sv_a2, s2do=1)
    call elements%add(-eps * v03, sv_a1, sv_a3, s2do=1)

    ! ==================== Cubic * dCubic ====================
    call elements%add(-ic * v01, sv_a2, sv_a2, s2do=1)
    call elements%add(-ic * eps * v01, sv_a3, sv_a3, s2do=1)

    if (.not. settings%physics%is_incompressible) then
      ! ==================== Quadratic * Quadratic ====================
      call elements%add(-ic * (v01 * dT0 + gamma_1 * drv01 * T0 / eps), sv_T1, sv_rho1)
      call elements%add( &
        rho * (Vop + ic * dv01 - ic * gamma_1 * drv01 / eps) &
        + ic * v01 * (deps * rho / eps + drho), &
        sv_T1, &
        sv_T1 &
      )
      ! ==================== dQuadratic * Quadratic ====================
      call elements%add(ic * rho * v01, sv_T1, sv_T1, s1do=1)
    end if
  end procedure add_flow_matrix_terms

end submodule smod_flow_matrix
