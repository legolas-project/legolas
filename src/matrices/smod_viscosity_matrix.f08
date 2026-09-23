submodule (mod_matrix_manager) smod_viscosity_matrix
  implicit none

contains

  module procedure add_viscosity_matrix_terms
    real(dp)  :: eps, deps
    real(dp)  :: v01, dv01, ddv01
    real(dp)  :: v02, dv02
    real(dp)  :: v03, dv03, ddv03
    real(dp)  :: mu
    real(dp)  :: WVop
    real(dp) :: gamma_1
    logical :: viscous_heating, is_compressible

    gamma_1 = settings%physics%get_gamma_1()
    viscous_heating = settings%physics%viscosity%has_viscous_heating()
    is_compressible = .not. settings%physics%is_incompressible

    ! grid variables
    eps = grid%get_eps(x)
    deps = grid%get_deps()
    ! viscous heating variables
    v01 = background%velocity%v01(x)
    dv01 = background%velocity%dv01(x)
    ddv01 = background%velocity%ddv01(x)
    v02 = background%velocity%v02(x)
    dv02 = background%velocity%dv02(x)
    v03 = background%velocity%v03(x)
    dv03 = background%velocity%dv03(x)
    ddv03 = background%velocity%ddv03(x)
    ! viscosity value
    mu = settings%physics%viscosity%get_viscosity_value()
    ! operators
    WVop = k2**2 / eps + eps * k3**2

    ! ==================== Cubic * Cubic ====================
    call elements%add(-ic * mu * (deps / eps + WVop) / eps, sv_v1, sv_v1)
    ! ==================== Cubic * dCubic ====================
    call elements%add(-ic * mu * deps / (3.0d0 * eps), sv_v1, sv_v1, s2do=1)
    ! ==================== dCubic * Cubic ====================
    call elements%add(ic * mu * deps / eps, sv_v1, sv_v1, s1do=1)
    ! ==================== dCubic * dCubic ====================
    call elements%add(-4.0d0 * ic * mu / 3.0d0, sv_v1, sv_v1, s1do=1, s2do=1)
    ! ==================== Cubic * Quadratic ====================
    call elements%add(7.0d0 * deps * ic * mu * k2 / (3.0d0 * eps), sv_v1, sv_v2)
    call elements%add(ic * mu * deps * k3 / (3.0d0 * eps), sv_v1, sv_v3)
    ! ==================== dCubic * Quadratic ====================
    call elements%add(ic * mu * k2 / 3.0d0, sv_v1, sv_v2, s1do=1)
    call elements%add(ic * mu * k3 / 3.0d0, sv_v1, sv_v3, s1do=1)
    ! ==================== Quadratic * Cubic ====================
    call elements%add(ic * mu * deps * 2.0d0 * k2 / eps**2, sv_v2, sv_v1)
    ! ==================== Quadratic * dCubic ====================
    call elements%add(ic * mu * k2 / (3.0d0 * eps), sv_v2, sv_v1, s2do=1)
    call elements%add(ic * mu * k3 / 3.0d0, sv_v3, sv_v1, s2do=1)
    ! ==================== Quadratic * Quadratic ====================
    call elements%add( &
      -ic * mu * (deps / eps + 4.0d0 * k2**2 / (3.0d0 * eps) + eps * k3**2), &
      sv_v2, &
      sv_v2 &
    )
    call elements%add(-ic * mu * k2 * k3 / (3.0d0 * eps), sv_v2, sv_v3)
    call elements%add(-ic * mu * k2 * k3 / 3.0d0, sv_v3, sv_v2)
    call elements%add(-ic * mu * (k2**2 / eps**2 + 4.0d0 * k3**2 / 3.0d0), sv_v3, sv_v3)
    ! ==================== dQuadratic * dQuadratic ====================
    call elements%add(-ic * mu * eps, sv_v2, sv_v2, s1do=1, s2do=1)
    call elements%add(-ic * mu, sv_v3, sv_v3, s1do=1, s2do=1)
    ! ==================== dQuadratic * Quadratic ====================
    call elements%add(ic * mu * deps / eps, sv_v3, sv_v3, s1do=1)

    if (viscous_heating .and. is_compressible) then
      ! ==================== Quadratic * Cubic ====================
      call elements%add( &
        2.0d0 * gamma_1 * mu * ( &
          (deps**2 * v01 - ic * deps * k2 * v02) / eps**2 - deps * dv01 / eps - ddv01 &
        ), &
        sv_T1, &
        sv_v1 &
      )
      ! ==================== Quadratic * Quadratic ====================
      call elements%add( &
        2.0d0 * gamma_1 * mu * (deps**2 * ic * v02 - deps * k2 * v01) / eps, &
        sv_T1, &
        sv_v2 &
      )
      call elements%add( &
        -2.0d0 * ic * gamma_1 * mu * (deps * dv03 / eps + ddv03), sv_T1, sv_v3 &
      )
      ! ==================== dQuadratic * Quadratic ====================
      call elements%add(-2.0d0 * ic * gamma_1 * mu * dv03, sv_T1, sv_v3, s1do=1)
      ! ==================== dQuadratic * Cubic ====================
      call elements%add(-2.0d0 * gamma_1 * mu * dv01, sv_T1, sv_v1, s1do=1)
      ! ==================== Quadratic * dQuadratic ====================
      call elements%add(2.0d0 * ic * gamma_1 * mu * eps * dv02, sv_T1, sv_v2, s2do=1)
    end if
  end procedure add_viscosity_matrix_terms

end submodule smod_viscosity_matrix
