module mod_inspections
  use mod_global_variables, only: dp
  use mod_types, only: density_type, temperature_type, bfield_type, velocity_type, &
                       gravity_type, cooling_type, conduction_type
  use mod_logging, only: log_message, dp_fmt, exp_fmt, int_fmt, char_log
  implicit none

  private

  public :: perform_sanity_checks
  public :: handle_spurious_eigenvalues

contains

  subroutine perform_sanity_checks(rho_field, T_field, B_field, v_field, grav_field, rc_field, kappa_field)
    type(density_type), intent(in)      :: rho_field
    type(temperature_type), intent(in)  :: T_field
    type(bfield_type), intent(in)       :: B_field
    type(velocity_type), intent(in)     :: v_field
    type(gravity_type), intent(in)      :: grav_field
    type(cooling_type), intent(in)      :: rc_field
    type(conduction_type), intent(in)   :: kappa_field

    call check_wavenumbers()
    call check_on_axis_values(B_field, v_field)
    call standard_equil_conditions(rho_field, T_field, B_field, v_field, grav_field)
    call nonadiab_equil_conditions(rho_field, T_field, rc_field, kappa_field)
  end subroutine perform_sanity_checks


  subroutine handle_spurious_eigenvalues(eigenvalues)
    use mod_global_variables, only: matrix_gridpts, remove_spurious_eigenvalues, nb_spurious_eigenvalues

    complex(dp), intent(inout)  :: eigenvalues(matrix_gridpts)
    integer                     :: i, idx
    complex(dp)                 :: replacement

    if (.not. remove_spurious_eigenvalues) then
      return
    end if

    call log_message("handling spurious eigenvalues", level='debug')

    ! For now, the largest real eigenvalues are set to a large number so they
    ! do not appear on the plots.
    ! Do NOT sort the eigenvalues, otherwise the order is messed up for the eigenfunctions
    replacement = (1.0d20, 0.0d0)

    do i = 1, nb_spurious_eigenvalues
      ! handle real values, take large values from boundaries into account
      idx = maxloc(real(eigenvalues), dim=1, mask=(real(eigenvalues) < 1.0d15))
      eigenvalues(idx) = replacement
      idx = minloc(real(eigenvalues), dim=1, mask=(real(eigenvalues) < 1.0d15))
      eigenvalues(idx) = replacement
    end do

    write(char_log, int_fmt) nb_spurious_eigenvalues
    call log_message("spurious eigenvalues removed on every side: " // adjustl(char_log), level='warning')
  end subroutine handle_spurious_eigenvalues


  subroutine check_wavenumbers()
    use mod_global_variables, only: geometry, dp_LIMIT
    use mod_equilibrium_params, only: k2

    integer   :: k2_int

    k2_int = int(k2)

    if (geometry == 'cylindrical') then
      ! in cylindrical geometry k2 should be an integer
      if (abs(k2_int - k2) > dp_LIMIT) then
        write(char_log, dp_fmt) k2
        call log_message("cylindrical geometry but k2 is not an integer! Value: " // trim(char_log), level='error')
      end if
    end if
  end subroutine check_wavenumbers


  subroutine check_on_axis_values(B_field, v_field)
    use mod_global_variables, only: geometry, x_start

    type(bfield_type), intent(in)   :: B_field
    type(velocity_type), intent(in) :: v_field

    real(dp)  :: on_axis_limit

    if (geometry == 'Cartesian') then
      return
    end if

    on_axis_limit = 1.0d-3
    if (x_start > on_axis_limit) then
      return
    end if

    if (abs(B_field % B02(1)) > on_axis_limit) then
      write(char_log, exp_fmt) B_field % B02(1)
      call log_message("B_theta non-zero on axis! Value: " // trim(char_log), level='warning')
    end if
    if (abs(B_field % d_B03_dr(1)) > on_axis_limit) then
      write(char_log, exp_fmt) B_field % d_B03_dr(1)
      call log_message("dBz/dr non-zero on axis! Value: " // trim(char_log), level='warning')
    end if
    if (abs(v_field % v02(1)) > on_axis_limit) then
      write(char_log, exp_fmt) v_field % v02(1)
      call log_message("v_theta non-zero on axis! Value: " // trim(char_log), level='warning')
    end if
    if (abs(v_field % d_v03_dr(1)) > on_axis_limit) then
      write(char_log, exp_fmt) v_field % d_v03_dr(1)
      call log_message("dvz_dr non-zero on axis! Value: " // trim(char_log), level='warning')
    end if
  end subroutine check_on_axis_values


  subroutine standard_equil_conditions(rho_field, T_field, B_field, v_field, grav_field)
    use mod_global_variables, only: gauss_gridpts, dp_LIMIT
    use mod_grid, only: eps_grid, d_eps_grid_dr

    type(density_type), intent(in)      :: rho_field
    type(temperature_type), intent(in)  :: T_field
    type(bfield_type), intent(in)       :: B_field
    type(velocity_type), intent(in)     :: v_field
    type(gravity_type), intent(in)      :: grav_field

    real(dp)  :: rho, drho, B02, dB02, B03, dB03, T0, dT0, grav, v02, v03
    real(dp)  :: eps, d_eps
    real(dp)  :: eq_cond(gauss_gridpts)
    integer   :: i

    do i = 1, gauss_gridpts
      rho = rho_field % rho0(i)
      drho = rho_field % d_rho0_dr(i)
      B02 = B_field % B02(i)
      B03 = B_field % B03(i)
      dB02 = B_field % d_B02_dr(i)
      dB03 = B_field % d_B03_dr(i)
      T0 = T_field % T0(i)
      dT0 = T_field % d_T0_dr(i)
      grav = grav_field % grav(i)
      v02 = v_field % v02(i)
      v03 = v_field % v03(i)
      eps = eps_grid(i)
      d_eps = d_eps_grid_dr(i)

      eq_cond(i) = drho * T0 + rho * dT0 + B02 * dB02 + B03 * dB03 + rho * grav - (d_eps/eps) * (rho * v02**2 - B02**2)
      if (abs(eq_cond(i)) > dp_LIMIT) then
        call log_message("standard equilibrium conditions not satisfied!", level='warning')
      end if
    end do
  end subroutine standard_equil_conditions


  subroutine nonadiab_equil_conditions(rho_field, T_field, rc_field, kappa_field)
    use mod_global_variables, only: gauss_gridpts, dp_LIMIT
    use mod_grid, only: grid_gauss, eps_grid, d_eps_grid_dr

    type(density_type), intent(in)      :: rho_field
    type(temperature_type), intent(in)  :: T_field
    type(cooling_type), intent(in)      :: rc_field
    type(conduction_type), intent(in)   :: kappa_field

    real(dp)  :: rho, dT0, ddT0, L0, kperp, dkperpdT
    real(dp)  :: eps, d_eps
    real(dp)  :: eq_cond(gauss_gridpts)
    integer   :: i

    do i = 1, gauss_gridpts-1
      rho = rho_field % rho0(i)
      dT0 = T_field % d_T0_dr(i)
      eps = eps_grid(i)
      d_eps = d_eps_grid_dr(i)
      kperp = kappa_field % kappa_perp(i)
      dkperpdT = kappa_field % d_kappa_perp_dT(i)
      L0 = rc_field % heat_loss(i)

      ! Do numerical differentiation for second T0 derivative, as it is only used here.
      ! This prevents having to calculate it every time in the submodules, 'approximately' equal here is fine.
      ddT0 = (T_field % d_T0_dr(i + 1) - T_field % d_T0_dr(i)) / (grid_gauss(i + 1) - grid_gauss(i))

      eq_cond(i) = d_eps / eps * kperp * dT0 + dkperpdT * dT0**2 + kperp * ddT0 - rho * L0
      if (abs(eq_cond(i)) > dp_LIMIT) then
        call log_message("non-adiabatic equilibrium conditions not satisfied!", level='warning')
      end if
    end do
  end subroutine nonadiab_equil_conditions

end module mod_inspections
