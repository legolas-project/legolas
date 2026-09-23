! =============================================================================
!> This submodule defines an equilibrium containing magnetothermal instabilities
!! in a cylindrical geometry. The geometry can be overridden in the parfile.
!!
!! This equilibrium is taken from
!! _Van der Linden, R. A. M., Goossens, M., & Hood, A. W. (1992).
!!  The relevance of the ballooning approximation for magnetic, thermal,
!!  and coalesced magnetothermal instabilities. Solar physics, 140(2), 317-342_.
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 0
!!     - <tt>k3</tt> = 1
!!     - <tt>cte_T0</tt> = 1 : used to set the temperature.
!!     - cooling_curve = 'Rosner'
!!     - parallel thermal conduction, no perpendicular conduction
!!     
!!     and normalisations given by
!!     
!!     - <tt>unit_temperature</tt> = 2.6e6 K
!!     - <tt>unit_magneticfield</tt> = 10 Gauss
!!     - <tt>unit_length</tt> = 1e8 cm
!!     - pure proton plasma (a=1, b=1)
!!     
!!     and can all be changed in the parfile.
!! @endnote
!!
!! @note
!!     The default setup handles _CASE I_ in the original paper. For _CASE II_,
!!     you can set k2 = 10. To reproduce the thermal continuum plot,
!!     set <tt>unit_length</tt> = 2.44e9 cm in _CASE I_.
!! @endnote
submodule(mod_equilibrium) smod_equil_magnetothermal_instabilities
  use mod_equilibrium_params, only: cte_T0, eq_bool
  implicit none

contains

  !> Sets the equilibrium
  module procedure magnetothermal_instability_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("cylindrical")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      call settings%physics%enable_cooling(cooling_curve="Rosner")
      call settings%physics%enable_heating(force_thermal_balance=.true.)
      call settings%physics%enable_parallel_conduction()
      call settings%units%set_units_from_temperature( &
        unit_temperature=2.6e6_dp, &
        unit_magneticfield=10.0_dp, &
        unit_length=1.0e8_dp, &
        a=1.0_dp, b=1.0_dp & ! pure proton plasma
      )

      cte_T0 = 1.0_dp
      k2 = 0.0_dp
      k3 = 1.0_dp
    end if ! LCOV_EXCL_STOP

    ! When eq_bool is true, override the default definitions of a and b that are based
    ! on He abundance. This is needed to reproduce the results of the original paper.
    if (eq_bool) then
      call settings%units%set_units_from_temperature( &
        unit_temperature=settings%units%get_unit_temperature(), &
        unit_magneticfield=settings%units%get_unit_magneticfield(), &
        unit_length=settings%units%get_unit_length(), &
        a=1.0_dp, b=1.0_dp & ! pure proton plasma
      )
    end if

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_temperature_funcs(T0_func=T0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02)
  end procedure magnetothermal_instability_eq


  real(dp) function p0(r)
    real(dp), intent(in) :: r
    p0 = 1.0_dp / (2.0_dp * (1.0_dp + r**2)**2)
  end function p0

  real(dp) function rho0(r)
    real(dp), intent(in) :: r
    rho0 = p0(r) / cte_T0
  end function rho0

  real(dp) function drho0(r)
    real(dp), intent(in) :: r
    drho0 = -2.0_dp * r / (cte_T0 * (r**2 + 1.0_dp)**3)
  end function drho0

  real(dp) function T0()
    T0 = cte_T0
  end function T0

  real(dp) function B02(r)
    real(dp), intent(in) :: r
    B02 = r / (1.0_dp + r**2)
  end function B02

  real(dp) function dB02(r)
    real(dp), intent(in) :: r
    dB02 = (1.0_dp - r**2) / (r**4 + 2.0_dp * r**2 + 1.0_dp)
  end function dB02

end submodule smod_equil_magnetothermal_instabilities
