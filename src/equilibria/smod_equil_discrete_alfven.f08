! =============================================================================
!> This submodule defines an equilibrium in cylindrical geometry with
!! an axial current profile (\(\nu = 2\)), modelling a solar coronal loop
!! in which discrete Alfvén waves are present. The geometry can be overridden in the parfile.
!!
!! This equilibrium is taken from
!! _Keppens, Rony, Ronald AM Van Der Linden, and Marcel Goossens.
!! "Non-adiabatic discrete Alfven waves in coronal loops and prominences.",
!! Solar physics 144.2 (1993): 267-281_.
!!
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 1
!!     - <tt>k3</tt> = 0.05
!!     - <tt>j0</tt> = 0.125 : used to set the current.
!!     - <tt>delta</tt> = 0.2 : used in the density profile.
!!     - cooling_curve = 'Rosner'
!!     - parallel thermal conduction, no perpendicular conduction
!!     
!!     and normalisations given by
!!     
!!     - <tt>unit_density</tt> = 1.5e-15 gcm-3
!!     - <tt>unit_magneticfield</tt> = 50 Gauss
!!     - <tt>unit_length</tt> = 1e10 cm
!!     - pure proton plasma (a=1, b=1)
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_discrete_alfven
  use mod_equilibrium_params, only: j0, delta, eq_bool
  implicit none

  real(dp) :: x_end

contains

  module procedure discrete_alfven_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("cylindrical")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      call settings%physics%enable_cooling(cooling_curve="Rosner")
      call settings%physics%enable_heating(force_thermal_balance=.true.)
      call settings%physics%enable_parallel_conduction()
      call settings%units%set_units_from_density( &
        unit_density=1.5e-15_dp, &
        unit_magneticfield=50.0_dp, &
        unit_length=1.0e10_dp, &
        a=1.0_dp, b=1.0_dp & ! pure proton plasma
      )

      j0 = 0.125_dp
      delta = 0.2_dp   ! d parameter in density prescription
      k2 = 1.0_dp
      k3 = 0.05_dp
    end if ! LCOV_EXCL_STOP

    ! When eq_bool is true, override the default definitions of a and b that are based
    ! on He abundance. This is needed to reproduce the results of the original paper.
    if (eq_bool) then
      call settings%units%set_units_from_density( &
        unit_density=settings%units%get_unit_density(), &
        unit_magneticfield=settings%units%get_unit_magneticfield(), &
        unit_length=settings%units%get_unit_length(), &
        a=1.0_dp, b=1.0_dp & ! pure proton plasma
      )
    end if

    x_end = settings%grid%get_grid_end()

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02)
    call background%set_magnetic_3_funcs(B03_func=B03)
  end procedure discrete_alfven_eq


  real(dp) function rho0(r)
    real(dp), intent(in) :: r
    rho0 = 1.0_dp - (1.0_dp - delta) * (r / x_end)**2
  end function rho0

  real(dp) function drho0(r)
    real(dp), intent(in) :: r
    drho0 = 2.0_dp * r * (delta - 1.0_dp) / x_end**2
  end function drho0


  real(dp) function T0(r)
    real(dp), intent(in) :: r
    T0 = p0(r) / rho0(r)
  end function T0

  real(dp) function dT0(r)
    real(dp), intent(in) :: r
    dT0 = (dp0(r) * rho0(r) - drho0(r) * p0(r)) / rho0(r)**2
  end function dT0

  real(dp) function B02(r)
    real(dp), intent(in) :: r
    B02 = j0 * r * (r**4 - 3.0_dp * r**2 + 3.0_dp) / 6.0_dp
  end function B02

  real(dp) function dB02(r)
    real(dp), intent(in) :: r
    dB02 = j0 * (5.0_dp * r**4 - 9.0_dp * r**2 + 3.0_dp) / 6.0_dp
  end function dB02

  real(dp) function B03()
    B03 = 1.0_dp
  end function B03

  real(dp) function p0(r)
    real(dp), intent(in) :: r
    p0 = (j0**2 / 720.0_dp) * ( &
      12.0_dp * (x_end**10 - r**10) &
      - 75.0_dp * (x_end**8 - r**8) &
      + 200.0_dp * (x_end**6 - r**6) &
      - 270.0_dp * (x_end**4 - r**4) &
      + 180.0_dp * (x_end**2 - r**2) &
    )
  end function p0


  real(dp) function dp0(r)
    real(dp), intent(in) :: r
    dp0 = j0**2 * r * ( &
      -r**8 + 5.0_dp * r**6 - 10.0_dp * r**4 + 9.0_dp * r**2 - 3.0_dp &
    ) / 6.0_dp
  end function dp0

end submodule smod_equil_discrete_alfven
