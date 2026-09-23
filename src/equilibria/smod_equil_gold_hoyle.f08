! =============================================================================
!> This submodule defines a Gold-Hoyle equilibrium in cylindrical geometry.
!! This equilibrium configuration models a filament with a uniform twist
!! such that all fieldlines perform an equal amount of turns around the cylinder axis.
!! The geometry can be overridden in the parfile.
!!
!! This equilibrium is taken from
!! _Van der Linden, R. A. M., Goossens, M. (1991).
!! "The thermal continuum in coronal loops: instability criteria and the influence of
!!  perpendicular thermal conduction." Solar physics, 134.2, 247-273._
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 1
!!     - <tt>k3</tt> = 1
!!     - <tt>cte_T0</tt> = 1 : used to set the temperature.
!!     - <tt>cte_rho0</tt> = 1 : used to set the density
!!     - <tt>alpha</tt> = 20 : used in the magnetic field components
!!     - cooling_curve = 'Rosner'
!!     - parallel thermal conduction, no perpendicular conduction
!!     
!!     and can all be changed in the parfile.
!! @endnote
!!
!! @note
!!     To reproduce the three profiles of the original paper you can supply
!!     one of the following normalisations:
!!     
!!     - cold plasma: <tt>unit_magneticfield</tt> = 10 Gauss,
!!                    <tt>unit_density</tt> = 1.6726e-15 g/cm3,
!!                    <tt>unit_length</tt> = 1.0e10 cm,
!!                    corresponding to a temperature of 5.7e4 K.
!!     - cool plasma: <tt>unit_magneticfield</tt> = 22.5 Gauss,
!!                    <tt>unit_density</tt> = 1.6726e-15 g/cm3,
!!                    <tt>unit_length</tt> = 1.0e10 cm,
!!                    corresponding to a temperature of 2.9e5 K (default).
!!     - hot plasma:  <tt>unit_magneticfield</tt> = 67 Gauss,
!!                    <tt>unit_density</tt> = 1.6726e-15 g/cm3,
!!                    <tt>unit_length</tt> = 1.0e11 cm,
!!                    corresponding to a temperature of 2.6e6 K.
!!     for a pure proton plasma (a=1, b=1).
!! @endnote
submodule (mod_equilibrium) smod_equil_gold_hoyle
  use mod_equilibrium_params, only: cte_T0, cte_rho0, alpha, eq_bool
  implicit none

contains

  module procedure gold_hoyle_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("cylindrical")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      call settings%physics%enable_cooling(cooling_curve="Rosner")
      call settings%physics%enable_heating(force_thermal_balance=.true.)
      call settings%physics%enable_parallel_conduction()

      k2 = 1.0_dp
      k3 = 1.0_dp
      cte_rho0 = 1.0_dp
      cte_T0 = 0.001_dp
      alpha = 20.0_dp

      ! cool plasma: B = 22.5 G, rho = 1.6726e-12 kg/m3, R = 1e8 m, T = 2.9e5 K
      call settings%units%set_units_from_density( &
        unit_density=1.6727e-15_dp, &
        unit_magneticfield=22.5_dp, &
        unit_length=1.0e10_dp, &
        a=1.0_dp, b=1.0_dp & ! pure proton plasma
      )
      ! hot plasma: B = 67.0 G, rho = 1.6726e-12 kg/m3, R = 1e9 m, T = 2.6e6 K
      ! call settings%units%set_units_from_density( &
      !   unit_density=1.6727e-15_dp, &
      !   unit_magneticfield=67.0_dp, &
      !   unit_length=1.0e11_dp, &
      !   He_abundance=0.0_dp &
      ! )
      ! cold plasma: B = 10.0 G, rho = 1.6726e-12 kg/m3, R = 1e8 m, T = 5.7e4 K
      ! call settings%units%set_units_from_density( &
      !   unit_density=1.6727e-15_dp, &
      !   unit_magneticfield=10.0_dp, &
      !   unit_length=1.0e10_dp, &
      !   He_abundance=0.0_dp &
      ! )
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

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_temperature_funcs(T0_func=T0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02)
    call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03)
  end procedure gold_hoyle_eq


  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function T0()
    T0 = cte_T0
  end function T0

  real(dp) function B02(r)
    real(dp) :: r
    B02 = alpha * r / (1.0_dp + alpha**2 * r**2)
  end function B02

  real(dp) function dB02(r)
    real(dp) :: r
    dB02 = -(alpha**3 * r**2 - alpha) / ( &
      alpha**4 * r**4 + 2.0_dp * alpha**2 * r**2 + 1.0_dp &
    )
  end function dB02

  real(dp) function B03(r)
    real(dp) :: r
    B03 = 1.0_dp / (1.0_dp + alpha**2 * r**2)
  end function B03

  real(dp) function dB03(r)
    real(dp) :: r
    dB03 = -2.0_dp * alpha**2 * r / (alpha**2 * r**2 + 1.0_dp)**2
  end function dB03

end submodule smod_equil_gold_hoyle
