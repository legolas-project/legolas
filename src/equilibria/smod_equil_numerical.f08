!> Example submodule for numerically-defined equilibria.
submodule (mod_equilibrium) smod_numerical
  use mod_equilibrium_params, only: eq_bool
  use mod_arrays, only: import_equilibrium_data, lookup_equilibrium_value
  implicit none

contains

  ! LCOV_EXCL_START <exclude this file from code coverage>
  module procedure numerical_eq
    if (settings%equilibrium%use_defaults) then
      call settings%grid%set_geometry("Cartesian")
      ! To use the edges of the imported data as x_start/x_end, set eq_bool = .true.
      eq_bool = .true.

      k2 = 1.0_dp
      k3 = 0.0_dp
    end if

    ! All Legolas grid settings should be set before importing the equilibrium data.
    ! Note that non-equally spaced grids are not supported at this time due to the
    ! implementation of the numerical derivative, i.e. do not use set_custom_grid or
    ! set_spacing_function
    call import_equilibrium_data(settings, grid)

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_velocity_2_funcs(v02_func=v02, dv02_func=dv02, ddv02_func=ddv02)
    call background%set_velocity_3_funcs(v03_func=v03, dv03_func=dv03, ddv03_func=ddv03)
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0, ddT0_func=ddT0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02, ddB02_func=ddB02)
    call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03, ddB03_func=ddB03)
  end procedure numerical_eq

  real(dp) function rho0(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("rho0", x, 0, rho0)
  end function rho0

  real(dp) function drho0(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("rho0", x, 1, drho0)
  end function drho0

  real(dp) function v02(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("v02", x, 0, v02)
  end function v02

  real(dp) function dv02(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("v02", x, 1, dv02)
  end function dv02

  real(dp) function ddv02(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("v02", x, 2, ddv02)
  end function ddv02

  real(dp) function v03(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("v03", x, 0, v03)
  end function v03

  real(dp) function dv03(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("v03", x, 1, dv03)
  end function dv03

  real(dp) function ddv03(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("v03", x, 2, ddv03)
  end function ddv03

  real(dp) function T0(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("T0", x, 0, T0)
  end function T0

  real(dp) function dT0(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("T0", x, 1, dT0)
  end function dT0

  real(dp) function ddT0(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("T0", x, 2, ddT0)
  end function ddT0

  real(dp) function B02(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("B02", x, 0, B02)
  end function B02

  real(dp) function dB02(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("B02", x, 1, dB02)
  end function dB02

  real(dp) function ddB02(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("B02", x, 2, ddB02)
  end function ddB02

  real(dp) function B03(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("B03", x, 0, B03)
  end function B03

  real(dp) function dB03(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("B03", x, 1, dB03)
  end function dB03

  real(dp) function ddB03(x)
    real(dp), intent(in) :: x
    call lookup_equilibrium_value("B03", x, 2, ddB03)
  end function ddB03

  ! LCOV_EXCL_STOP
end submodule smod_numerical
