!> Submodule for user-defined equilibria.
!! Generated with GIMLI v0.4.
submodule (mod_equilibrium) smod_user_defined
  use mod_logging, only: logger
  use mod_equilibrium_params, only: cte_rho0, cte_T0
  implicit none

contains

  module procedure user_defined_eq
    if(settings%equilibrium%use_defaults) then
      call logger%error("No default values specified.")
    end if

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_velocity_2_funcs(v02_func=v02, dv02_func=dv02, ddv02_func=ddv02)
    call background%set_velocity_3_funcs(v03_func=v03, dv03_func=dv03, ddv03_func=ddv03)
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0, ddT0_func=ddT0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02, ddB02_func=ddB02)
    call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03, ddB03_func=ddB03)

  end procedure user_defined_eq

  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function drho0()
    drho0 = 0.0d0
  end function drho0

  real(dp) function v02()
    v02 = 0.0d0
  end function v02

  real(dp) function dv02()
    dv02 = 0.0d0
  end function dv02

  real(dp) function ddv02()
    ddv02 = 0.0d0
  end function ddv02

  real(dp) function v03()
    v03 = 0.0d0
  end function v03

  real(dp) function dv03()
    dv03 = 0.0d0
  end function dv03

  real(dp) function ddv03()
    ddv03 = 0.0d0
  end function ddv03

  real(dp) function T0()
    T0 = cte_T0
  end function T0

  real(dp) function dT0()
    dT0 = 0.0d0
  end function dT0

  real(dp) function ddT0()
    ddT0 = 0.0d0
  end function ddT0

end submodule
