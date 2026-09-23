! =============================================================================
!> This submodule defines a simple, homogeneous medium in Cartesian
!! geometry with a constant resistivity value. The geometry can be overridden using the parfile.
!!
!! This equilibrium is taken from section 14.3, p. 550 in
!! _Goedbloed, H., Keppens, R., & Poedts, S. (2019). Magnetohydrodynamics of Laboratory
!!  and Astrophysical Plasmas. Cambridge University Press._ [DOI](http://doi.org/10.1017/9781316403679).
!! @note
!!     Default values are given by
!!     
!!     - <tt>k2</tt> = 0
!!     - <tt>k3</tt> = 1
!!     - <tt>cte_rho0</tt> = 1 : used to set the density value.
!!     - <tt>cte_B02</tt> = 0 : used to set the By value.
!!     - <tt>cte_B03</tt> = 1 : used to set the Bz value.
!!     - <tt>beta</tt> = 0.25 : used to constrain the temperature value.
!!     - fixed resistivity value of 0.001
!!     
!!     and can all be changed in the parfile.
!! @endnote
submodule (mod_equilibrium) smod_equil_resistive_homo
  use mod_equilibrium_params, only: beta, cte_rho0, cte_B02, cte_B03
  implicit none

contains

  !> Sets the equilibrium.
  module procedure resistive_homo_eq
    if (settings%equilibrium%use_defaults) then ! LCOV_EXCL_START
      call settings%grid%set_geometry("Cartesian")
      call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
      call settings%physics%enable_resistivity(fixed_resistivity_value=0.001_dp)

      k2 = 0.0_dp
      k3 = 1.0_dp
      beta = 0.25_dp
      cte_rho0 = 1.0_dp
      cte_B02 = 0.0_dp
      cte_B03 = 1.0_dp
    end if ! LCOV_EXCL_STOP

    call background%set_density_funcs(rho0_func=rho0)
    call background%set_temperature_funcs(T0_func=T0)
    call background%set_magnetic_2_funcs(B02_func=B02)
    call background%set_magnetic_3_funcs(B03_func=B03)
  end procedure resistive_homo_eq


  real(dp) function rho0()
    rho0 = cte_rho0
  end function rho0

  real(dp) function T0()
    T0 = beta * B0()**2 / 2.0_dp
  end function T0

  real(dp) function B02()
    B02 = cte_B02
  end function B02

  real(dp) function B03()
    B03 = cte_B03
  end function B03

  real(dp) function B0()
    B0 = sqrt(B02()**2 + B03()**2)
  end function B0

end submodule smod_equil_resistive_homo
