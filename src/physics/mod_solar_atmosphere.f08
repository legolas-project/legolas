! =============================================================================
!> Module to set a realistic solar atmosphere, using tabulated density and
!! temperature profiles (see <tt>mod_atmosphere_curves</tt>), in Cartesian
!! geometries only.
module mod_solar_atmosphere
  use mod_global_variables, only: dp
  use mod_physical_constants, only: gsun_cgs, Rsun_cgs
  use mod_logging, only: logger, str
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_physics, only: physics_t
  use mod_function_utils, only: zero_func
  use mod_interpolation, only: lookup_table_value
  implicit none

  !> interpolated heights from atmosphere tables
  real(dp), allocatable :: h_interp(:)
  !> interpolated temperatures from atmosphere tables
  real(dp), allocatable :: T_interp(:)
  !> interpolated numberdensity from atmosphere tables
  real(dp), allocatable :: nh_interp(:)
  !> derivative of interpolated temperature with respect to height
  real(dp), allocatable :: dT_interp(:)
  !> amount of points used for interpolation, defaults to <tt>ncool</tt>
  integer   :: nbpoints
  !> integrated density profile
  real(dp), allocatable :: rho_values(:)

  real(dp) :: unit_length
  real(dp) :: unit_time
  real(dp) :: unit_magneticfield

  private

  public :: set_solar_atmosphere
  public :: solar_atmosphere_dealloc

contains


  !> Sets the density, temperature, gravity and magnetic field attributes of
  !! the respective fields to a realistic solar atmosphere profile.
  !! This routine first interpolates the temperature and numberdensity table at
  !! <tt>n_interp</tt> resolution, then solves the following ODE for the density:
  !! $$ \rho'(x) = -\frac{T'(x) + g(x)}{T(x)}\rho(x) -
  !!               \frac{B_{02}(x)B'_{02}(x) + B_{03}(x)B'_{03}(x)}{T(x)} $$
  !! using a fifth order Runge-Kutta method.
  !! If the optional argument <tt>save_to</tt> is provided then the density profiles
  !! are saved to that file, which can be loaded back in on subsequent runs through
  !! the optional argument <tt>load_from</tt>. The integration is done over the entire
  !! table, the curve is sampled on the Gaussian grid, meaning that grid variations
  !! can all use the same result.
  !! @warning
  !!     Throws an error if the geometry is not Cartesian.
  !! @endwarning
  subroutine set_solar_atmosphere(settings, background, physics, n_interp)
    use mod_integration, only: integrate_ode_rk45

    type(settings_t), intent(inout) :: settings
    type(background_t), intent(inout) :: background
    type(physics_t), intent(inout) :: physics
    !> points used for interpolation, defaults to 4000 if not present
    integer, intent(in), optional :: n_interp

    real(dp)  :: rhoinit

    unit_length = settings%units%get_unit_length()
    unit_time = settings%units%get_unit_time()
    unit_magneticfield = settings%units%get_unit_magneticfield()

    nbpoints = 4000
    if (present(n_interp)) nbpoints = n_interp

    if (settings%grid%get_geometry() /= "Cartesian") then
      call logger%error("solar atmosphere can only be set in Cartesian geometries!")
      return
    end if

    call logger%info("setting solar atmosphere...")
    allocate(rho_values(nbpoints))
    ! interpolate atmospheric tables
    allocate(h_interp(nbpoints), T_interp(nbpoints), dT_interp(nbpoints))
    allocate(nh_interp(nbpoints))
    ! curves are normalised on return
    call create_atmosphere_curves(settings)

    ! set initial density value (numberdensity = density in normalised units)
    rhoinit = nh_interp(1)
    call logger%info( &
      "solving equilibrium ODE for density (" // str(nbpoints) // " points)" &
    )
    call integrate_ode_rk45( &
      x0=h_interp(1), &
      x1=h_interp(nbpoints), &
      ax_func=ax_values, &
      bx_func=bx_values, &
      nbpoints=nbpoints, &
      yinit=rhoinit, &
      yvalues=rho_values &
    )

    call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0)
    call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02)
    call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03)

    call settings%physics%enable_gravity()
    call physics%set_gravity_funcs(g0_func=g0)
  end subroutine set_solar_atmosphere


  !> Interpolates the atmospheric tables to the desired resolution. The temperature
  !! derivative is obtained numerically.
  subroutine create_atmosphere_curves(settings)
    use mod_atmosphere_curves, only: h_alc7, T_alc7, nh_alc7
    use mod_interpolation, only: interpolate_table, get_numerical_derivative

    type(settings_t), intent(in) :: settings
    real(dp) :: unit_temperature, unit_numberdensity

    unit_temperature = settings%units%get_unit_temperature()
    unit_numberdensity = settings%units%get_unit_numberdensity()

    ! interpolate nh vs height
    call interpolate_table(nbpoints, h_alc7, nh_alc7, h_interp, nh_interp)
    ! interpolate T vs height
    call interpolate_table(nbpoints, h_alc7, T_alc7, h_interp, T_interp)

    ! rescale interpolated tables to actual values and normalise
    h_interp = h_interp * 1.0d5 / unit_length ! height is in km, so scale to cm first
    T_interp = T_interp / unit_temperature
    nh_interp = nh_interp / unit_numberdensity

    ! find temperature derivative
    call get_numerical_derivative(h_interp, T_interp, dT_interp)
    ! save these curves to a file if we're in debug mode
    if (logger%get_logging_level() >= 3) then ! LCOV_EXCL_START
      open( &
        unit=1002, &
        file="debug_atmocurves", &
        access="stream", &
        status="unknown", &
        action="write" &
      )
      write(1002) size(h_alc7)
      write(1002) h_alc7 * 1.0d5  ! save dimensionfull in cm
      write(1002) T_alc7
      write(1002) nh_alc7
      write(1002) size(h_interp)
      write(1002) h_interp * unit_length
      write(1002) T_interp * unit_temperature
      write(1002) nh_interp * unit_numberdensity
      write(1002) dT_interp * (unit_temperature / unit_length)
      close(1002)
      call logger%debug("atmo curves saved to file 'debug_atmocurves'")
    end if ! LCOV_EXCL_STOP
  end subroutine create_atmosphere_curves


  real(dp) function ax_values(x)
    real(dp), intent(in) :: x
    ax_values = -(dT0(x) + g0(x)) / T0(x)
  end function ax_values

  real(dp) function bx_values(x)
    real(dp), intent(in) :: x
    bx_values = -(B02() * dB02() + B03() * dB03()) / T0(x)
  end function bx_values

  real(dp) function B02()
    B02 = 0.0_dp
  end function B02

  real(dp) function dB02()
    dB02 = 0.0_dp
  end function dB02

  real(dp) function B03()
    b03 = 10.0_dp / unit_magneticfield
  end function B03

  real(dp) function dB03()
    db03 = 0.0_dp
  end function dB03

  !> Default profile for the solar gravitational field
  !! $$ g(x) = g_\odot \left(\frac{R_\odot}{(R_\odot + x)}\right)^2 $$
  real(dp) function g0(x)
    real(dp), intent(in)  :: x
    g0 = ( &
      gsun_cgs &
      * (Rsun_cgs / (Rsun_cgs + x * unit_length))**2 &
      / (unit_length / unit_time**2) &
    )
  end function g0

  real(dp) function rho0(x)
    real(dp), intent(in) :: x
    rho0 = lookup_table_value(x, h_interp, rho_values)
  end function rho0

  real(dp) function T0(x)
    real(dp), intent(in) :: x
    T0 = lookup_table_value(x, h_interp, T_interp)
  end function T0

  real(dp) function dT0(x)
    real(dp), intent(in) :: x
    dT0 = lookup_table_value(x, h_interp, dT_interp)
  end function dT0

  !> Sets density derivative using the differential equation to ensure force balance,
  !! instead of relying on numerical differentiation.
  real(dp) function drho0(x)
    real(dp), intent(in) :: x
    drho0 = ( &
      - (dT0(x) + g0(x)) * rho0(x) + (B02() * dB02() + B03() * dB03()) &
    ) / T0(x)
  end function drho0


  subroutine solar_atmosphere_dealloc()
    if (allocated(h_interp)) deallocate(h_interp)
    if (allocated(T_interp)) deallocate(T_interp)
    if (allocated(dT_interp)) deallocate(dT_interp)
    if (allocated(nh_interp)) deallocate(nh_interp)
    if (allocated(rho_values)) deallocate(rho_values)
  end subroutine solar_atmosphere_dealloc

end module mod_solar_atmosphere
