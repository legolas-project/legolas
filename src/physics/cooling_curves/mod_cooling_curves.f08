module mod_cooling_curves
  use mod_global_variables, only: dp, str_len
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_logging, only: logger
  use mod_interpolation, only: lookup_table_value, get_numerical_derivative

  use mod_cooling_curve_names
  use mod_radloss_tables

  implicit none

  private

  real(dp), allocatable :: curve_T(:)
  real(dp), allocatable :: curve_lambda(:)
  real(dp), allocatable :: curve_dlambdadT(:)

  real(dp) :: minT, maxT

  public :: interpolate_cooling_curves
  public :: is_valid_cooling_curve
  public :: get_Rosner_lambdaT, get_Rosner_dlambdadT
  public :: get_interpolated_lambdaT, get_interpolated_dlambdadT
  public :: get_cooling_table
  public :: deallocate_cooling_curves

contains

  subroutine interpolate_cooling_curves(settings)
    use mod_interpolation, only: interpolate_table, get_numerical_derivative

    type(settings_t), intent(in) :: settings
    real(dp), allocatable :: table_T(:)
    real(dp), allocatable :: table_lambda(:)
    real(dp) :: unit_temperature, unit_lambdaT, unit_dlambdadT
    integer :: ncool

    call get_cooling_table( &
      name=settings%physics%cooling%get_cooling_curve(), &
      table_T=table_T, &
      table_lambda=table_lambda &
    )
    unit_temperature = settings%units%get_unit_temperature()
    unit_lambdaT = settings%units%get_unit_lambdaT()
    unit_dlambdadT = unit_lambdaT / unit_temperature

    !> @note
    !!     The cooling tables contain dimensionfull values on a logarithmic scale.
    !!     To avoid resampling the table on an unequally spaced temperature grid by doing
    !!     10**T, we interpolate the logarithmic table values on an equally spaced
    !!     T grid in log scale, so we get log10(lambda(T)) and log10(T) values.
    !! @endnote
    ncool = settings%physics%cooling%get_interpolation_points()
    allocate(curve_T(ncool), curve_lambda(ncool), curve_dlambdadT(ncool))
    call interpolate_table( &
      n_interp=ncool, &
      x_table=table_T, &
      y_table=table_lambda, &
      x_interp=curve_T, &
      y_interp=curve_lambda &
    )
    ! rescale to L(T) and normalise
    curve_lambda = 10.0_dp**curve_lambda / unit_lambdaT
    ! normalise logT values: log10(T/Tunit) = log10(T) - log10(Tunit)
    curve_T = curve_T - log10(unit_temperature)
    ! get dlambda(T)/dlogT, hence chain rule: dL(T)/dlogT = (dL(T)/dT) * T
    call get_numerical_derivative(x=curve_T, y=curve_lambda, dy=curve_dlambdadT)
    ! rescale to T (already normalised)
    curve_T = 10.0_dp**curve_T
    ! finally, dL(T)/dT = dL(T)/dlogT * (1 / T)
    curve_dlambdadT = curve_dlambdadT / curve_T

    ! set min/max temperature
    minT = minval(curve_T)
    maxT = maxval(curve_T)

    ! LCOV_EXCL_START
    ! save these curves to a file if we're in debug mode
    if (logger%get_logging_level() >= 3) then
      open( &
        unit=1002, &
        file="debug_coolingcurves", &
        access="stream", &
        status="unknown", &
        action="write" &
      )
      write(1002) size(table_T)
      write(1002) 10.0d0**table_T
      write(1002) 10.0d0**table_lambda
      write(1002) size(curve_T)
      write(1002) curve_T * unit_temperature
      write(1002) curve_lambda * unit_lambdaT
      write(1002) curve_dlambdadT * unit_dlambdadT
      close(1002)
      call logger%debug("cooling curves saved to file 'debug_coolingcurves'")
    end if
    ! LCOV_EXCL_STOP

    deallocate(table_T)
    deallocate(table_lambda)
  end subroutine interpolate_cooling_curves


  pure integer function get_Rosner_index(logT0)
    !> dimensionfull log10(T0) value
    real(dp), intent(in) :: logT0
    integer :: j

    get_Rosner_index = 1
    do j = 1, size(logT_Rosner)
      if (logT0 < logT_Rosner(j)) then
        get_Rosner_index = j - 1
        exit
      end if
    end do
  end function get_Rosner_index


  real(dp) function get_Rosner_lambdaT(x, settings, background)
    real(dp), intent(in) :: x
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background
    real(dp) :: logT0, logxi, alpha, logTmax
    real(dp) :: unit_temperature, unit_lambdaT
    integer :: idx

    unit_temperature = settings%units%get_unit_temperature()
    unit_lambdaT = settings%units%get_unit_lambdaT()
    logT0 = log10(background%temperature%T0(x) * unit_temperature)

    if (logT0 < logT_Rosner(1)) then
      get_Rosner_lambdaT = 0.0_dp
    else if (logT0 > logT_Rosner(n_Rosner+1)) then
      logxi = logxi_Rosner(n_Rosner)
      alpha = alpha_Rosner(n_Rosner)
      logTmax = logT_Rosner(n_Rosner+1)
      ! lambdaT = xi * T**alpha, so log10(lambdaT) = log10(xi) + alpha * log10(T)
      get_Rosner_lambdaT = 10.0_dp**(logxi + alpha * logTmax) / unit_lambdaT
      get_Rosner_lambdaT = get_Rosner_lambdaT * sqrt(10**(logT0-logTmax))
    else
      idx = get_Rosner_index(logT0)

      logxi = logxi_Rosner(idx)
      alpha = alpha_Rosner(idx)
      ! lambdaT = xi * T**alpha, so log10(lambdaT) = log10(xi) + alpha * log10(T)
      get_Rosner_lambdaT = 10.0_dp**(logxi + alpha * logT0) / unit_lambdaT
    end if
  end function get_Rosner_lambdaT


  real(dp) function get_Rosner_dlambdadT(x, settings, background)
    real(dp), intent(in) :: x
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background
    real(dp) :: logT0, logxi, alpha, logTmax
    real(dp) :: unit_temperature, unit_lambdaT
    integer :: idx

    unit_temperature = settings%units%get_unit_temperature()
    unit_lambdaT = settings%units%get_unit_lambdaT()
    logT0 = log10(background%temperature%T0(x) * unit_temperature)

    if (logT0 < logT_Rosner(1)) then
      get_Rosner_dlambdadT = 0.0_dp
    else if (logT0 > logT_Rosner(n_Rosner)) then
      logxi = logxi_Rosner(n_Rosner)
      alpha = alpha_Rosner(n_Rosner)
      logTmax = logT_Rosner(n_Rosner+1)
      ! dlambdadT = alpha * xi * T**(alpha - 1), and so
      !           = alpha * 10**(logxi + (alpha - 1) * logT0)
      get_Rosner_dlambdadT = ( &
        10.0_dp**(logxi + alpha * logTmax) &
      ) / (unit_lambdaT / unit_temperature)
      get_Rosner_dlambdadT = ( &
        0.5_dp * get_Rosner_dlambdadT &
      ) / sqrt(10**(logT0+logTmax))
    else
      idx = get_Rosner_index(logT0)

      logxi = logxi_Rosner(idx)
      alpha = alpha_Rosner(idx)
      ! dlambdadT = alpha * xi * T**(alpha - 1), and so
      !           = alpha * 10**(logxi + (alpha - 1) * logT0)
      get_Rosner_dlambdadT = ( &
        alpha * 10.0_dp**(logxi + (alpha - 1.0_dp) * logT0) &
      ) / (unit_lambdaT / unit_temperature)
    end if
  end function get_Rosner_dlambdadT


  real(dp) function get_interpolated_lambdaT(x, settings, background)
    real(dp), intent(in) :: x
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background
    real(dp) :: lambdaT, T0
    integer :: pts

    pts = size(curve_T)
    T0 = background%temperature%T0(x)
    if (T0 <= minT) then
      lambdaT = 0.0_dp
    else if (T0 >= maxT) then
      lambdaT = curve_lambda(pts) * sqrt(T0 / maxT)
    else
      lambdaT = lookup_table_value(T0, curve_T, curve_lambda)
    end if
    get_interpolated_lambdaT = lambdaT
  end function get_interpolated_lambdaT


  real(dp) function get_interpolated_dlambdadT(x, settings, background)
    real(dp), intent(in) :: x
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background
    real(dp) :: dlambdadT, T0
    integer :: pts

    pts = size(curve_T)
    T0 = background%temperature%T0(x)
    if (T0 <= minT) then
      dlambdadT = 0.0_dp
    else if (T0 >= maxT) then
      dlambdadT = 0.5_dp * curve_lambda(pts) / sqrt(T0 * maxT)
    else
      dlambdadT = lookup_table_value(T0, curve_T, curve_dlambdadT)
    end if
    get_interpolated_dlambdadT = dlambdadT
  end function get_interpolated_dlambdadT


  logical function is_valid_cooling_curve(name)
    character(len=*), intent(in) :: name

    is_valid_cooling_curve = any(name == KNOWN_CURVES)
    if (.not. is_valid_cooling_curve) then
      call logger%error("unknown cooling curve: " // name)
    end if
  end function is_valid_cooling_curve


  subroutine get_cooling_table(name, table_T, table_lambda)
    use mod_cooling_curve_names
    use mod_radloss_tables

    character(len=*), intent(in) :: name
    real(dp), intent(out), allocatable :: table_T(:)
    real(dp), intent(out), allocatable :: table_lambda(:)
    integer :: table_n

    select case(name)
    case(JC_CORONA)
      table_n = n_JCcorona
      table_T = logT_JCcorona
      table_lambda = logL_JCcorona
    case(DALGARNO)
      table_n = n_DM
      table_T = logT_DM
      table_lambda = logL_DM
    case(DALGARNO2)
      table_n = n_DM_2
      table_T = logT_DM_2
      table_lambda = logL_DM_2
    case(ML_SOLAR)
      table_n = n_MLsolar1
      table_T = logT_MLsolar1
      table_lambda = logL_MLsolar1
    case(SPEX)
      table_n = n_SPEX
      table_T = logT_SPEX
      table_lambda = logL_SPEX + log10(L_SPEX_enh)
    case(SPEX_DALGARNO)
      table_n = n_SPEX + n_SPEX_enh_DM - 6
      allocate(table_T(table_n))
      allocate(table_lambda(table_n))
      table_T(1:n_SPEX_enh_DM - 1) = logT_SPEX_enh_DM( &
        1:n_SPEX_enh_DM - 1 &
      )
      table_T(n_SPEX_enh_DM:) = logT_SPEX(6:n_SPEX)
      table_lambda(1:n_SPEX_enh_DM - 1) = logL_SPEX_enh_DM( &
        1:n_SPEX_enh_DM - 1 &
      )
      table_lambda(n_SPEX_enh_DM:) = ( &
        logL_SPEX(6:n_SPEX) + log10(L_SPEX_enh(6:n_SPEX)) &
      )
    case(COLGAN)
      table_n = n_Colgan
      table_T = logT_Colgan
      table_lambda = logL_Colgan
    case(COLGAN_DM)
      table_n = n_Colgan + n_DM_2
      allocate(table_T(table_n))
      allocate(table_lambda(table_n))
      table_T(1:n_DM_2) = logT_DM_2(1:n_DM_2)
      table_T(n_DM_2+1:) = logT_Colgan(1:n_Colgan)
      table_lambda(1:n_DM_2) = logL_DM_2(1:n_DM_2)
      table_lambda(n_DM_2+1:) = logL_Colgan(1:n_Colgan)
    end select
  end subroutine get_cooling_table


  subroutine deallocate_cooling_curves()
    if (allocated(curve_T)) deallocate(curve_T)
    if (allocated(curve_lambda)) deallocate(curve_lambda)
    if (allocated(curve_dlambdadT)) deallocate(curve_dlambdadT)
  end subroutine deallocate_cooling_curves

end module mod_cooling_curves
