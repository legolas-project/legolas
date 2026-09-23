module mod_iv_initial_conditions
    use mod_global_variables, only: dp
    use mod_logging,          only: logger
    use mod_iv_globals,       only: iv_fcn_ptr_t, profile_fcn, zero_fcn
    implicit none
  
    private
  
    type, public :: ic_density_t
      procedure(profile_fcn), pointer, nopass :: rho   => null()
      procedure(profile_fcn), pointer, nopass :: drho  => null()
    end type ic_density_t
  
    type, public :: ic_velocity_t
      procedure(profile_fcn), pointer, nopass :: v   => null()
      procedure(profile_fcn), pointer, nopass :: dv  => null()
    end type ic_velocity_t

    type, public :: ic_temperature_t
      procedure(profile_fcn), pointer, nopass :: T   => null()
      procedure(profile_fcn), pointer, nopass :: dT  => null()
    end type ic_temperature_t

    type, public :: ic_magnetic_t
      procedure(profile_fcn), pointer, nopass :: a   => null()
      procedure(profile_fcn), pointer, nopass :: da  => null()
    end type ic_magnetic_t

    type, public :: initial_conditions_t
      type(ic_density_t)     :: density
      type(ic_velocity_t)    :: velocity_1
      type(ic_velocity_t)    :: velocity_2
      type(ic_velocity_t)    :: velocity_3
      type(ic_temperature_t) :: temperature
      type(ic_magnetic_t)    :: magnetic_1
      type(ic_magnetic_t)    :: magnetic_2
      type(ic_magnetic_t)    :: magnetic_3
    contains
      procedure :: set_ic_density_funcs
      procedure :: set_ic_velocity_1_funcs
      procedure :: set_ic_velocity_2_funcs
      procedure :: set_ic_velocity_3_funcs
      procedure :: set_ic_temperature_funcs
      procedure :: set_ic_a1_funcs
      procedure :: set_ic_a2_funcs
      procedure :: set_ic_a3_funcs
    end type initial_conditions_t

    public :: new_initial_conditions
  
  contains
  
    !=====================================================
    ! Constructor
    !=====================================================
    function new_initial_conditions() result(ic)
      type(initial_conditions_t) :: ic
      
      ! Default everything to zero
      ic%density%rho   => zero_fcn
      ic%density%drho  => zero_fcn
  
      ic%velocity_1%v  => zero_fcn
      ic%velocity_1%dv => zero_fcn

      ic%velocity_2%v  => zero_fcn
      ic%velocity_2%dv => zero_fcn

      ic%velocity_3%v  => zero_fcn
      ic%velocity_3%dv => zero_fcn

      ic%temperature%T  => zero_fcn
      ic%temperature%dT => zero_fcn

      ic%magnetic_1%a  => zero_fcn
      ic%magnetic_1%da => zero_fcn

      ic%magnetic_2%a  => zero_fcn
      ic%magnetic_2%da => zero_fcn

      ic%magnetic_3%a  => zero_fcn
      ic%magnetic_3%da => zero_fcn

    end function new_initial_conditions
  
    !=====================================================
    ! Setter routines
    !=====================================================
    subroutine set_ic_density_funcs(self, rho_func, drho_func)
      class(initial_conditions_t), intent(inout) :: self
      procedure(profile_fcn) :: rho_func
      procedure(profile_fcn), optional :: drho_func
  
      call logger%debug("Setting ICs for component rho.")
      self%density%rho => rho_func
      if (present(drho_func)) self%density%drho => drho_func
    end subroutine set_ic_density_funcs

  
    subroutine set_ic_velocity_1_funcs(self, v01_func, dv01_func)
      class(initial_conditions_t), intent(inout) :: self
      procedure(profile_fcn) :: v01_func
      procedure(profile_fcn) :: dv01_func
  
      call logger%debug("Setting ICs for component v1.")
      self%velocity_1%v => v01_func
      self%velocity_1%dv => dv01_func
    end subroutine set_ic_velocity_1_funcs

    subroutine set_ic_velocity_2_funcs(self, v02_func, dv02_func)
      class(initial_conditions_t), intent(inout) :: self
      procedure(profile_fcn) :: v02_func
      procedure(profile_fcn) :: dv02_func

      call logger%debug("Setting ICs for component v2.")
      self%velocity_2%v => v02_func
      self%velocity_2%dv => dv02_func
    end subroutine set_ic_velocity_2_funcs

    subroutine set_ic_velocity_3_funcs(self, v03_func, dv03_func)
      class(initial_conditions_t), intent(inout) :: self
      procedure(profile_fcn) :: v03_func
      procedure(profile_fcn) :: dv03_func

      call logger%debug("Setting ICs for component v3.")
      self%velocity_3%v => v03_func
      self%velocity_3%dv => dv03_func
    end subroutine set_ic_velocity_3_funcs

    subroutine set_ic_temperature_funcs(self, T_func, dT_func)
      class(initial_conditions_t), intent(inout) :: self
      procedure(profile_fcn) :: T_func
      procedure(profile_fcn), optional :: dT_func

      call logger%debug("Setting ICs for component T.")
      self%temperature%T => T_func
      if (present(dT_func)) self%temperature%dT => dT_func
    end subroutine set_ic_temperature_funcs

    ! a1 defaults to a quadratic basis function, so its derivative is optional.
    subroutine set_ic_a1_funcs(self, a1_func, da1_func)
      class(initial_conditions_t), intent(inout) :: self
      procedure(profile_fcn) :: a1_func
      procedure(profile_fcn), optional :: da1_func

      call logger%debug("Setting ICs for component a1.")
      self%magnetic_1%a => a1_func
      if (present(da1_func)) self%magnetic_1%da => da1_func
    end subroutine set_ic_a1_funcs

    ! a2 defaults to a cubic basis function, so its derivative is required.
    subroutine set_ic_a2_funcs(self, a2_func, da2_func)
      class(initial_conditions_t), intent(inout) :: self
      procedure(profile_fcn) :: a2_func
      procedure(profile_fcn) :: da2_func

      call logger%debug("Setting ICs for component a2.")
      self%magnetic_2%a => a2_func
      self%magnetic_2%da => da2_func
    end subroutine set_ic_a2_funcs

    ! a3 defaults to a cubic basis function, so its derivative is required.
    subroutine set_ic_a3_funcs(self, a3_func, da3_func)
      class(initial_conditions_t), intent(inout) :: self
      procedure(profile_fcn) :: a3_func
      procedure(profile_fcn) :: da3_func

      call logger%debug("Setting ICs for component a3.")
      self%magnetic_3%a => a3_func
      self%magnetic_3%da => da3_func
    end subroutine set_ic_a3_funcs

  end module mod_iv_initial_conditions
  