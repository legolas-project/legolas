module mod_iv_state_vector
  use mod_global_variables, only: dp, ic
  use mod_logging, only: logger, str
  use mod_settings, only: settings_t
  use mod_grid, only: grid_t
  use mod_state_vector, only: state_vector_t
  use mod_state_vector_names
  use mod_iv_globals, only: profile_fcn, iv_fcn_ptr_t, zero_fcn
  use mod_iv_state_vector_component, only: iv_sv_component_t, new_iv_component
  use mod_iv_initial_conditions, only: initial_conditions_t
  
  implicit none

  private

  ! Type to hold the component pointers
  type, private :: iv_comp_ptr_t
    class(iv_sv_component_t), pointer :: ptr
  end type iv_comp_ptr_t

  type, public :: iv_state_vector_t
    type(state_vector_t), pointer :: base => null()  ! pointer to existing state_vector_t instance
    class(iv_comp_ptr_t), allocatable :: components(:)
    logical :: is_initialised = .false.
    integer :: num_components
    integer :: stride

    complex(dp), allocatable :: x0(:)

    contains
      procedure, public :: initialise_components
      procedure, public :: assemble_iv_array
      procedure, public :: disassemble_iv_array
      procedure, public :: reassemble_from_block

  end type iv_state_vector_t

  type(iv_sv_component_t), public, protected, target :: iv_rho1
  type(iv_sv_component_t), public, protected, target :: iv_v1
  type(iv_sv_component_t), public, protected, target :: iv_v2
  type(iv_sv_component_t), public, protected, target :: iv_v3
  type(iv_sv_component_t), public, protected, target :: iv_T1
  type(iv_sv_component_t), public, protected, target :: iv_a1
  type(iv_sv_component_t), public, protected, target :: iv_a2
  type(iv_sv_component_t), public, protected, target :: iv_a3

  public :: init_and_bind

  contains
  ! -----------------------------------------------------------------
  ! Constructor
  ! -----------------------------------------------------------------
  function init_and_bind(base_instance) result(iv_state_vector)
    type(state_vector_t), intent(in), target :: base_instance
    type(iv_state_vector_t) :: iv_state_vector

    ! Make base point to the passed-in instance.
    iv_state_vector%base => base_instance
  end function init_and_bind


  subroutine initialise_components(self, physics_type, initial_conditions)
    class(iv_state_vector_t), intent(inout) :: self
    character(len=*), intent(in) :: physics_type
    class(initial_conditions_t), intent(inout) :: initial_conditions

    procedure(profile_fcn), pointer :: fcn  => null()
    procedure(profile_fcn), pointer :: dfcn => null()
    character(len=:), allocatable :: name
    integer :: i
      
    if (self%is_initialised) then
      call logger%error("IV state vector is already initialised.")
    end if

    ! Initialise all of the components
    iv_rho1 = new_iv_component("rho")
    iv_v1 = new_iv_component("v1")
    iv_v2 = new_iv_component("v2")
    iv_v3 = new_iv_component("v3")
    iv_T1 = new_iv_component("T")
    iv_a1 = new_iv_component("a1")
    iv_a2 = new_iv_component("a2")
    iv_a3 = new_iv_component("a3")

    ! Store pointers to the active components
    select case(physics_type)
    case("isothermal-1d")
      self%num_components = 2
      allocate(self%components(self%num_components))
      self%components(1)%ptr => iv_rho1
      self%components(2)%ptr => iv_v1

    case("hd-1d")
      self%num_components = 3
      allocate(self%components(self%num_components))
      self%components(1)%ptr => iv_rho1
      self%components(2)%ptr => iv_v1
      self%components(3)%ptr => iv_T1
  
    case("hd")
      self%num_components = 5
      allocate(self%components(self%num_components))
      self%components(1)%ptr => iv_rho1
      self%components(2)%ptr => iv_v1
      self%components(3)%ptr => iv_v2
      self%components(4)%ptr => iv_v3
      self%components(5)%ptr => iv_T1

    case("mhd")
      self%num_components = 8
      allocate(self%components(self%num_components))
      self%components(1)%ptr => iv_rho1
      self%components(2)%ptr => iv_v1
      self%components(3)%ptr => iv_v2
      self%components(4)%ptr => iv_v3
      self%components(5)%ptr => iv_T1
      self%components(6)%ptr => iv_a1
      self%components(7)%ptr => iv_a2
      self%components(8)%ptr => iv_a3

    case default
      call logger%error( &
        "IV state vector: unsupported physics type '" // physics_type // "'" &
      )
      return
    end select
  
    self%stride = 2 * self%num_components
  
    if (self%num_components /= size(self%base%components)) then
      call logger%error("Mismatch in number of state components.")
    end if
  
    ! Bind initial conditions to the components
    do i = 1, self%num_components
      ! Retrieve the name of this component
      name = self%components(i)%ptr%name
      select case(name)
      case("rho")
         fcn  => initial_conditions%density%rho
         dfcn => initial_conditions%density%drho
  
      case("v1")
         fcn  => initial_conditions%velocity_1%v
         dfcn => initial_conditions%velocity_1%dv

      case("v2")
         fcn  => initial_conditions%velocity_2%v
         dfcn => initial_conditions%velocity_2%dv

      case("v3")
         fcn  => initial_conditions%velocity_3%v
         dfcn => initial_conditions%velocity_3%dv

      case("T")
        fcn  => initial_conditions%temperature%T
        dfcn => initial_conditions%temperature%dT

      case("a1")
         fcn  => initial_conditions%magnetic_1%a
         dfcn => initial_conditions%magnetic_1%da

      case("a2")
         fcn  => initial_conditions%magnetic_2%a
         dfcn => initial_conditions%magnetic_2%da

      case("a3")
         fcn  => initial_conditions%magnetic_3%a
         dfcn => initial_conditions%magnetic_3%da

      case default
         ! If no match, assume 0
         fcn  => zero_fcn
         dfcn => zero_fcn
         call logger%info("No perturbation found for component "//name//", set to zero")
  
      end select
  
      ! Now actually bind the pointers
      call self%components(i)%ptr%bind_iv_component( &
           self%base%components(i)%ptr, fcn, dfcn)
    end do
  
    self%is_initialised = .true.
  end subroutine initialise_components


  ! -----------------------------------------------------------------
  ! Assemble IV array from profiles
  ! -----------------------------------------------------------------
  subroutine assemble_iv_array(self, N, nodes)
    !>
    class(iv_state_vector_t), intent(inout) :: self
    !> Number of grid points
    integer, intent(in) :: N
    !> Array of grid points
    real(dp), intent(in) :: nodes(N)

    integer :: i, j, idx

    ! Compute coefficients for each component
    do i = 1, self%num_components
      call self%components(i)%ptr%compute_coeffs(N, nodes)
    end do

    allocate(self%x0(self%stride * N))

    ! Assembly of interleaved array into block format
    do i = 1, self%num_components  ! components
      do j = 1, 2                  ! c1 or c2
        idx = 2*(i - 1) + j        ! compute the offset in x0
        select case(j)
        case(1)
          ! Multiply by i if needed
          if (self%components(i)%ptr%cplx_trans) then
            self%x0(idx : self%stride*N : self%stride) = &
              cmplx(self%components(i)%ptr%c1, 0.0_dp, kind=dp) * ic
          else
            self%x0(idx : self%stride*N : self%stride) = &
              cmplx(self%components(i)%ptr%c1, 0.0_dp, kind=dp)
          end if          
        case(2)
          if (self%components(i)%ptr%cplx_trans) then
            self%x0(idx : self%stride*N : self%stride) = &
              cmplx(self%components(i)%ptr%c2, 0.0_dp, kind=dp) * ic
          else
            self%x0(idx : self%stride*N : self%stride) = &
              cmplx(self%components(i)%ptr%c2, 0.0_dp, kind=dp)
          end if
        end select
      end do
    end do

  end subroutine assemble_iv_array


  subroutine reassemble_from_block(self, settings, grid)
    class(iv_state_vector_t), intent(inout) :: self
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid

    integer :: i

    ! Get each component to compute its profile
    do i = 1, self%num_components
      call self%components(i)%ptr%reconstruct_profile(settings, grid)
    end do

  end subroutine reassemble_from_block

  
  !> Gather each component's c1 and c2 coefficients from the
  !! block-format array x0. This is the inverse of assemble_iv_array.
  subroutine disassemble_iv_array(self, N)
    class(iv_state_vector_t), intent(inout) :: self
    !> # of grid points (nodes) for each component
    integer, intent(in) :: N
  
    integer :: i, j, idx
  
    ! Loop over all components and fill c1, c2 from x0
    do i = 1, self%num_components
      ! Make sure c1 and c2 are allocated
      if (.not. allocated(self%components(i)%ptr%c1)) then
        allocate(self%components(i)%ptr%c1(N), self%components(i)%ptr%c2(N))
      end if
  
      do j = 1, 2  ! c1 or c2
        idx = 2*(i - 1) + j  ! offset in the x0 array
        select case (j)
          case(1)  ! c1
            if (self%components(i)%ptr%cplx_trans) then
              self%components(i)%ptr%c1 = real(self%x0(idx : self%stride*N : self%stride) / ic)
            else
              self%components(i)%ptr%c1 = real(self%x0(idx : self%stride*N : self%stride))
            end if
          case(2)  ! c2
            if (self%components(i)%ptr%cplx_trans) then
              self%components(i)%ptr%c2 = real(self%x0(idx : self%stride*N : self%stride) / ic)
            else
              self%components(i)%ptr%c2 = real(self%x0(idx : self%stride*N : self%stride))
            end if
        end select
      end do
    end do
  
  end subroutine disassemble_iv_array
  

end module mod_iv_state_vector