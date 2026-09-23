module mod_iv_state_vector_component
  use mod_global_variables, only: dp, str_len_arr
  use mod_logging, only: logger, str
  use mod_basis_functions, only: basis_function
  use mod_settings, only: settings_t
  use mod_grid, only: grid_t
  use mod_iv_globals, only: profile_fcn
  use mod_state_vector_component, only: sv_component_t
  implicit none

  type, public :: iv_sv_component_t
    type(sv_component_t), pointer :: base => null()  ! pointer to existing sv_component_t instance
    character(len=:), allocatable :: name

    logical :: is_bound = .false.
    logical :: cplx_trans = .false.

    real(dp), allocatable :: c1(:), c2(:)   ! coefficients
    complex(dp), allocatable :: profile(:)  ! profile on grid

    ! Callback procedure pointers for profile function & derivative
    procedure(profile_fcn), nopass, pointer :: p_fcn  => null()
    procedure(profile_fcn), nopass, pointer :: p_dfcn => null()

    contains
      procedure, public :: bind_iv_component
      procedure, public :: compute_coeffs
      procedure, public :: compute_quad_coeffs
      procedure, public :: compute_cubic_coeffs
      procedure, public :: reconstruct_profile
  end type iv_sv_component_t

  public :: new_iv_component

contains
  ! -----------------------------------------------------------------
  ! Constructor
  ! -----------------------------------------------------------------
  function new_iv_component(name) result(comp)
    type(iv_sv_component_t) :: comp
    character(len=*), intent(in) :: name

    allocate(character(len=len_trim(name)) :: comp%name)
    comp%name = trim(name)
  end function new_iv_component


  subroutine bind_iv_component(self, base_instance, fcn, dfcn)
    class(iv_sv_component_t), intent(inout)  :: self
    type(sv_component_t), intent(in), target :: base_instance
    procedure(profile_fcn), pointer          :: fcn
    procedure(profile_fcn), pointer          :: dfcn

    if (.not. self%is_bound) then
      ! Make base point to the passed-in instance
      self%base => base_instance

      ! Store the function pointers
      self%p_fcn  => fcn
      self%p_dfcn => dfcn

      if (self%base%get_name() == 'v1' .or. self%base%get_name() == 'a1') then
        self%cplx_trans = .true.
      end if

      self%is_bound = .true.
    else
      call logger%error("iv_sv_component_t is already bound!")
    end if
  end subroutine bind_iv_component


  ! -----------------------------------------------------------------
  ! Compute coefficients
  ! -----------------------------------------------------------------
  subroutine compute_coeffs(self, N, nodes)
    class(iv_sv_component_t), intent(inout) :: self
    integer, intent(in)                     :: N
    real(dp), intent(in)                    :: nodes(N)

    if (allocated(self%c1) .or. allocated(self%c2)) then
      call logger%warning("IV component coefficients already allocated")
      deallocate(self%c1)
      deallocate(self%c2)
    end if

    allocate(self%c1(N))
    allocate(self%c2(N))

    ! Compute the coefficients associated to each basis fcn.
    select case(self%base%get_basis_function_name())
    case('quadratic')
      call self%compute_quad_coeffs(N, nodes)
    case('cubic')
      call self%compute_cubic_coeffs(N, nodes)
    case default
      call logger%error("Basis function type is unknown to IV state vector component.")
    end select
  end subroutine compute_coeffs


  ! -----------------------------------------------------------------
  ! Compute expansion coefficients for each type
  ! -----------------------------------------------------------------
  subroutine compute_quad_coeffs(self, N, nodes)
    class(iv_sv_component_t), intent(inout) :: self
    integer, intent(in)    :: N
    real(dp), intent(in)   :: nodes(N)

    real(dp), allocatable :: midpoints(:)
    allocate(midpoints(N - 1))

    ! c1 = fcn(midpoints)
    midpoints = 0.5 * (nodes(2:) + nodes(:N-1))
    self%c1 = 0.0d0
    self%c1(2:) = self%p_fcn(midpoints)  ! 0 followed by N-1 midpoint values

    ! c2 = fcn(nodes)
    self%c2 = self%p_fcn(nodes)

    deallocate(midpoints)
  end subroutine compute_quad_coeffs


  subroutine compute_cubic_coeffs(self, N, nodes)
    class(iv_sv_component_t), intent(inout) :: self
    integer, intent(in)    :: N
    real(dp), intent(in)   :: nodes(N)

    ! u1 = fcn(nodes)
    self%c1 = self%p_fcn(nodes)

    ! u2 = dfcn(nodes)
    self%c2 = self%p_dfcn(nodes)
  end subroutine compute_cubic_coeffs


  ! -----------------------------------------------------------------
  ! Reconstruct the profiles from coefficients
  ! -----------------------------------------------------------------
  subroutine reconstruct_profile(self, settings, grid)
    class(iv_sv_component_t), intent(inout) :: self
    type(settings_t), intent(in) :: settings
    type(grid_t), intent(in) :: grid
    
    procedure(basis_function), pointer :: f
    real(dp) :: h(4)
    integer :: grid_idx, ef_grid_idx

    if (allocated(self%profile)) deallocate(self%profile)
    allocate(self%profile(settings%grid%get_ef_gridpts()))

    call self%base%get_spline_function(0, f)

    ! first gridpoint contribution
    h = f(x=grid%ef_grid(1), x0=grid%base_grid(1), x1=grid%base_grid(2))
    self%profile(1) =         &
          self%c1(2) * h(1) + &
          self%c1(1) * h(2) + &
          self%c2(2) * h(3) + &
          self%c2(1) * h(4)

    do grid_idx = 1, settings%grid%get_gridpts() - 1
      do ef_grid_idx = 2 * grid_idx, 2 * grid_idx + 1
        h = f(x=grid%ef_grid(ef_grid_idx), x0=grid%base_grid(grid_idx), x1=grid%base_grid(grid_idx + 1))
    
        self%profile(ef_grid_idx) =   &
              self%c1(grid_idx + 1) * h(1) + &
              self%c1(grid_idx)     * h(2) + &
              self%c2(grid_idx + 1) * h(3) + &
              self%c2(grid_idx)     * h(4)
      end do
    end do
    nullify(f)
   
  end subroutine reconstruct_profile

end module mod_iv_state_vector_component