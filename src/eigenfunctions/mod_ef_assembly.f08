module mod_ef_assembly
  use mod_global_variables, only: dp, ic, NaN
  use mod_settings, only: settings_t
  use mod_grid, only: grid_t
  use mod_get_indices, only: get_index
  use mod_logging, only: logger, str
  implicit none

  private

  public :: retransform_eigenfunction
  public :: assemble_eigenfunction

contains

  function retransform_eigenfunction(ef_name, ef_eps, eigenfunction) &
    result(ef_transformed)
    use mod_state_vector_names

    character(len=*), intent(in) :: ef_name
    real(dp), intent(in) :: ef_eps(:)
    complex(dp), intent(in) :: eigenfunction(:)
    complex(dp) :: ef_transformed(size(eigenfunction))

    select case(ef_name)
    case(sv_rho1_name, sv_v3_name, sv_T1_name, sv_a2_name) ! var -> eps * var
      ef_transformed = eigenfunction / ef_eps
    case(sv_v1_name) ! v1 -> i * eps * v1
      ef_transformed = eigenfunction / (ef_eps * ic)
    case(sv_v2_name, sv_a3_name) ! var -> var
      ! do nothing
      ef_transformed = eigenfunction
    case(sv_a1_name) ! a1 -> i*a1
      ef_transformed = eigenfunction / ic
    case default
      call logger%error("wrong eigenfunction name during retransform")
    end select
  end function retransform_eigenfunction


  function assemble_eigenfunction( &
    settings, sv_component, grid, state_vector_index, eigenvector, derivative_order &
  ) result(assembled_ef)
    use mod_state_vector_component, only: sv_component_t
    use mod_basis_functions, only: basis_function

    type(settings_t), intent(in) :: settings
    type(sv_component_t), intent(in) :: sv_component
    type(grid_t), intent(in) :: grid
    integer, intent(in) :: state_vector_index
    complex(dp), intent(in) :: eigenvector(:)
    complex(dp) :: assembled_ef(settings%grid%get_ef_gridpts())

    integer, intent(in), optional :: derivative_order
    integer :: diff_order, subblock_idx, grid_idx, ef_grid_idx, dim_subblock
    real(dp) :: spline(4)
    procedure(basis_function), pointer :: spline_func

    diff_order = 0
    if (present(derivative_order)) diff_order = derivative_order
    dim_subblock = settings%dims%get_dim_subblock()

    ! map state vector index to actual subblock index, so 1 -> 1, 2 -> 3, 3 -> 5, etc.
    subblock_idx = 2 * state_vector_index - 1

    ! first gridpoint contribution
    call sv_component%get_spline_function(diff_order, spline_func)
    spline = spline_func(x=grid%ef_grid(1), x0=grid%base_grid(1), x1=grid%base_grid(2))
    assembled_ef(1) = get_combined_value_from_eigenvector( &
      eigenvector, idx=subblock_idx, dim_subblock=dim_subblock, spline=spline &
    )
    do grid_idx = 1, settings%grid%get_gridpts() - 1
      ! center of interval = 2 * i, end of interval = 2 * i + 1
      do ef_grid_idx = 2 * grid_idx, 2 * grid_idx + 1
        spline = spline_func( &
          x=grid%ef_grid(ef_grid_idx), &
          x0=grid%base_grid(grid_idx), &
          x1=grid%base_grid(grid_idx + 1) &
        )
        assembled_ef(ef_grid_idx) = get_combined_value_from_eigenvector( &
          eigenvector, idx=subblock_idx, dim_subblock=dim_subblock, spline=spline &
        )
      end do
      subblock_idx = subblock_idx + dim_subblock
    end do
    nullify(spline_func)
  end function assemble_eigenfunction


  pure complex(dp) function get_combined_value_from_eigenvector( &
    eigenvector, idx, dim_subblock, spline &
  )
    complex(dp), intent(in) :: eigenvector(:)
    integer, intent(in) :: idx
    integer, intent(in) :: dim_subblock
    real(dp), intent(in) :: spline(:)

    get_combined_value_from_eigenvector = ( &
      eigenvector(idx) * spline(2) &
      + eigenvector(idx + 1) * spline(4) &
      + eigenvector(idx + dim_subblock) * spline(1) &
      + eigenvector(idx + dim_subblock + 1) * spline(3) &
    )
  end function get_combined_value_from_eigenvector

end module mod_ef_assembly
