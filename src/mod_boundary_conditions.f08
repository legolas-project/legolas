!
! MODULE: mod_boundary_conditions
!
!> @author
!> Niels Claes
!> niels.claes@kuleuven.be
!
! DESCRIPTION:
!> Module to calculate the boundary conditions for the eigenvalue problem.
!
module mod_boundary_conditions
  use mod_global_variables, only: dp, matrix_gridpts, dim_quadblock, dim_subblock
  implicit none

  private

  logical, save   :: kappa_perp_is_zero

  public :: apply_boundary_conditions

contains

  subroutine apply_boundary_conditions(matrix_A, matrix_B)
    use mod_global_variables, only: dp_LIMIT
    use mod_equilibrium, only: kappa_field

    complex(dp), intent(inout)  :: matrix_A(matrix_gridpts, matrix_gridpts)
    real(dp), intent(inout)     :: matrix_B(matrix_gridpts, matrix_gridpts)
    complex(dp)                 :: quadblock(dim_quadblock, dim_quadblock)
    integer                     :: idx_end_left, idx_start_right

    ! check if perpendicular thermal conduction is present
    kappa_perp_is_zero = .true.
    if (any(abs(kappa_field % kappa_perp) > dp_LIMIT)) then
      kappa_perp_is_zero = .false.
    end if

    ! end of index first-gridpoint quadblock
    idx_end_left = dim_quadblock
    ! start of index last-gridpoint quadblock
    idx_start_right = matrix_gridpts - dim_quadblock + 1

    ! matrix B left-edge quadblock
    quadblock = matrix_B(1:idx_end_left, 1:idx_end_left)
    call essential_boundaries(quadblock, edge='l_edge', matrix='B')
    matrix_B(1:idx_end_left, 1:idx_end_left) = real(quadblock)
    ! matrix B right-edge quadblock
    quadblock = matrix_B(idx_start_right:matrix_gridpts, idx_start_right:matrix_gridpts)
    call essential_boundaries(quadblock, edge='r_edge', matrix='B')
    matrix_B(idx_start_right:matrix_gridpts, idx_start_right:matrix_gridpts) = real(quadblock)

    ! matrix A left-edge quadblock
    quadblock = matrix_A(1:idx_end_left, 1:idx_end_left)
    call essential_boundaries(quadblock, edge='l_edge', matrix='A')
    call natural_boundaries(quadblock, edge='l_edge')
    matrix_A(1:idx_end_left, 1:idx_end_left) = quadblock
    ! matrix A right-edge quadblock
    quadblock = matrix_A(idx_start_right:matrix_gridpts, idx_start_right:matrix_gridpts)
    call essential_boundaries(quadblock, edge='r_edge', matrix='A')
    call natural_boundaries(quadblock, edge='r_edge')
    matrix_A(idx_start_right:matrix_gridpts, idx_start_right:matrix_gridpts) = quadblock
  end subroutine apply_boundary_conditions

  subroutine essential_boundaries(quadblock, edge, matrix)
    use mod_global_variables, only: boundary_type
    use mod_logging, only: log_message

    complex(dp), intent(inout)    :: quadblock(dim_quadblock, dim_quadblock)
    character(len=6), intent(in)  :: edge
    character, intent(in)         :: matrix

    complex(dp)                   :: diagonal_factor
    integer                       :: i, j, qua_zeroes(5), wall_idx_left(4), wall_idx_right(4)

    if (matrix == 'B') then
      diagonal_factor = (1.0d0, 0.0d0)
    else if (matrix == 'A') then
      diagonal_factor = (1.0d20, 0.0d0)
    else
      call log_message("essential boundaries: invalid matrix argument", level='error')
    end if

    ! Always: the contribution from the 0 basis function automatically
    ! zeroes out the odd rows/columns for the quadratic variables on the left edge
    ! so we handle those indices explicitly
    qua_zeroes = [1, 5, 7, 9, 11]
    if (edge == 'l_edge') then
      do i = 1, size(qua_zeroes)
        j = qua_zeroes(i)
        quadblock(j, j) = diagonal_factor
      end do
    end if

    ! Wall/regularity conditions: handling of v1, a2 and a3 (and T if conduction).
    ! v1, a2 and a3 are cubic elements, so omit non-zero basis functions (odd rows/columns)
    ! T is a quadratic element, so omit even row/columns
    wall_idx_left = [3, 13, 15, 10]
    wall_idx_right = [19, 29, 31, 26]

    select case(boundary_type)
    case('wall')
      if (edge == 'l_edge') then
        ! left regularity/wall conditions
        do i = 1, size(wall_idx_left)
          j = wall_idx_left(i)
          if (j == 10 .and. kappa_perp_is_zero) then
            cycle
          end if
          quadblock(j, :) = (0.0d0, 0.0d0)
          quadblock(:, j) = (0.0d0, 0.0d0)
          quadblock(j, j) = diagonal_factor
        end do
      else if (edge == 'r_edge') then
        do i = 1, size(wall_idx_right)
          j = wall_idx_right(i)
          if ((j == 26) .and. kappa_perp_is_zero) then
            cycle
          end if
          quadblock(j, :) = (0.0d0, 0.0d0)
          quadblock(:, j) = (0.0d0, 0.0d0)
          quadblock(j, j) = diagonal_factor
        end do
      else
        call log_message("essential boundaries: invalid edge argument", level='error')
      end if

    case default
      call log_message( "essential boundaries: invalid boundary_type", level='error')
    end select

  end subroutine essential_boundaries

  subroutine natural_boundaries(quadblock, edge)
    use mod_global_variables, only: boundary_type, gauss_gridpts, gamma_1, ic, dim_subblock
    use mod_logging, only: log_message
    use mod_equilibrium, only: B_field, eta_field
    use mod_grid, only: eps_grid, d_eps_grid_dr
    use mod_equilibrium_params, only: k2, k3

    complex(dp), intent(inout)    :: quadblock(dim_quadblock, dim_quadblock)
    character(len=6), intent(in)  :: edge

    complex(dp), allocatable  :: surface_terms(:)
    real(dp)                  :: eps, d_eps_dr, eta, B02, dB02, B03, dB03, drB02
    integer, allocatable      :: positions(:, :)
    integer                   :: idx, i

    if (.not. boundary_type == 'wall') then
      call log_message('natural boundaries: only wall is implemented!', level='error')
    end if

    ! For now only the terms concerning the solid wall boundary are implemented.
    ! If free boundary conditions are added (eg. vacuum-wall), then additional
    ! terms have to be added.
    ! Hence:
    !   - v1 momentum equation: currently no terms added (v1 = 0 for a wall)
    !   - T1 energy equation: resistive and conductive terms
    !   - a2 induction equation: currently no terms added (a2 = 0 for a wall)
    !   - a3 induction equation: currently no terms added (a3 = 0 for a wall)
    ! Note:
    !   For a wall with perpendicular thermal conduction we also have the condition T1 = 0.
    !   Hence, in that case there are no surface terms to be added, and we simply return.
    !   So for now we only have resistive terms in the energy equation.
    if (.not. kappa_perp_is_zero) then
      return
    end if

    if (edge == 'l_edge') then
      idx = 1
    else if (edge == 'r_edge') then
      idx = gauss_gridpts
    else
      call log_message('natural boundaries: wrong edge supplied' // edge, level='error')
    end if

    ! retrieve variables at current edge
    eps = eps_grid(idx)
    d_eps_dr = d_eps_grid_dr(idx)
    B02 = B_field % B02(idx)
    dB02 = B_field % d_B02_dr(idx)
    B03 = B_field % B03(idx)
    dB03 = B_field % d_B03_dr(idx)
    drB02 = d_eps_dr * B02 + eps * dB02
    eta = eta_field % eta(idx)

    allocate(positions(3, 2))
    allocate(surface_terms(3))

    ! surface term for element (5, 6)
    surface_terms(1) = 2.0d0 * ic * gamma_1 * (1.0d0 / eps) * eta * (k3 * drB02 - k2 * dB03)
    positions(1, :) = [5, 6]
    ! surface term for element (5, 7)
    surface_terms(2) = 2.0d0 * ic * gamma_1 * (1.0d0 / eps) * eta * dB03
    positions(2, :) = [5, 7]
    ! surface term for element (5, 8)
    surface_terms(3) = -2.0d0 * ic * gamma_1 * (1.0d0 / eps) * eta * drB02
    positions(3, :) = [5, 8]

    ! l_edge: add to bottom-right of 2x2 block, for top-left subblock only
    ! r_edge: add to bottom-right of 2x2 block, for bottom-right subblock only
    if (edge == 'l_edge') then
      positions = 2 * positions
    else if (edge == 'r_edge') then
      positions = 2 * positions + dim_subblock
    end if

    do i = 1, size(surface_terms)
      quadblock(positions(i, 1), positions(i, 2)) = quadblock(positions(i, 1), positions(i, 2)) + surface_terms(i)
    end do

    deallocate(positions)
    deallocate(surface_terms)
  end subroutine natural_boundaries

end module mod_boundary_conditions
