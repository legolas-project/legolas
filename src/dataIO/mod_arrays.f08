! =============================================================================
!> Module to handle imported numerical equilibria.
!! Contains subroutines to retrieve the equilibrium arrays from a file specified in the parfile.
module mod_arrays
    use, intrinsic :: iso_fortran_env, only: iostat_end
    use mod_equilibrium_params, only: input_file, eq_bool
    use mod_global_variables, only: dp, str_len, dp_LIMIT
    use mod_interpolation, only: get_numerical_derivative
    use mod_logging, only: logger
    use mod_settings, only: settings_t
    use mod_grid, only: grid_t
    implicit none

    private

    public :: import_equilibrium_data
    public :: interpolate_equilibrium_to_grid
    public :: lookup_equilibrium_value
    public :: deallocate_input

    integer :: file_id = 123
    integer :: num_var = 10

    real(dp), allocatable :: input(:,:)
    real(dp), allocatable :: equil_on_grid(:,:), d_equil_on_grid(:,:), dd_equil_on_grid(:,:)

contains

    !> Imports arrays from the file specified by the parfile parameter input_file.
    !! To be called in the equilibrium submodule.
    subroutine import_equilibrium_data(settings, grid)
        type(settings_t), intent(inout) :: settings
        type(grid_t), intent(inout) :: grid

        integer :: lpts, idx
        real(dp), allocatable  :: array(:)

        open( &
            file_id, &
            file=input_file, &
            form="unformatted" &
        )
        
        read(file_id) lpts
        allocate(array(lpts))
        allocate(input(lpts, num_var))
        input(:,:) = 0.0_dp

        do idx = 1, num_var
            read(file_id) array
            input(:, idx) = array
        end do

        close(file_id)
        deallocate(array)

        if (maxval(abs(input(:, 1))) < dp_LIMIT) then
            call logger%error("Coordinate array in imported data is absent or zero")
        end if

        if (eq_bool) then
            call settings%grid%set_grid_boundaries( &
                grid_start=input(1, 1), grid_end=input(lpts, 1) &
            )
        end if
    end subroutine import_equilibrium_data


    !> Transders input to Legolas's grid, and subsequently calls to numerical
    !! derivation. Called in mod_equilibrium after grid initialisation.
    subroutine interpolate_equilibrium_to_grid(settings, grid)
        type(settings_t), intent(inout) :: settings
        type(grid_t), intent(inout) :: grid
        integer  :: gridpts

        if (.not. allocated(input)) return

        gridpts = settings%grid%get_gridpts()
        allocate(equil_on_grid(gridpts, num_var))
        equil_on_grid(:, :) = 0.0_dp

        equil_on_grid(:, 1) = grid%base_grid
        call interpolate(gridpts)

        allocate(d_equil_on_grid(gridpts, num_var))
        d_equil_on_grid(:, :) = 0.0_dp
        allocate(dd_equil_on_grid(gridpts, num_var))
        dd_equil_on_grid(:, :) = 0.0_dp

        call derivatives(gridpts)
    end subroutine interpolate_equilibrium_to_grid


    !> Handles the interpolation from input to Legolas's grid.
    subroutine interpolate(gpts)
        integer, intent(in) :: gpts
        integer  :: ipts, idl, idu
        integer  :: i, j
        real(dp) :: x

        ipts = size(input, dim=1)
        if (equil_on_grid(1, 1) < input(1, 1) .or. &
            equil_on_grid(gpts, 1) > input(ipts, 1)) then
            call logger%warning("Linear extrapolation of the imported data to the grid")
        end if

        do i = 1, gpts
            x = equil_on_grid(i, 1)
            if (x <= input(1, 1)) then
                idl = 1
                idu = 2
            else if (x >= input(ipts, 1)) then
                idl = ipts - 1
                idu = ipts
            else
                idl = maxloc(input(:, 1), mask=(input(:, 1) < x), dim=1)
                idu = minloc(input(:, 1), mask=(input(:, 1) > x), dim=1)
            end if

            do j = 1, num_var
                !!! linear interpolation
                equil_on_grid(i, j) = input(idl, j) + (x - input(idl, 1)) * &
                    (input(idu, j) - input(idl, j)) / (input(idu, 1) - input(idl, 1))
            end do
        end do
    end subroutine interpolate


    !> Calculates the numerical derivatives on Legolas's grid.
    subroutine derivatives(gpts)
        integer, intent(in) :: gpts
        real(dp) :: x(gpts), array(gpts), d_array(gpts)
        integer  :: i

        x = equil_on_grid(:, 1)
        d_equil_on_grid(:, 1) = x
        dd_equil_on_grid(:, 1) = x

        do i = 2, num_var
            ! first derivative
            array = equil_on_grid(:, i)
            call get_numerical_derivative(x, array, d_array)
            d_equil_on_grid(:, i) = d_array
            ! second derivative
            array = d_array
            call get_numerical_derivative(x, array, d_array)
            dd_equil_on_grid(:, i) = d_array
        end do
    end subroutine derivatives


    !> Looks up the equilibrium value for given quantity and position.
    subroutine lookup_equilibrium_value(type, x, derivative, out)
        character(len=*), intent(in) :: type
        real(dp), intent(in)  :: x
        integer, intent(in) :: derivative
        real(dp), intent(out) :: out
        integer :: idx, idx_t

        idx = minloc(abs(equil_on_grid(:, 1) - x), dim=1)
        call tag_to_index(type, idx_t)

        select case(derivative)
            case(0)
                out = equil_on_grid(idx, idx_t)
            case(1)
                out = d_equil_on_grid(idx, idx_t)
            case(2)
                out = dd_equil_on_grid(idx, idx_t)
            case default
                call logger%error("Specified derivative not available")
        end select 
    end subroutine lookup_equilibrium_value


    !> Translates equilibrium name to index.
    subroutine tag_to_index(tag, index)
        character(len=*), intent(in) :: tag
        integer, intent(out) :: index

        select case(trim(tag))
            case("u1")
                index = 1
            case("x")
                index = 1
            case("r")
                index = 1
            case("rho0")
                index = 2
            case("v01")
                index = 3
            case("v02")
                index = 4
            case("v03")
                index = 5
            case("T0")
                index = 6
            case("B01")
                index = 7
            case("B02")
                index = 8
            case("B03")
                index = 9
            case("grav")
                index = 10
            case default
                call logger%warning( &
                    "Unknown quantity " // trim(tag) &
                )
                index = -1
        end select
    end subroutine tag_to_index


    !> Deallocates this module's arrays. Called in main as part of cleanup.
    subroutine deallocate_input()
        deallocate(input)
        deallocate(equil_on_grid)
        deallocate(d_equil_on_grid)
        deallocate(dd_equil_on_grid)
    end subroutine deallocate_input

end module mod_arrays
