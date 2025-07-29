! =============================================================================
!> Module to handle imported numerical equilibria.
!! Contains subroutines to retrieve the equilibrium arrays from a file specified in the parfile.
module mod_arrays
    use, intrinsic :: iso_fortran_env, only: iostat_end
    use mod_equilibrium_params, only: input_file, eq_bool, n_input
    use mod_global_variables, only: dp, str_len, dp_LIMIT
    use mod_interpolation
    use mod_logging, only: logger
    use mod_settings, only: settings_t
    use mod_grid, only: grid_t
    implicit none

    private

    public :: import_equilibrium_data
    public :: lookup_equilibrium_value
    public :: deallocate_input

    integer :: file_id = 123
    integer :: num_var = 10

    real(dp), allocatable :: input(:,:)
    real(dp), allocatable :: interp(:,:), d_interp(:,:), dd_interp(:,:)

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
        else if (settings%grid%get_grid_start() < input(1, 1) .or. &
            settings%grid%get_grid_end() > input(lpts, 1)) then
            call logger%error("Grid boundaries outside of imported data range")
        end if

        call interpolate_and_derive()
    end subroutine import_equilibrium_data


    subroutine interpolate_and_derive()
        integer  :: i
        real(dp) :: x_out(n_input), y_out(n_input)
        real(dp) :: derivative(n_input)

        allocate(interp(n_input, num_var))
        allocate(d_interp(n_input, num_var))
        allocate(dd_interp(n_input, num_var))

        do i = 2, num_var
            call interpolate_table(n_input, input(:, 1), input(:, i), x_out, y_out)
            interp(:, i) = y_out

            call get_numerical_derivative(x_out, y_out, derivative)
            d_interp(:, i) = derivative

            call get_second_numerical_derivative(x_out, y_out, derivative)
            dd_interp(:, i) = derivative

            if (i == 2) then
                interp(:, 1) = x_out
                d_interp(:, 1) = x_out
                dd_interp(:, 1) = x_out
            end if
        end do
    end subroutine interpolate_and_derive


    !> Looks up the equilibrium value for given quantity and position.
    subroutine lookup_equilibrium_value(type, x, derivative, out)
        character(len=*), intent(in) :: type
        real(dp), intent(in)  :: x
        integer, intent(in) :: derivative
        real(dp), intent(out) :: out
        integer :: idx

        call tag_to_index(type, idx)

        select case(derivative)
            case(0)
                out = lookup_table_value(x, interp(:, 1), interp(:, idx))
            case(1)
                out = lookup_table_value(x, d_interp(:, 1), d_interp(:, idx))
            case(2)
                out = lookup_table_value(x, dd_interp(:, 1), dd_interp(:, idx))
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
        if (allocated(input)) deallocate(input)
        if (allocated(interp)) deallocate(interp)
        if (allocated(d_interp)) deallocate(d_interp)
        if (allocated(dd_interp)) deallocate(dd_interp)
    end subroutine deallocate_input

end module mod_arrays
