module mod_basis_functions
  use mod_global_variables, only: dp
  implicit none

  private

  interface
    pure function basis_function(x, x0, x1) result(spline)
      use mod_global_variables, only: dp
      real(dp), intent(in) :: x
      real(dp), intent(in) :: x0
      real(dp), intent(in) :: x1
      real(dp) :: spline(4)
    end function basis_function
  end interface

  public :: hquad, dhquad
  public :: hcubic, dhcubic, ddhcubic
  public :: basis_function

contains

  pure function hquad(x, x0, x1) result(quad)
    real(dp), intent(in) :: x
    real(dp), intent(in) :: x0
    real(dp), intent(in) :: x1
    real(dp) :: quad(4)
    real(dp) :: dx

    dx = x1 - x0
    quad(1) = 4.0_dp * (x - x0) * (x1 - x) / dx**2
    quad(2) = 0.0_dp
    quad(3) = (2.0_dp * x - x1 - x0) * (x - x0) / dx**2
    quad(4) = (2.0_dp * x - x1 - x0) * (x - x1) / dx**2
  end function hquad


  pure function dhquad(x, x0, x1) result(dquad)
    real(dp), intent(in) :: x
    real(dp), intent(in) :: x0
    real(dp), intent(in) :: x1
    real(dp) :: dquad(4)
    real(dp) :: dx

    dx = x1 - x0
    dquad(1) = 4.0_dp * (-2.0_dp * x + x1 + x0) / dx**2
    dquad(2) = 0.0_dp
    dquad(3) = (4.0_dp * x - x1 - 3.0_dp * x0) / dx**2
    dquad(4) = (4.0_dp * x - x0 - 3.0_dp * x1) / dx**2
  end function dhquad


  pure function hcubic(x, x0, x1) result(cubic)
    real(dp), intent(in) :: x
    real(dp), intent(in) :: x0
    real(dp), intent(in) :: x1
    real(dp) :: cubic(4)
    real(dp) :: dx

    dx = x1 - x0
    cubic(1) = 3.0_dp * ((x - x0) / dx)**2 - 2.0_dp * ((x - x0) / dx)**3
    cubic(2) = 3.0_dp * ((x1 - x) / dx)**2 - 2.0_dp * ((x1 - x) / dx)**3
    cubic(3) = (x - x1) * ((x - x0) / dx)**2
    cubic(4) = (x - x0) * ((x1 - x) / dx)**2
  end function hcubic


  pure function dhcubic(x, x0, x1) result(dcubic)
    real(dp), intent(in) :: x
    real(dp), intent(in) :: x0
    real(dp), intent(in) :: x1
    real(dp) :: dcubic(4)
    real(dp) :: dx

    dx = x1 - x0
    dcubic(1) = 6.0_dp * (x - x0) / dx**2 - 6.0_dp * (x - x0)**2 / dx**3
    dcubic(2) = -6.0_dp * (x1 - x) / dx**2 + 6.0_dp * (x1 - x)**2 / dx**3
    dcubic(3) = (2.0_dp * (x - x1) * (x - x0) + (x - x0)**2) / dx**2
    dcubic(4) = (2.0_dp * (x - x0) * (x - x1) + (x - x1)**2) / dx**2
  end function dhcubic


  pure function ddhcubic(x, x0, x1) result(ddcubic)
    real(dp), intent(in) :: x
    real(dp), intent(in) :: x0
    real(dp), intent(in) :: x1
    real(dp) :: ddcubic(4)
    real(dp) :: dx

    dx = x1 - x0
    ddcubic(1) =  6.0_dp / dx**2 - 12.0_dp * (x - x0) / dx**3
    ddcubic(2) = 6.0_dp / dx**2 - 12.0_dp * (x1 - x) / dx**3
    ddcubic(3) = (2.0_dp * (x - x0) + 2.0_dp * (x - x1) + 2.0_dp * (x - x0)) / dx**2
    ddcubic(4) = (2.0_dp * (x - x1) + 2.0_dp * (x - x0) + 2.0_dp * (x - x1)) / dx**2
  end function ddhcubic

end module mod_basis_functions
