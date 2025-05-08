! =============================================================================
!> Simple module, containing only version-related stuff. Versioning is done in a
!! separate module to avoid cluttering the commit history of for example
!! <tt>mod_global_variables</tt> or <tt>mod_output</tt> every time an update
!! to the code is done.
!! The Legolas version is added to the datfile as a string and has
!! common MAJOR.MINOR.PATCH formatting.
!! This means:
!!
!! - MAJOR: increased for API changes, possibly breaking backwards compatibility.
!! - MINOR: increased for new features and extensions
!! - PATCH: increased for bugfixes and minor edits
module mod_version
  implicit none

  !> legolas version number
  character(len=10), parameter    :: LEGOLAS_VERSION = "2.2.1"

end module mod_version
