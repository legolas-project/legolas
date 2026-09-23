!=================================================================
!> Module defining convenient index retrieval functions on various arrays.
module mod_get_indices
  implicit none

  private

  !> interface to retrieve the index of an element in an array.
  !! @note
  !!     Replace get_index by `findloc` once we drop support for gfortran<9
  !! @endnote
  interface get_index
    module procedure find_index_in_character_array
    module procedure find_indices_in_character_array
  end interface get_index

  public :: get_index

contains

  !> Function to locate the index of a given character in a character array.
  !! Iterates over the elements and returns on the first hit, if no match
  !! was found zero is returned.
  pure function find_index_in_character_array(name, array) result(match_idx)
    !> the name to search for
    character(len=*), intent(in)  :: name
    !> array with the names to search in
    character(len=*), intent(in)  :: array(:)
    !> index of first match
    integer :: match_idx
    integer :: i

    match_idx = 0
    do i = 1, size(array)
      if (array(i) == trim(adjustl(name))) then
        match_idx = i
        exit
      end if
    end do
  end function find_index_in_character_array


  !> Function to locate the indices of an array of characters in another
  !! character array. Returns the indices of the first hit, it no match was
  !! found zero is returned.
  pure function find_indices_in_character_array(names, array) result(match_idxs)
    !> the names to search for
    character(len=*), intent(in)  :: names(:)
    !> array in which to sarch in
    character(len=*), intent(in)  :: array(:)
    !> index of first matches
    integer :: match_idxs(size(names))
    integer :: i

    do i = 1, size(names)
      match_idxs(i) = find_index_in_character_array(names(i), array)
    end do
  end function find_indices_in_character_array

end module mod_get_indices
