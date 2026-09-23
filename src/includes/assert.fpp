!===================================================================
!> Defines the `assert(<cond>)` macro which is a no-op when `NDEBUG`
!! is set, else it evaluates &lt;cond&gt; and logs an error if
!! &lt;cond&gt; is false. This error describes the assertion and
!! specifies the location of the assertion.
!!
!!### Usage
!! This file defines a macro and `use`s the [[mod_assert]] module,
!! it should thus be `#include`ed between the start of a program
!! unit and `implicit none`. Also note that `#include <assert.fpp>`
!! cannot have any whitespace before it.
!!
!! @warning
!!     Unfortunately the compilation error when
!!     `#include <assert.fpp>` is in the wrong location is not very
!!     helpful, so please double check.
!! @endwarning
!!
!! After which one can assert a condition &lt;cond&gt; with
!! `assert(<cond>)`.
!!
!!### Example
!!```fortran
!!module mod_a
!!#include <assert.fpp>
!!  use mod_b
!!  implicit none
!!
!!contains
!!
!!  [...]
!!  assert(3 > 2)
!!  [...]
!!
!!end module mod_a
!!```

#ifndef NDEBUG

use mod_assert, only: assert_macro_mod_assert_assert => assert
#if defined(__GFORTRAN__)
#define assert(cond) call assert_macro_mod_assert_assert(cond, "cond", __FILE__, __LINE__)
#else
#define assert(cond) call assert_macro_mod_assert_assert(cond, #cond, __FILE__, __LINE__)
#endif

#else

#define assert(cond)

#endif
