---
title: Source code documentation
layout: single
sidebar:
  nav: "leftcontents"
last_modified_at: 2023-04-13
---

Most of the internal source codes of both Legolas and the post-processing framework Pylbo contain
extensive docstrings used to automatically generate documentation.
For Pylbo we are currently slowly transitioning towards using type-hints instead of fully-fledged (and sometimes cumbersome)
documentation. Methods and variable names should natively be explicit and clear into what is being done.

However, since Legolas is written in Fortran and Pylbo is written in Python we use
language-specific tools: [FORD](https://github.com/Fortran-FOSS-Programmers/ford) for
the generated Legolas documentation, and [Sphinx](https://www.sphinx-doc.org/en/master/) for the
generated Pylbo documentation.

Both of these have their own webpages and are accessible through this page using the buttons below.
These are generated completely automatic and are hence kept independent of the main website.
If you feel something is missing or needs a better explanation please let us know, preferably through
an [issue on GitHub](https://github.com/legolas-project/legolas/issues).

[Legolas docs](../ford/index.html){: .btn .btn--primary .btn--large }
[Pylbo docs](../sphinx/index.html){: .btn .btn--primary .btn--large }
