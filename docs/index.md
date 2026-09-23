---
title: Legolas
layout: splash
last_modified_at: 2025-01-27
intro:
  - image_path: /assets/images/logo_legolas_1280x640.png
    excerpt:
      Legolas, short for **L**arge **E**igensystem **G**enerator for **O**ne-dimensional p**LAS**mas, is a novel
      finite element code tailored to (M)HD spectroscopy of 3D Cartesian and cylindrical equilibria with nontrivial 1D variation.
      The code includes a myriad of physical effects, most of which are fully user-customisable.

feature_row:
  - image_path: /assets/images/fortran_python.png
    excerpt:
      Legolas is written in object-oriented Fortran (2008 standard) with a complementary Python framework (Pylbo)
      used for post-processing, data analysis, interfacing and parallel running. Both source codes are extensively
      documented and we provide a detailed guide on how to get started with the code.
    title: Getting started
    url: /getting-started/installation
    btn_label: I want to get started
    btn_class: btn--primary
  - image_path: /assets/images/slack_github.png
    excerpt:
      We are happy to answer any questions you may have in using Legolas and/or Pylbo. Feel free to open an issue
      in the [GitHub repository](https://github.com/legolas-project/legolas) or start a [Discussion thread](https://github.com/legolas-project/legolas/discussions)
      there. We also have a dedicated Slack workspace, which you can also use to ask questions or have a nice chat with the developers.
      Email addresses can be found on the About page.
    title: Contact
    url: https://join.slack.com/t/the-legolas-code/shared_invite/zt-3ouzfyn74-~lyDKVuUGExNZrMWKMsY_A
    btn_label: Join legolas @ Slack!
    btn_class: btn--primary
  - image_path: /assets/images/prominent_logo.png
    excerpt:
      Legolas was developed thanks to funding from the European Research Council (ERC) under the European
      Unions Horizon 2020 research and innovation programme,
      Grant agreement No. 833251 PROMINENT ERC-ADG 2018; from the VSC (Flemish Supercomputer Center),
      funded by the Research Foundation – Flanders (FWO) and the Flemish Government – department EWI;
      and from Internal Funds KU Leuven, project C14/19/089 TRACESpace.
    title: Funding
    url: https://erc-prominent.github.io
    btn_label: Read more on PROMINENT
    btn_class: btn--primary

what_legolas_can_do:
  - image_path: /assets/images/example_suydam.png
    excerpt:
      Legolas can do a multitude of things, ranging from full spectrum calculations to eigenfunctions
      of specific modes, to full-on parametric studies of various equilibrium configurations in different geometries.
      You can take a look at all the various [equilibria](/general/equilibria/) that are already implemented, or read
      more on our About section.
    title: What Legolas can do
    url: /about
    btn_label: Read more
    btn_class: btn--primary

using_legolas:
  - image_path: /assets/images/history.png
    excerpt:
      Legolas is the result of months and months of development and testing. Since this takes a lot of effort and time,
      we kindly ask that _**the first published peer-reviewed paper from applying Legolas is done in co-authorship with at least
      one of the original authors**_. Additionally, if you use Legolas in a publication we kindly request that you cite the code paper.
    title: Using Legolas
    url: https://ui.adsabs.harvard.edu/abs/2020arXiv201014148C/exportcitation
    btn_label: BibTex citation
    btn_class: btn--primary
---

{% include feature_row id="intro" type="center" %}

{% include feature_row %}

{% include feature_row id="what_legolas_can_do" type="right" %}

{% include feature_row id="using_legolas" type="left" %}
