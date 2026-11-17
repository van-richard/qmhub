Gradients after coordinate recentering
======================================

QMHub's input readers call ``System.wrap_positions()`` before evaluating each
geometry. This subtracts the arithmetic QM centroid from every atom, followed
by periodic image wrapping when a cell is present. The output gradient must
differentiate the energy through that coordinate transformation.

The missing derivative was present in the original baseline ``qmhub2``
(``82c6d46``) and ``qmhub2-mlpui`` (``df11ca2``) installations. A periodic analytic test using
the original 24-cubed PME mesh reproduced the same discrepancy in both, even
without MLPUI or Torch. Its maximum tested discrepancy was
``7.46e-6`` hartree/bohr, about ``0.0089`` kcal/mol/angstrom. This example does
not quantify errors in other geometries or previous production simulations.

Coordinate derivative
---------------------

Let ``R`` denote the input coordinates, ``X`` the wrapped coordinates, and
``N_QM`` the number of QM atoms. Away from image changes, the transformation is

.. math::

   X_i = R_i - \frac{1}{N_{\rm QM}}\sum_{a\in\rm QM}R_a - Ln_i.

For a gradient ``g`` evaluated with respect to ``X``, the corresponding input
gradient is

.. math::

   g^{\rm input}_a = g^{\rm wrapped}_a
   - \mathbf{1}_{a\in\rm QM}\frac{\sum_i g^{\rm wrapped}_i}{N_{\rm QM}}.

The sum includes every atom represented in the QMHub gradient. Only QM columns
receive the correction. It follows from differentiating the energy already
evaluated at recentered coordinates; it is not a force-balancing operation on
the complete Amber force array.

Finite-grid PME has small translation dependence, so the omitted term is
measurable. A denser mesh reduces that dependence but does not supply the missing
coordinate derivative. Mesh convergence remains a separate accuracy check.

Runtime behavior and API
-------------------------

``System.map_energy_gradient(gradient)`` accepts a floating-point ``3 x N``
gradient of an energy evaluated after ``wrap_positions()``. It returns an
independent NumPy array in the same units and layout. It does not modify the
input or cached engine/model results.

``QMMM.return_results()`` applies this mapping lazily after simulation
aggregation and before text, binary, or FIFO serialization. FIFO's first
geometry arrives inside its output loop, so eagerly evaluating or copying the
simulation gradient before that loop would use the wrong frame. Each frame's
normal dependency invalidation refreshes the mapped result.

The mapping is linear, so existing engine sums, scaling, and MD/MTS combinations
are retained. Direct ``Model``, ``Result``, and ``Simulation`` users still receive
gradients with respect to the coordinates supplied to those objects. Call the
mapping explicitly only when those coordinates were recentered with the System
transformation. An initially zero QM centroid still requires the derivative;
its numerical value does not make the transformation an identity.

Numerical consequences
-----------------------

**Returned QM gradients change.** Energies and MM gradients are unchanged at the
same input geometry. Nonperiodic translation-invariant analytic cases receive
only floating-point roundoff changes. Future trajectories can differ because
the corrected gradient now differentiates the reported recentered energy.

Coordinate wrapping, PME mesh defaults, helPME binaries, units, signs, backend
commands, and exchange formats are unchanged. Local finite differences apply
away from wrapping and cutoff discontinuities.

Validation and deployment
---------------------------

Regression tests cover an independently differentiated origin-dependent
potential, the zero-centroid case, immutable cached gradients, text/binary
outputs, repeated FIFO frames, repeated output calls, engine sums, scaling,
and MD/MTS steps. The original upstream MLPUI periodic test retains its mesh and
tolerance; fixed-origin and nonperiodic controls are kept separately.

Initial candidate validation passed on 2026-11-16:

* All 66 repository tests passed with native helPME enabled, including the 15 new
  recentering regressions. Those 15 also passed against the built wheel.
* All 14 upstream MLPUI analytic/synthetic tests passed against both candidate
  source and wheel, including the previously failing periodic case.
* The periodic fixture's nine finite-difference components passed at the original
  mesh and tolerance, with maximum error ``1.87e-12`` hartree/bohr.
* Fixture energies and MM gradients were bitwise identical before and after the
  change. The separate direct-Model control retained bitwise-identical energy
  and raw gradients.

The candidate version is ``1.0.0+227.gdf11ca2.recenter.a0362ddacc1e``. Its
manifest, wheel, source patch, guarded import records, and comparison results
are retained under ``build/recentering-fix-20261116``. The wheel is built from
the immutable ``df11ca2`` source plus exactly the two runtime file changes;
new regression sources and their hashes are recorded alongside validation.

The candidate above records the initial validation, rather than an installed
release. Subsequent installations should build verified wheels from the final
committed branch snapshots and record their exact versions and hashes in local
deployment manifests. Real Amber/Q-Chem execution and production validation
still require separately authorized runs.
