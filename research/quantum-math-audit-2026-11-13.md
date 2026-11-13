# QCH: quantum/math implementation audit

**Date:** 2026-11-13  
**Scope:** Static review and independent numerical checks of selected mathematical routines on `main`.  
**Purpose:** Identify which equations are executable, mathematically well-defined, and physically interpretable. No existing runtime code is changed.

## Executive result

QCH contains useful classical and quantum-circuit scaffolding, but several routines labeled as quantum coherence, resonance, or quantum-classical unification are not valid physical models as currently written. Some fail at runtime; others return numerical values without satisfying the mathematical requirements of the named object (for example, a density matrix or a convergent integral).

## Verified findings

| Source | Test | Result |
|---|---|---|
| `quantum_sacred_math.py` | Evaluate complex exponential used by `timeline_collapse_function` | **FAIL** — it calls `math.exp(-1j * omega * t)`; Python's `math.exp` rejects complex inputs with `TypeError` |
| `quantum_sacred_math.py` | Check integration domain of `timeline_collapse_function` | **NOT CONVERGENT AS WRITTEN** — its oscillatory integrand has no decaying envelope, yet is integrated over ((-∞,∞)); the ordinary improper integral is not defined |
| `quantum_sacred_math.py` | Check `divine_unity_field` dependence on its field argument | **FAIL** — it returns (4\pi) regardless of the supplied field; this is a constant, not a field-dependent surface integral |
| `quantum_harmonic_resonance.py` | Evaluate rotation-branch conditions for 12 wires | **FAIL** — `i % 3 == 0` captures every index divisible by 6 or 9 first, so the RZ and RX branches are unreachable; only the RY branch can execute |
| `quantum_harmonic_resonance.py` | Check generated field closure over (t\in[0,2\pi]) | **FAIL for closed-loop interpretation** — with irrational golden-ratio modulation, endpoint separation is approximately 1.864, not zero |
| `quantum_classical_unification.py` | Inspect `_create_decoherence_matrix` | **FAIL as density-matrix evolution** — it uses elementwise multiplication of `rho_initial` and an exponential matrix, not (U\rho U^\dagger); the result need not be Hermitian, trace-one, or positive |

## Technical explanation

### 1. Infinite-domain coherence integral

The integrand in `timeline_collapse_function` contains `math.exp` applied to a complex argument. Replacing it with `cmath.exp` or `numpy.exp` would address only the immediate type error. It would not make the infinite integral converge: the coherence function is a sum of undamped sinusoidal terms. A well-posed calculation needs a finite observation window, a physically justified envelope, or a distributional/Fourier-transform definition with explicit conventions.

### 2. Unreachable gate branches

In `create_resonance_circuit`, the ordered conditions are:

- `if i % 3 == 0`: RY
- `elif i % 6 == 0`: RZ
- `elif i % 9 == 0`: RX

Every integer divisible by 6 or 9 is also divisible by 3. Therefore the later branches can never be selected. If distinct 3/6/9 classes are intended, the predicates or their ordering must be redesigned and tested.

### 3. Density-matrix evolution

A closed-system density matrix evolves as (\rho(t)=U(t)\rho(0)U(t)^\dagger), with (U(t)=e^{-iHt/\hbar}). Open-system decoherence requires a specified completely positive, trace-preserving map or a stated master equation (for example, a Lindblad generator). Elementwise multiplication by an exponential matrix is neither construction. Tests should assert Hermiticity, unit trace, and nonnegative eigenvalues at every time step.

### 4. Geometry and resonance labels

The golden-ratio-modulated path is a valid parametric curve, but it does not close over the selected (2\pi) interval. A closed curve needs a compatible period or a different rational winding ratio. Similarly, a frequency value used as a gate angle or spatial multiplier is not automatically a physical oscillator: the model needs units, time dependence, and a measurement protocol.

## Recommendations

1. Replace the infinite-domain integral with a mathematically specified finite-window or damped transform; test convergence as the window changes.
2. Correct the gate branch logic and add a unit test proving every intended gate family is reached.
3. Implement density-matrix evolution using matrix products and a trace-preserving decoherence model; add Hermiticity, trace, and positivity tests.
4. Define the toroidal path's period and winding numbers; test endpoint closure and winding invariants.
5. Keep symbolic, geometric, circuit-simulation, and physical claims in separate layers. A simulator output is not evidence of a biological or material effect without an independently measurable prediction and controls.

## Scope and limitations

This is a scoped review of selected files, not a full QCH audit. It does not establish that all QCH code is incorrect. No quantum hardware, biological system, or material was measured. The findings are about the inspected implementations and their mathematical contracts.
