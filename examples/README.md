# qig examples

Each example is runnable from the repo root. Output excerpts below are real,
captured from release runs.

## Which example should I run?

| I want to... | Example |
|---|---|
| Relate classical and quantum divergences | `classical_quantum_divergences` |
| Check Bures-Wasserstein equivalence | `bures_wasserstein` |
| Parameterize the Bures manifold on the Bloch sphere | `bures_manifold` |
| Run flow matching between quantum states | `flow_on_states` |
| Inspect random density matrix statistics | `random_density_matrices` |

## Divergence Identities

### `classical_quantum_divergences`: how do diagonal quantum states recover classical geometry?

Embeds two categorical distributions as diagonal density matrices, then checks
the Bures-Hellinger and Bures-angle/Fisher-Rao identities.

```bash
cargo run --release --example classical_quantum_divergences
```

```text
Classical (logp):
  KL(p || q) = 0.344618
  KL(q || p) = 0.352653  (asymmetric)
  Hellinger  = 0.292736
  Hellinger^2= 0.085695

Quantum (qig) on diagonal density matrices:
  Fidelity        = 0.914305
  Bures distance  = 0.413992
  Bures dist^2    = 0.171389

Verification:
  Bures^2          = 0.1713892745
  2 * Hellinger^2  = 0.1713892745
  |difference|     = 0.00e0
  PASSED: Bures^2 == 2 * Hellinger^2 for diagonal density matrices.
```

### `bures_wasserstein`: does qig match the Gaussian W2 formula?

Computes the Bures-Wasserstein distance for two centered Gaussian covariance
matrices, then cross-checks the diagonal case against `wass`.

```bash
cargo run --release --example bures_wasserstein
```

```text
W_2(N(0,A), N(0,B)):
  W_2^2 = 0.9293118550
  W_2   = 0.9640082235

qig (density matrices A/tr(A), B/tr(B)):
  Fidelity F = 0.8762283587
  d_B^2      = 0.2475432826

Equivalence check:
  W_2^2 (direct)       = 0.9293118550
  W_2^2 (from fidelity)= 0.9293118550
  |difference|         = 1.78e-15
  PASSED

Diagonal case: A = diag(4, 9), B = diag(1, 4)
  W_2^2 (qig inline)  = 2.0000000000
  W_2   (wass crate)  = 1.4142135382
  PASSED -- qig's Bures, the inline closed form, and the wass crate all agree
```

## Bloch Sphere

### `bures_manifold`: can the Bloch sphere implement `skel::Manifold`?

Implements exp/log/transport on pure qubit states parameterized by Bloch sphere
coordinates, then compares Bures angle with sphere geodesic distance.

```bash
cargo run --release --example bures_manifold
```

```text
Bloch sphere manifold (skel::Manifold):
  x = [0.7854, 0.0000]
  y = [1.0472, 1.5708]
  v = log_x(y) = [-0.457121, 1.583515]
  exp_x(v)     = [1.047198, 1.570796]
  roundtrip error = 0.00e0
  PASSED: exp_x(log_x(y)) = y

Bures distance (qig) vs sphere geodesic:
  Fidelity       = 0.822664
  Bures angle    = 0.604715
  Sphere dist    = 1.209429
  Sphere dist/2  = 0.604715
```

### `flow_on_states`: can flow matching reach a target quantum state?

Uses `flowmatch` to integrate a velocity field over Bloch coordinates, while
`qig` measures fidelity to the target density matrix.

```bash
cargo run --release --example flow_on_states
```

```text
Flow matching on Bloch sphere (qig + flowmatch):
  source = [theta=0.524, phi=0.000]
  target = [theta=1.571, phi=1.047]
  steps  = 100, method = Euler

  t      theta    phi      fidelity
  ------ -------- -------- --------
  0.00     0.5236   0.0000   0.7906
  0.50     1.0471   0.5235   0.9354
  1.00     1.5707   1.0471   1.0000

  Final fidelity: 1.000000
  PASSED: flow reached target state.
```

## Random States

### `random_density_matrices`: what do random density matrices look like?

Samples Wishart density matrices, compares them to the maximally mixed state,
and checks the empirical eigenvalue density against Marchenko-Pastur.

```bash
cargo run --release --example random_density_matrices
```

```text
=== Random Density Matrices and RMT ===

  200 random 8x8 density matrices
  Mean fidelity to I/8:     0.8513
  Mean Bures distance to I/8: 0.5443

  100 random pairs:
  Mean pairwise fidelity:     0.7584
  Mean pairwise Bures dist:   0.6930

  Eigenvalue distribution of d*lambda (d=8, gamma=0.50)
  MP support: [0.0858, 2.9142]

  d=  4: mean Bures distance to I/d = 0.5245
  d=  8: mean Bures distance to I/d = 0.5475
  d= 16: mean Bures distance to I/d = 0.5494
  d= 32: mean Bures distance to I/d = 0.5496
  (For this square-Wishart ensemble, the distance is stable across d)
```
