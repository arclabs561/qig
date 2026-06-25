# qig

Quantum information geometry primitives.

[![crates.io](https://img.shields.io/crates/v/qig.svg)](https://crates.io/crates/qig)
[![docs.rs](https://docs.rs/qig/badge.svg)](https://docs.rs/qig)

## Install

```toml
[dependencies]
qig = "0.1"
```

## Example

```rust
use qig::fidelity;

let rho = [0.9, 0.1]; // Diagonal of density matrix
let sigma = [0.1, 0.9];

// Classical fidelity (Bhattacharyya coefficient)
let f = fidelity(&rho, &sigma);
assert!(f > 0.0 && f < 1.0);
```

## Examples

Runnable examples live in [`examples/`](examples/):

- `bures_manifold` parameterizes the Bures manifold via the Bloch sphere (qig + skel).
- `bures_wasserstein` shows the Bures-Wasserstein equivalence for centered Gaussians.
- `classical_quantum_divergences` relates classical and quantum divergences (logp + infogeom + qig).
- `flow_on_states` runs flow matching between quantum states on the Bloch sphere (qig + flowmatch).
- `random_density_matrices` samples random density matrices and reports random-matrix-theory eigenvalue statistics.

## License

MIT OR Apache-2.0
