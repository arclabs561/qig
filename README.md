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

Runnable examples live in [`examples/`](examples/). These are mathematical
demonstrations: each verifies a geometric identity that motivates the library,
not a production workload.

- `bures_manifold` parameterizes the Bures manifold via the Bloch sphere (qig + skel), the natural geometry for comparing single-qubit states.
- `bures_wasserstein` shows the Bures distance between density matrices equals the L2-Wasserstein distance between the corresponding Gaussians, the bridge from quantum fidelity to optimal transport.
- `classical_quantum_divergences` relates classical and quantum divergences for diagonal states (logp + infogeom + qig), showing the classical case as a special case of the quantum one.
- `flow_on_states` runs flow matching between quantum states on the Bloch sphere (qig + flowmatch), generative modeling applied to state preparation.
- `random_density_matrices` samples random density matrices and reports the random-matrix-theory eigenvalue statistics used to reason about entanglement and typicality.

## License

MIT OR Apache-2.0
