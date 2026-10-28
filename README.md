# qig

Quantum information geometry primitives.

## Install

Use the Git source while the crate is unpublished:

```toml
[dependencies]
qig = { git = "https://github.com/arclabs561/qig", rev = "5ee707b9694389cc8bbdc61286ed6a7d547d6d17" }
faer = "0.19" # qig's public API takes faer matrix types
```

## Example

```rust
use faer::Mat;
use faer::complex_native::c64;
use qig::fidelity;

// Two orthogonal pure states: |0><0| and |1><1|.
let mut rho = Mat::<c64>::zeros(2, 2);
rho[(0, 0)] = c64::new(1.0, 0.0);

let mut sigma = Mat::<c64>::zeros(2, 2);
sigma[(1, 1)] = c64::new(1.0, 0.0);

// Uhlmann fidelity: 0 for orthogonal states, 1 for identical ones.
let f = fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
assert!(f.abs() < 1e-9);
```

`fidelity` returns root fidelity (the unsquared convention). For pure states,
this is the absolute inner product, not its square. Inputs must be valid density
matrices: Hermitian, positive semidefinite, and trace one.

## Examples

See [examples/README.md](examples/README.md) for mathematical demonstrations
with captured output.

## License

MIT OR Apache-2.0
