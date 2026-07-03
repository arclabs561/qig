# qig

Quantum information geometry primitives.

[![crates.io](https://img.shields.io/crates/v/qig.svg)](https://crates.io/crates/qig)
[![docs.rs](https://docs.rs/qig/badge.svg)](https://docs.rs/qig)

## Install

```toml
[dependencies]
qig = "0.1"
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

## Examples

See [examples/README.md](examples/README.md) for mathematical demonstrations
with captured output.

## License

MIT OR Apache-2.0
