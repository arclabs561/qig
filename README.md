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

## License

MIT OR Apache-2.0
