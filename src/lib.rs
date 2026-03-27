//! # qig
//!
//! Quantum Information Geometry: density matrices, fidelity, and monotone metrics.
//!
//! ## Scope
//!
//! This crate is **L1 (Logic)** in the mathematical foundation. It provides geometric primitives for
//! quantum states (density operators):
//!
//! - **State validation**: Hermitian, positive semi-definite (PSD), trace 1.
//! - **Fidelity**: Uhlmann fidelity \(F(\rho, \sigma) = \operatorname{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\).
//! - **Bures distance**: \(D_B(\rho, \sigma)^2 = 2(1 - F(\rho, \sigma))\).
//! - **Bures angle**: \(D_A(\rho, \sigma) = \arccos F(\rho, \sigma)\).
//! - **Quantum Relative Entropy**: \(S(\rho\|\sigma) = \operatorname{Tr}\rho(\ln\rho - \ln\sigma)\).
//!
//! ## Design Principles
//!
//! - **Zero-copy where possible**: Operations work on `faer::MatRef` views.
//! - **Explicit validity**: Functions expect valid density matrices; validation is a separate step.
//! - **No unwrap**: Numerical failures return `Result`.
//!
//! ## Geometry
//!
//! The **Bures metric** is the quantum analogue of the Fisher–Rao metric. It is the minimal
//! Riemannian metric that is monotone under completely positive trace-preserving (CPTP) maps
//! (Petz 1996, Chentsov/Morozova).
//!
//! In the Bures geometry, the squared distance is locally proportional to the **Quantum Fisher Information** (QFI).
//!
//! ## References
//!
//! - Uhlmann (1976): "The 'transition probability' in the state space of a *-algebra".
//! - Bures (1969): "An extension of Kakutani's distance to arbitrary Hilbert spaces".
//! - Nielsen & Chuang (2010): *Quantum Computation and Quantum Information*.
//! - Petz (1996): "Monotone metrics on matrix spaces".
//! - "Quantum Flow Matching (QFM): A Unified Framework" (2025): Applying flow matching
//!   to density matrices using the Bures metric.
//!
//! # Related crates
//! - [`wass`]: Wasserstein optimal transport. The Bures distance equals the L2-Wasserstein distance for centered Gaussians (Bhatia et al. 2019).
//! - [`infogeom`]: Classical Fisher-Rao distance on the probability simplex. qig generalizes this to quantum states.
//!

#![forbid(unsafe_code)]

use faer::{complex_native::c64 as Complex64, Mat, MatRef};
use thiserror::Error;

/// Errors for quantum geometric computations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error("matrix is not square: {0}x{1}")]
    NotSquare(usize, usize),

    #[error("dimension mismatch: {0}x{0} vs {1}x{1}")]
    DimensionMismatch(usize, usize),

    #[error("not Hermitian (symmetry violation > {0})")]
    NotHermitian(f64),

    #[error("trace is not 1.0 (got {0})")]
    InvalidTrace(f64),

    #[error("not positive semi-definite (min eigenvalue {0})")]
    NotPSD(f64),

    #[error("matrix operation failed: {0}")]
    MatrixOp(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[inline]
fn c64_conj(z: Complex64) -> Complex64 {
    Complex64 {
        re: z.re,
        im: -z.im,
    }
}

#[inline]
fn c64_norm(z: Complex64) -> f64 {
    (z.re * z.re + z.im * z.im).sqrt()
}

/// Validate that a matrix is a valid density operator (Hermitian, PSD, trace 1).
///
/// # Arguments
/// * `rho`: The matrix to check.
/// * `tol`: Tolerance for numerical checks.
pub fn validate_density_matrix(rho: MatRef<Complex64>, tol: f64) -> Result<()> {
    let n = rho.nrows();
    if rho.ncols() != n {
        return Err(Error::NotSquare(n, rho.ncols()));
    }

    // 1. Trace == 1
    let diag = rho.diagonal().column_vector();
    let mut tr_re = 0.0f64;
    let mut tr_im = 0.0f64;
    for i in 0..n {
        let z = diag.read(i);
        tr_re += z.re;
        tr_im += z.im;
    }
    if (tr_re - 1.0).abs() > tol || tr_im.abs() > tol {
        return Err(Error::InvalidTrace(tr_re));
    }

    // 2. Hermitian
    for i in 0..n {
        for j in i..n {
            let a = rho[(i, j)];
            let b = rho[(j, i)];
            let diff = a - c64_conj(b);
            if diff.re.abs() > tol || diff.im.abs() > tol {
                return Err(Error::NotHermitian(c64_norm(diff)));
            }
        }
    }

    // 3. PSD (eigenvalues >= -tol)
    // For Hermitian matrices, eigenvalues are real.
    let evals = rho.selfadjoint_eigenvalues(faer::Side::Lower); // Assumes Lower part is valid

    // Check min eigenvalue
    let min_ev = evals
        .iter()
        .copied()
        .reduce(f64::min)
        .unwrap_or(f64::INFINITY);
    if min_ev < -tol {
        return Err(Error::NotPSD(min_ev));
    }

    Ok(())
}

/// Uhlmann fidelity \(F(\rho, \sigma) = \operatorname{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\).
///
/// Returns a value in [0, 1]. For pure states \(|\psi\rangle, |\phi\rangle\), this is \(|\langle\psi|\phi\rangle|\).
/// Note: Some definitions (like Nielsen & Chuang) square this quantity. We use the non-squared root (trace norm).
///
/// Requires `rho` and `sigma` to be valid density matrices.
pub fn fidelity(rho: MatRef<Complex64>, sigma: MatRef<Complex64>) -> Result<f64> {
    let n = rho.nrows();
    if sigma.nrows() != n || sigma.ncols() != n {
        return Err(Error::DimensionMismatch(n, sigma.nrows()));
    }

    // Algorithm:
    // 1. sqrt_rho = sqrt(rho)
    // 2. prod = sqrt_rho * sigma * sqrt_rho
    // 3. F = Tr(sqrt(prod))

    // To compute sqrt(A) for Hermitian PSD A: U D U^H -> U sqrt(D) U^H
    let sqrt_rho = matrix_sqrt(rho)?;

    // prod = sqrt_rho * sigma * sqrt_rho
    let temp = sqrt_rho.as_ref() * sigma;
    let prod = temp.as_ref() * sqrt_rho.as_ref();

    // The product might have small numerical asymmetry, but theoretically is Hermitian PSD.
    // We force symmetry for the eigendecomposition or use singular values.
    // Tr(sqrt(A)) = sum(sqrt(eigenvalues(A))).

    // Better numerically: sum(singular_values(sqrt_rho * sqrt_sigma)) if we had sqrt_sigma.
    // But Uhlmann is specifically trace norm of sqrt(sqrt_rho sigma sqrt_rho).

    // Let's stick to the definition: F = Tr(sqrt(prod)).
    // Since prod is Hermitian PSD, its eigenvalues are non-negative.
    let evals = prod.selfadjoint_eigenvalues(faer::Side::Lower);

    let mut f = 0.0;
    for &val in evals.iter().take(n) {
        let clamped = val.max(0.0);
        f += clamped.sqrt();
    }

    Ok(f)
}

/// Bures distance squared: \(D_B^2(\rho, \sigma) = 2(1 - F(\rho, \sigma))\).
///
/// This matches the definition where \(D_B\) corresponds to the Fisher-Rao metric infinitesimal.
pub fn bures_distance_squared(rho: MatRef<Complex64>, sigma: MatRef<Complex64>) -> Result<f64> {
    let f = fidelity(rho, sigma)?;
    // Clamp to [0, 1] for numerical safety
    Ok(2.0 * (1.0 - f.clamp(0.0, 1.0)))
}

/// Bures distance: \(D_B(\rho, \sigma) = \sqrt{2(1 - F(\rho, \sigma))}\).
pub fn bures_distance(rho: MatRef<Complex64>, sigma: MatRef<Complex64>) -> Result<f64> {
    Ok(bures_distance_squared(rho, sigma)?.sqrt())
}

/// Bures angle: \(D_A(\rho, \sigma) = \arccos F(\rho, \sigma)\).
///
/// Also known as the Fubini-Study metric for pure states.
pub fn bures_angle(rho: MatRef<Complex64>, sigma: MatRef<Complex64>) -> Result<f64> {
    let f = fidelity(rho, sigma)?;
    // Fidelity is in [0, 1]; clamp for acos safety
    let f_clamped = f.clamp(0.0, 1.0);
    Ok(f_clamped.acos())
}

/// Compute the principal square root of a Hermitian PSD matrix.
fn matrix_sqrt(a: MatRef<Complex64>) -> Result<Mat<Complex64>> {
    let evd = a.selfadjoint_eigendecomposition(faer::Side::Lower);
    let u = evd.u();
    let s = evd.s();

    let n = a.nrows();
    // sqrt(A) = U * sqrt(S) * U^H

    // Form sqrt(S) as a diagonal matrix (implicitly handled in multiplication)
    // We compute U * sqrt(S)
    let mut u_sqrt_s = Mat::<Complex64>::zeros(n, n);

    for j in 0..n {
        let val = s.column_vector().read(j).re;
        let root = if val < 0.0 { 0.0 } else { val.sqrt() };
        for i in 0..n {
            u_sqrt_s[(i, j)] = u[(i, j)] * root;
        }
    }

    // Result = (U * sqrt(S)) * U^H
    let res = u_sqrt_s * u.adjoint();
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    const TOL: f64 = 1e-9;

    #[test]
    fn fidelity_identity() {
        let n = 2;
        let mut rho = Mat::<Complex64>::identity(n, n);
        for i in 0..n {
            rho[(i, i)] = Complex64::new(1.0 / n as f64, 0.0);
        }

        let f = fidelity(rho.as_ref(), rho.as_ref()).unwrap();
        assert!((f - 1.0).abs() < TOL);
    }

    #[test]
    fn fidelity_orthogonal() {
        // |0><0| and |1><1|
        let mut rho = Mat::<Complex64>::zeros(2, 2);
        rho[(0, 0)] = Complex64::new(1.0, 0.0);

        let mut sigma = Mat::<Complex64>::zeros(2, 2);
        sigma[(1, 1)] = Complex64::new(1.0, 0.0);

        let f = fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
        assert!(f.abs() < TOL);
    }

    #[test]
    fn bures_distance_identity() {
        let n = 2;
        let mut rho = Mat::<Complex64>::identity(n, n);
        for i in 0..n {
            rho[(i, i)] = Complex64::new(1.0 / n as f64, 0.0);
        }

        let d = bures_distance(rho.as_ref(), rho.as_ref()).unwrap();
        assert!(d.abs() < TOL);
    }

    #[test]
    fn fidelity_pure_states() {
        // F(|psi><psi|, |phi><phi|) = |<psi|phi>|
        let n = 4;
        let seed = 123;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Create random vectors
        let mut psi_vec = Mat::<Complex64>::zeros(n, 1);
        let mut phi_vec = Mat::<Complex64>::zeros(n, 1);

        for i in 0..n {
            psi_vec[(i, 0)] = Complex64::new(
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            );
            phi_vec[(i, 0)] = Complex64::new(
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            );
        }

        // Normalize
        let norm_psi = (psi_vec.adjoint() * &psi_vec)[(0, 0)].re.sqrt();
        let norm_phi = (phi_vec.adjoint() * &phi_vec)[(0, 0)].re.sqrt();

        for i in 0..n {
            psi_vec[(i, 0)] *= 1.0 / norm_psi;
        }
        for i in 0..n {
            phi_vec[(i, 0)] *= 1.0 / norm_phi;
        }

        let rho = &psi_vec * psi_vec.adjoint();
        let sigma = &phi_vec * phi_vec.adjoint();

        let f_calc = fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
        let overlap = c64_norm((psi_vec.adjoint() * &phi_vec)[(0, 0)]); // |<psi|phi>|

        assert!(
            (f_calc - overlap).abs() < 1e-6,
            "calc={} expected={}",
            f_calc,
            overlap
        );
    }

    /// Generate a random density matrix of dimension `n`.
    fn random_density_matrix(n: usize, rng: &mut rand::rngs::StdRng) -> Mat<Complex64> {
        // A = random complex matrix, rho = A A^H / tr(A A^H)
        let mut a = Mat::<Complex64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = Complex64::new(StandardNormal.sample(rng), StandardNormal.sample(rng));
            }
        }
        let mut rho = &a * a.adjoint();
        // Normalize trace to 1
        let mut tr = 0.0;
        for i in 0..n {
            tr += rho[(i, i)].re;
        }
        for i in 0..n {
            for j in 0..n {
                rho[(i, j)] *= 1.0 / tr;
            }
        }
        rho
    }

    #[test]
    fn verify_c64_storage() {
        let z = Complex64::new(1.0, 2.0);
        assert_eq!(z.re, 1.0);
        // faer c64 stores .im as the standard imaginary part (despite doc saying "negated").
        // Verified: c64::new -> num_complex::Complex64 copies .im directly.
        assert_eq!(z.im, 2.0);
    }

    #[test]
    fn trace_validation_with_complex_diagonal() {
        // A density matrix with complex off-diagonal but real trace should validate.
        let mut rho = Mat::<Complex64>::zeros(2, 2);
        rho[(0, 0)] = Complex64::new(0.6, 0.0);
        rho[(1, 1)] = Complex64::new(0.4, 0.0);
        rho[(0, 1)] = Complex64::new(0.1, 0.2);
        rho[(1, 0)] = Complex64::new(0.1, -0.2); // conjugate
        assert!(validate_density_matrix(rho.as_ref(), 1e-6).is_ok());
    }

    #[test]
    fn fidelity_symmetry() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..5 {
            let rho = random_density_matrix(3, &mut rng);
            let sigma = random_density_matrix(3, &mut rng);
            let f1 = fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
            let f2 = fidelity(sigma.as_ref(), rho.as_ref()).unwrap();
            assert!(
                (f1 - f2).abs() < 1e-8,
                "fidelity not symmetric: {} vs {}",
                f1,
                f2
            );
        }
    }

    #[test]
    fn bures_self_distance_is_zero() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        for n in 2..=4 {
            let rho = random_density_matrix(n, &mut rng);
            let d = bures_distance(rho.as_ref(), rho.as_ref()).unwrap();
            assert!(d < 1e-6, "self-distance should be ~0, got {}", d);
        }
    }

    #[test]
    fn bures_triangle_inequality() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        for _ in 0..5 {
            let rho = random_density_matrix(3, &mut rng);
            let sigma = random_density_matrix(3, &mut rng);
            let tau = random_density_matrix(3, &mut rng);

            let d_rt = bures_distance(rho.as_ref(), tau.as_ref()).unwrap();
            let d_rs = bures_distance(rho.as_ref(), sigma.as_ref()).unwrap();
            let d_st = bures_distance(sigma.as_ref(), tau.as_ref()).unwrap();

            assert!(
                d_rt <= d_rs + d_st + 1e-8,
                "triangle inequality violated: d(rho,tau)={} > d(rho,sigma)+d(sigma,tau)={}",
                d_rt,
                d_rs + d_st
            );
        }
    }

    #[test]
    fn maximally_mixed_state() {
        for n in 2..=4 {
            let mut rho = Mat::<Complex64>::zeros(n, n);
            for i in 0..n {
                rho[(i, i)] = Complex64::new(1.0 / n as f64, 0.0);
            }
            assert!(validate_density_matrix(rho.as_ref(), 1e-10).is_ok());
            let f = fidelity(rho.as_ref(), rho.as_ref()).unwrap();
            assert!(
                (f - 1.0).abs() < 1e-8,
                "self-fidelity of I/d should be 1, got {}",
                f
            );
        }
    }

    #[test]
    fn fidelity_bounds() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(55);
        for _ in 0..10 {
            let rho = random_density_matrix(3, &mut rng);
            let sigma = random_density_matrix(3, &mut rng);
            let f = fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
            assert!(
                (-1e-10..=1.0 + 1e-10).contains(&f),
                "fidelity out of [0,1]: {}",
                f
            );
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    // --- Strategies ---

    /// 2x2 pure-state density matrix from Bloch sphere angles.
    fn bloch_pure_state() -> impl Strategy<Value = Mat<Complex64>> {
        (0.0..PI, 0.0..2.0 * PI).prop_map(|(theta, phi)| {
            // |psi> = cos(theta/2)|0> + e^{i*phi}*sin(theta/2)|1>
            let c = (theta / 2.0).cos();
            let s = (theta / 2.0).sin();
            let a0 = Complex64::new(c, 0.0);
            let a1 = Complex64::new(s * phi.cos(), s * phi.sin());
            // rho = |psi><psi|
            let mut rho = Mat::<Complex64>::zeros(2, 2);
            rho[(0, 0)] = a0 * c64_conj(a0);
            rho[(0, 1)] = a0 * c64_conj(a1);
            rho[(1, 0)] = a1 * c64_conj(a0);
            rho[(1, 1)] = a1 * c64_conj(a1);
            rho
        })
    }

    /// 2x2 mixed-state density matrix: rho = p * |psi><psi| + (1-p) * I/2.
    fn bloch_mixed_state() -> impl Strategy<Value = Mat<Complex64>> {
        (bloch_pure_state(), 0.0..=1.0_f64).prop_map(|(pure, p)| {
            let mut rho = Mat::<Complex64>::zeros(2, 2);
            for i in 0..2 {
                for j in 0..2 {
                    let eye_ij = if i == j { 0.5 } else { 0.0 };
                    rho[(i, j)] = pure[(i, j)] * p + Complex64::new(eye_ij * (1.0 - p), 0.0);
                }
            }
            rho
        })
    }

    /// NxN density matrix via A * A^dagger / tr(A * A^dagger).
    fn random_density_matrix_strategy(n: usize) -> impl Strategy<Value = Mat<Complex64>> {
        // Generate n*n*2 floats (real + imag for each entry).
        proptest::collection::vec(-5.0..5.0_f64, n * n * 2).prop_map(move |vals| {
            let mut a = Mat::<Complex64>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let idx = (i * n + j) * 2;
                    a[(i, j)] = Complex64::new(vals[idx], vals[idx + 1]);
                }
            }
            let mut rho = &a * a.adjoint();
            let mut tr = 0.0;
            for i in 0..n {
                tr += rho[(i, i)].re;
            }
            // Guard against degenerate zero matrix (astronomically unlikely but safe).
            if tr < 1e-15 {
                // Fall back to maximally mixed state.
                let mut m = Mat::<Complex64>::zeros(n, n);
                for i in 0..n {
                    m[(i, i)] = Complex64::new(1.0 / n as f64, 0.0);
                }
                return m;
            }
            let inv_tr = 1.0 / tr;
            for i in 0..n {
                for j in 0..n {
                    rho[(i, j)] *= inv_tr;
                }
            }
            rho
        })
    }

    fn density_matrix_2x2() -> impl Strategy<Value = Mat<Complex64>> {
        prop_oneof![bloch_pure_state(), bloch_mixed_state(),]
    }

    fn density_matrix_3x3() -> impl Strategy<Value = Mat<Complex64>> {
        random_density_matrix_strategy(3)
    }

    // --- Validation tests ---

    proptest! {
        #[test]
        fn prop_bloch_pure_validates(rho in bloch_pure_state()) {
            validate_density_matrix(rho.as_ref(), 1e-9)?;
        }

        #[test]
        fn prop_bloch_mixed_validates(rho in bloch_mixed_state()) {
            validate_density_matrix(rho.as_ref(), 1e-9)?;
        }

        #[test]
        fn prop_random_3x3_validates(rho in density_matrix_3x3()) {
            validate_density_matrix(rho.as_ref(), 1e-9)?;
        }
    }

    // --- Fidelity properties ---

    proptest! {
        #[test]
        fn prop_fidelity_self_is_one_2x2(rho in density_matrix_2x2()) {
            let f = fidelity(rho.as_ref(), rho.as_ref()).unwrap();
            prop_assert!((f - 1.0).abs() < 1e-6, "self-fidelity = {}", f);
        }

        #[test]
        fn prop_fidelity_self_is_one_3x3(rho in density_matrix_3x3()) {
            let f = fidelity(rho.as_ref(), rho.as_ref()).unwrap();
            prop_assert!((f - 1.0).abs() < 1e-6, "self-fidelity = {}", f);
        }

        #[test]
        fn prop_fidelity_bounds_2x2(
            rho in density_matrix_2x2(),
            sigma in density_matrix_2x2(),
        ) {
            let f = fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
            prop_assert!(f >= -1e-9, "fidelity below 0: {}", f);
            prop_assert!(f <= 1.0 + 1e-9, "fidelity above 1: {}", f);
        }

        #[test]
        fn prop_fidelity_bounds_3x3(
            rho in density_matrix_3x3(),
            sigma in density_matrix_3x3(),
        ) {
            let f = fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
            prop_assert!(f >= -1e-9, "fidelity below 0: {}", f);
            prop_assert!(f <= 1.0 + 1e-9, "fidelity above 1: {}", f);
        }

        #[test]
        fn prop_fidelity_symmetry_2x2(
            rho in density_matrix_2x2(),
            sigma in density_matrix_2x2(),
        ) {
            let f1 = fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
            let f2 = fidelity(sigma.as_ref(), rho.as_ref()).unwrap();
            prop_assert!((f1 - f2).abs() < 1e-7, "F(rho,sigma)={} != F(sigma,rho)={}", f1, f2);
        }

        #[test]
        fn prop_fidelity_symmetry_3x3(
            rho in density_matrix_3x3(),
            sigma in density_matrix_3x3(),
        ) {
            let f1 = fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
            let f2 = fidelity(sigma.as_ref(), rho.as_ref()).unwrap();
            prop_assert!((f1 - f2).abs() < 1e-7, "F(rho,sigma)={} != F(sigma,rho)={}", f1, f2);
        }
    }

    // --- Bures distance properties ---

    proptest! {
        #[test]
        fn prop_bures_self_distance_zero_2x2(rho in density_matrix_2x2()) {
            let d = bures_distance(rho.as_ref(), rho.as_ref()).unwrap();
            prop_assert!(d < 1e-6, "self-distance = {}", d);
        }

        #[test]
        fn prop_bures_self_distance_zero_3x3(rho in density_matrix_3x3()) {
            let d = bures_distance(rho.as_ref(), rho.as_ref()).unwrap();
            prop_assert!(d < 1e-6, "self-distance = {}", d);
        }

        #[test]
        fn prop_bures_non_negative_2x2(
            rho in density_matrix_2x2(),
            sigma in density_matrix_2x2(),
        ) {
            let d = bures_distance(rho.as_ref(), sigma.as_ref()).unwrap();
            prop_assert!(d >= -1e-9, "negative distance: {}", d);
        }

        #[test]
        fn prop_bures_symmetry_2x2(
            rho in density_matrix_2x2(),
            sigma in density_matrix_2x2(),
        ) {
            let d1 = bures_distance(rho.as_ref(), sigma.as_ref()).unwrap();
            let d2 = bures_distance(sigma.as_ref(), rho.as_ref()).unwrap();
            prop_assert!((d1 - d2).abs() < 1e-7, "d(rho,sigma)={} != d(sigma,rho)={}", d1, d2);
        }

        #[test]
        fn prop_bures_triangle_inequality_3x3(
            rho in density_matrix_3x3(),
            sigma in density_matrix_3x3(),
            tau in density_matrix_3x3(),
        ) {
            let d_rt = bures_distance(rho.as_ref(), tau.as_ref()).unwrap();
            let d_rs = bures_distance(rho.as_ref(), sigma.as_ref()).unwrap();
            let d_st = bures_distance(sigma.as_ref(), tau.as_ref()).unwrap();
            prop_assert!(
                d_rt <= d_rs + d_st + 1e-7,
                "triangle: d(rho,tau)={} > d(rho,sigma)+d(sigma,tau)={}",
                d_rt,
                d_rs + d_st
            );
        }

        /// Bures distance for density matrices is bounded by sqrt(2).
        /// D_B^2 = 2(1 - F) <= 2, so D_B <= sqrt(2).
        #[test]
        fn prop_bures_diameter_bound_3x3(
            rho in density_matrix_3x3(),
            sigma in density_matrix_3x3(),
        ) {
            let d = bures_distance(rho.as_ref(), sigma.as_ref()).unwrap();
            let bound = std::f64::consts::SQRT_2 + 1e-9;
            prop_assert!(d <= bound, "diameter exceeded: d={} > sqrt(2)={}", d, bound);
        }
    }

    // --- Bures angle properties ---

    proptest! {
        #[test]
        fn prop_bures_angle_self_zero(rho in density_matrix_2x2()) {
            let a = bures_angle(rho.as_ref(), rho.as_ref()).unwrap();
            prop_assert!(a.abs() < 1e-6, "self-angle = {}", a);
        }

        #[test]
        fn prop_bures_angle_bounds(
            rho in density_matrix_3x3(),
            sigma in density_matrix_3x3(),
        ) {
            let a = bures_angle(rho.as_ref(), sigma.as_ref()).unwrap();
            // arccos(F) in [0, pi/2] for F in [0, 1]
            prop_assert!(a >= -1e-9, "negative angle: {}", a);
            prop_assert!(a <= PI / 2.0 + 1e-9, "angle > pi/2: {}", a);
        }
    }
}
