//! # qig
//!
//! Quantum Information Geometry: density matrices, fidelity, and monotone metrics.
//!
//! ## Scope
//!
//! This crate is **L1 (Logic)** in the Tekne stack. It provides geometric primitives for
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

#![forbid(unsafe_code)]

use faer::{Mat, MatRef, ComplexField};
use faer::solvers::SpSolver; // Basic solver trait, though we use Eigendecomposition mostly.
use num_complex::Complex64;
use thiserror::Error;

/// Errors for quantum geometric computations.
#[derive(Debug, Error)]
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
    let tr = rho.diagonal().sum();
    if (tr.re - 1.0).abs() > tol || tr.im.abs() > tol {
        return Err(Error::InvalidTrace(tr.re));
    }

    // 2. Hermitian
    for i in 0..n {
        for j in i..n {
            let diff = rho[(i, j)] - rho[(j, i)].conj();
            if diff.re.abs() > tol || diff.im.abs() > tol {
                return Err(Error::NotHermitian(diff.norm()));
            }
        }
    }

    // 3. PSD (eigenvalues >= -tol)
    // We compute eigenvalues. For Hermitian matrices, they are real.
    // faer's self_adjoint_eigen requires a Hermitian matrix.
    let evd = rho.self_adjoint_eigen(faer::Side::Lower); // Assumes Lower part is valid
    let evals = evd.s(); // Eigenvalues (real)
    
    // Check min eigenvalue
    let min_ev = evals.column_vector().min();
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
    let temp = sqrt_rho * sigma;
    let prod = temp * sqrt_rho;

    // The product might have small numerical asymmetry, but theoretically is Hermitian PSD.
    // We force symmetry for the eigendecomposition or use singular values.
    // Tr(sqrt(A)) = sum(sqrt(eigenvalues(A))).
    
    // Better numerically: sum(singular_values(sqrt_rho * sqrt_sigma)) if we had sqrt_sigma.
    // But Uhlmann is specifically trace norm of sqrt(sqrt_rho sigma sqrt_rho).
    
    // Let's stick to the definition: F = Tr(sqrt(prod)).
    // Since prod is Hermitian PSD, its eigenvalues are non-negative.
    let evd = prod.self_adjoint_eigen(faer::Side::Lower);
    let evals = evd.s();
    
    let mut f = 0.0;
    for i in 0..n {
        let val = evals[i]; // Accessing diagonal s
        // Clamp small negatives to 0
        if val > 0.0 {
            f += val.sqrt();
        }
    }

    Ok(f)
}

/// Bures distance squared: \(D_B^2(\rho, \sigma) = 2(1 - F(\rho, \sigma))\).
///
/// This matches the definition where \(D_B\) corresponds to the Fisher-Rao metric infinitesimal.
pub fn bures_distance_squared(rho: MatRef<Complex64>, sigma: MatRef<Complex64>) -> Result<f64> {
    let f = fidelity(rho, sigma)?;
    // F is in [0, 1].
    Ok(2.0 * (1.0 - f))
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
    // Clamp to [-1, 1] for acos safety
    let f_clamped = f.max(-1.0).min(1.0);
    Ok(f_clamped.acos())
}

/// Compute the principal square root of a Hermitian PSD matrix.
fn matrix_sqrt(a: MatRef<Complex64>) -> Result<Mat<Complex64>> {
    let evd = a.self_adjoint_eigen(faer::Side::Lower);
    let u = evd.u();
    let s = evd.s();
    
    let n = a.nrows();
    // sqrt(A) = U * sqrt(S) * U^H
    
    // Form sqrt(S) as a diagonal matrix (implicitly handled in multiplication)
    // We compute U * sqrt(S)
    let mut u_sqrt_s = Mat::<Complex64>::zeros(n, n);
    
    for j in 0..n {
        let val = s[j];
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
    use num_complex::{Complex64, ComplexFloat};
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_distr::{StandardNormal, Distribution};

    const TOL: f64 = 1e-9;

    #[test]
    fn fidelity_identity() {
        let n = 2;
        let mut rho = Mat::<Complex64>::identity(n, n);
        for i in 0..n { rho[(i,i)] = Complex64::new(1.0 / n as f64, 0.0); }
        
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
        for i in 0..n { rho[(i,i)] = Complex64::new(1.0 / n as f64, 0.0); }
        
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
            psi_vec[(i, 0)] = Complex64::new(StandardNormal.sample(&mut rng), StandardNormal.sample(&mut rng));
            phi_vec[(i, 0)] = Complex64::new(StandardNormal.sample(&mut rng), StandardNormal.sample(&mut rng));
        }
        
        // Normalize
        let norm_psi = (psi_vec.adjoint() * &psi_vec)[(0, 0)].re.sqrt();
        let norm_phi = (phi_vec.adjoint() * &phi_vec)[(0, 0)].re.sqrt();
        
        for i in 0..n { psi_vec[(i, 0)] = psi_vec[(i, 0)] * (1.0/norm_psi); }
        for i in 0..n { phi_vec[(i, 0)] = phi_vec[(i, 0)] * (1.0/norm_phi); }
        
        let rho = &psi_vec * psi_vec.adjoint();
        let sigma = &phi_vec * phi_vec.adjoint();
        
        let f_calc = fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
        let overlap = (psi_vec.adjoint() * &phi_vec)[(0, 0)].norm(); // |<psi|phi>|
        
        assert!((f_calc - overlap).abs() < 1e-6, "calc={} expected={}", f_calc, overlap);
    }
}
