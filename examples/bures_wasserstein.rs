//! Bures-Wasserstein equivalence for centered Gaussians.
//!
//! The Bures distance on positive definite matrices equals the L2-Wasserstein
//! distance between centered Gaussians with those covariance matrices:
//!
//!   W_2(N(0, A), N(0, B))^2 = tr(A) + tr(B) - 2 * tr(sqrt(sqrt(A) * B * sqrt(A)))
//!
//! This is exactly the squared Bures distance d_B(A, B)^2 when A, B are
//! interpreted as quantum states (scaled to unit trace).
//!
//! For density matrices (trace 1), the Bures distance uses the Uhlmann
//! fidelity: d_B^2 = 2(1 - F). For general positive definite matrices
//! (arbitrary trace), the direct formula above applies.
//!
//! This example:
//! 1. Constructs two 2x2 real symmetric positive definite covariance matrices.
//! 2. Computes d_B via qig (normalizing to density matrices).
//! 3. Computes W_2 via the closed-form formula.
//! 4. Verifies they agree up to the trace normalization.
//!
//! Reference: Bhatia, Jain & Lim (2019), "On the Bures-Wasserstein distance
//! between positive definite matrices", Expositiones Mathematicae 37(2).

use faer::complex_native::c64;
use faer::Mat;

/// Matrix square root of a 2x2 real symmetric PD matrix via eigendecomposition.
///
/// For a 2x2 symmetric matrix [[a, b], [b, d]], eigenvalues are:
///   lambda = ((a+d) +/- sqrt((a-d)^2 + 4b^2)) / 2
fn sym2_sqrt(m: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let a = m[0][0];
    let b = m[0][1];
    let d = m[1][1];

    let tr = a + d;
    let det = a * d - b * b;
    let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();

    let l1 = ((tr + disc) / 2.0).max(0.0);
    let l2 = ((tr - disc) / 2.0).max(0.0);

    let s1 = l1.sqrt();
    let s2 = l2.sqrt();

    // Reconstruct via Cayley-Hamilton: sqrt(M) = (M + sqrt(det)*I) / (s1 + s2)
    // where sqrt(det) = s1*s2 and s1+s2 = tr(sqrt(M)).
    let s_tr = s1 + s2;
    let s_det = s1 * s2; // = sqrt(det(M))

    if s_tr < 1e-15 {
        return [[0.0; 2]; 2];
    }

    let inv = 1.0 / s_tr;
    [[(a + s_det) * inv, b * inv], [b * inv, (d + s_det) * inv]]
}

/// Multiply two 2x2 matrices.
fn mul2(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn trace2(m: &[[f64; 2]; 2]) -> f64 {
    m[0][0] + m[1][1]
}

/// W_2^2(N(0, A), N(0, B)) = tr(A) + tr(B) - 2 * tr(sqrt(sqrt(A) * B * sqrt(A)))
fn w2_squared_gaussians(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2]) -> f64 {
    let sqrt_a = sym2_sqrt(a);
    let prod = mul2(&mul2(&sqrt_a, b), &sqrt_a); // sqrt(A) * B * sqrt(A)
    let sqrt_prod = sym2_sqrt(&prod);
    trace2(a) + trace2(b) - 2.0 * trace2(&sqrt_prod)
}

/// Embed a real symmetric 2x2 matrix as a complex density matrix (normalize trace to 1).
fn to_density(m: &[[f64; 2]; 2]) -> Mat<c64> {
    let tr = m[0][0] + m[1][1];
    let mut rho = Mat::<c64>::zeros(2, 2);
    rho[(0, 0)] = c64::new(m[0][0] / tr, 0.0);
    rho[(0, 1)] = c64::new(m[0][1] / tr, 0.0);
    rho[(1, 0)] = c64::new(m[1][0] / tr, 0.0);
    rho[(1, 1)] = c64::new(m[1][1] / tr, 0.0);
    rho
}

fn main() {
    // Two 2x2 real symmetric positive definite covariance matrices.
    // A = [[2, 0.5], [0.5, 1]]   (eigenvalues ~2.14, 0.86)
    // B = [[1, -0.3], [-0.3, 3]] (eigenvalues ~3.03, 0.97)
    let a = [[2.0, 0.5], [0.5, 1.0]];
    let b = [[1.0, -0.3], [-0.3, 3.0]];

    // -- W2 via closed-form --
    let w2_sq = w2_squared_gaussians(&a, &b);
    let w2 = w2_sq.sqrt();

    println!("Bures-Wasserstein equivalence (2D centered Gaussians)");
    println!("======================================================");
    println!();
    println!(
        "Covariance A = [[{}, {}], [{}, {}]]",
        a[0][0], a[0][1], a[1][0], a[1][1]
    );
    println!(
        "Covariance B = [[{}, {}], [{}, {}]]",
        b[0][0], b[0][1], b[1][0], b[1][1]
    );
    println!();
    println!("W_2(N(0,A), N(0,B)):");
    println!("  W_2^2 = {w2_sq:.10}");
    println!("  W_2   = {w2:.10}");

    // -- Bures via qig --
    // qig works with density matrices (trace 1), so we normalize.
    // The Bures distance formula for density matrices rho = A/tr(A), sigma = B/tr(B) gives:
    //   d_B(rho, sigma)^2 = 2(1 - F(rho, sigma))
    //
    // The W2 formula for the unnormalized matrices A, B is:
    //   W_2^2 = tr(A) + tr(B) - 2 * tr(sqrt(sqrt(A) B sqrt(A)))
    //
    // If we factor out traces: let t_a = tr(A), t_b = tr(B), then
    //   tr(sqrt(sqrt(A) B sqrt(A))) = sqrt(t_a * t_b) * F(A/t_a, B/t_b)
    //
    // because sqrt(sqrt(A/t_a) * (B/t_b) * sqrt(A/t_a)) scales as sqrt(t_a * t_b) * ...
    // Actually, let's verify directly:
    //   sqrt(A) = sqrt(t_a) * sqrt(A/t_a)   (NO -- sqrt doesn't scale linearly)
    //
    // The correct relation: for PD matrices,
    //   F(A/t_a, B/t_b) = tr(sqrt(sqrt(A/t_a) * (B/t_b) * sqrt(A/t_a)))
    //
    //   tr(sqrt(sqrt(A) * B * sqrt(A)))
    //     = tr(sqrt(t_a * sqrt(A/t_a) * t_b * (B/t_b) * sqrt(A/t_a) * ... ))
    //
    // Actually sqrt(A) = sqrt(t_a * (A/t_a)) != sqrt(t_a) * sqrt(A/t_a).
    // Let's just verify numerically that the two formulas agree.

    let rho_a = to_density(&a);
    let rho_b = to_density(&b);
    let fid = qig::fidelity(rho_a.as_ref(), rho_b.as_ref()).unwrap();
    let bures_sq = qig::bures_distance_squared(rho_a.as_ref(), rho_b.as_ref()).unwrap();

    println!();
    println!("qig (density matrices A/tr(A), B/tr(B)):");
    println!("  Fidelity F = {fid:.10}");
    println!("  d_B^2      = {bures_sq:.10}");

    // -- Relate the two --
    // For general PD matrices A, B, the Bures distance is:
    //   d_B(A, B)^2 = tr(A) + tr(B) - 2 * tr(sqrt(sqrt(A) B sqrt(A)))
    //
    // This IS the W_2^2 formula. They are the same quantity.
    //
    // For normalized matrices (rho = A/t_a, sigma = B/t_b), qig gives:
    //   d_B(rho, sigma)^2 = tr(rho) + tr(sigma) - 2 * F(rho, sigma)
    //                      = 1 + 1 - 2*F = 2(1-F)
    //
    // The un-normalized W_2^2 can be recovered from the fidelity:
    //   W_2^2 = t_a + t_b - 2 * sqrt(t_a * t_b) * F(rho, sigma)
    //
    // ... but only if the cross-term scales as sqrt(t_a * t_b) * F.
    // Let's verify this scaling numerically.

    let t_a = trace2(&a);
    let t_b = trace2(&b);
    let w2_sq_from_fidelity = t_a + t_b - 2.0 * (t_a * t_b).sqrt() * fid;

    println!();
    println!("Equivalence check:");
    println!("  tr(A) = {t_a}, tr(B) = {t_b}");
    println!("  W_2^2 (direct)       = {w2_sq:.10}");
    println!("  W_2^2 (from fidelity)= {w2_sq_from_fidelity:.10}");

    let err = (w2_sq - w2_sq_from_fidelity).abs();
    println!("  |difference|         = {err:.2e}");
    assert!(err < 1e-8, "W_2^2 mismatch: {err}");
    println!("  PASSED");

    // Also verify with identity (W2 = 0 when A = B).
    let w2_self = w2_squared_gaussians(&a, &a);
    println!();
    println!("Self-distance: W_2^2(A, A) = {w2_self:.2e}");
    assert!(w2_self < 1e-12, "self-distance should be ~0");
    println!("  PASSED");

    // -- Second pair: diagonal matrices (easy analytic check) --
    // For diagonal A = diag(a1, a2), B = diag(b1, b2):
    //   W_2^2 = (sqrt(a1) - sqrt(b1))^2 + (sqrt(a2) - sqrt(b2))^2
    let a_diag = [[4.0, 0.0], [0.0, 9.0]];
    let b_diag = [[1.0, 0.0], [0.0, 4.0]];
    let w2_sq_diag = w2_squared_gaussians(&a_diag, &b_diag);
    let expected = (2.0 - 1.0_f64).powi(2) + (3.0 - 2.0_f64).powi(2); // (2-1)^2 + (3-2)^2 = 2
    println!();
    println!("Diagonal case: A = diag(4, 9), B = diag(1, 4)");
    println!("  W_2^2 (computed) = {w2_sq_diag:.10}");
    println!("  W_2^2 (analytic) = {expected:.10}");
    let err_diag = (w2_sq_diag - expected).abs();
    println!("  |difference|     = {err_diag:.2e}");
    assert!(err_diag < 1e-10, "diagonal case mismatch: {err_diag}");
    println!("  PASSED");

    println!();
    println!("Bhatia, Jain & Lim (2019): Bures metric = L2-Wasserstein for centered Gaussians.");
}
