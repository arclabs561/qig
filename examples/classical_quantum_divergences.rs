//! Classical vs quantum divergences: logp + infogeom + qig.
//!
//! For diagonal density matrices (classical distributions embedded as quantum
//! states), the Bures distance relates to the classical Hellinger distance:
//!
//!   D_Bures(diag(p), diag(q))^2 = 2(1 - sum(sqrt(p_i * q_i)))
//!                                = 2 * H^2(p, q)
//!
//! where H^2 is the squared Hellinger distance (logp::hellinger_squared).
//!
//! This example computes KL divergence (logp), Fisher-Rao distance (infogeom),
//! and Bures distance (qig) on the same pair of distributions, then verifies
//! the Bures-Hellinger identity on diagonal density matrices.

use faer::complex_native::c64;
use faer::Mat;

fn main() {
    let p = [0.7, 0.2, 0.1];
    let q = [0.3, 0.4, 0.3];
    let tol = 1e-12;

    // -- Classical divergences (logp) --
    let kl_pq = logp::kl_divergence(&p, &q, tol).unwrap();
    let kl_qp = logp::kl_divergence(&q, &p, tol).unwrap();
    let hell = logp::hellinger(&p, &q, tol).unwrap();
    let hell_sq = logp::hellinger_squared(&p, &q, tol).unwrap();

    println!("Classical (logp):");
    println!("  KL(p || q) = {kl_pq:.6}");
    println!("  KL(q || p) = {kl_qp:.6}  (asymmetric)");
    println!("  Hellinger  = {hell:.6}");
    println!("  Hellinger^2= {hell_sq:.6}");
    println!();

    // -- Fisher-Rao distance (infogeom) --
    let rao = infogeom::rao_distance_categorical(&p, &q, tol).unwrap();
    println!("Fisher-Rao (infogeom):");
    println!("  d_FR(p, q) = {rao:.6} rad");
    println!();

    // -- Quantum: embed as diagonal density matrices --
    let rho = diag_density(&p);
    let sigma = diag_density(&q);

    let fid = qig::fidelity(rho.as_ref(), sigma.as_ref()).unwrap();
    let bures = qig::bures_distance(rho.as_ref(), sigma.as_ref()).unwrap();
    let bures_sq = qig::bures_distance_squared(rho.as_ref(), sigma.as_ref()).unwrap();
    let angle = qig::bures_angle(rho.as_ref(), sigma.as_ref()).unwrap();

    println!("Quantum (qig) on diagonal density matrices:");
    println!("  Fidelity        = {fid:.6}");
    println!("  Bures distance  = {bures:.6}");
    println!("  Bures dist^2    = {bures_sq:.6}");
    println!("  Bures angle     = {angle:.6} rad");
    println!();

    // -- Verify: Bures distance^2 == 2 * Hellinger^2 for diagonal states --
    //
    // Bures^2 = 2(1 - F), where F = sum(sqrt(p_i * q_i)) = Bhattacharyya coeff.
    // Hellinger^2 = 1 - BC = 1 - sum(sqrt(p_i * q_i)).
    // So Bures^2 = 2 * Hellinger^2.
    let err = (bures_sq - 2.0 * hell_sq).abs();
    println!("Verification:");
    println!("  Bures^2          = {bures_sq:.10}");
    println!("  2 * Hellinger^2  = {:.10}", 2.0 * hell_sq);
    println!("  |difference|     = {err:.2e}");
    assert!(
        err < 1e-10,
        "Bures-Hellinger identity violated: err = {err}"
    );
    println!("  PASSED: Bures^2 == 2 * Hellinger^2 for diagonal density matrices.");
    println!();

    // -- Verify: Bures angle == Fisher-Rao / 2 for diagonal states --
    //
    // Bures angle = arccos(F) = arccos(BC).
    // Fisher-Rao  = 2 * arccos(BC).
    // So Bures angle = Fisher-Rao / 2.
    let err_angle = (angle - rao / 2.0).abs();
    println!("  Bures angle      = {angle:.10}");
    println!("  Fisher-Rao / 2   = {:.10}", rao / 2.0);
    println!("  |difference|     = {err_angle:.2e}");
    assert!(
        err_angle < 1e-10,
        "Bures angle / Fisher-Rao identity violated: err = {err_angle}"
    );
    println!("  PASSED: Bures angle == Fisher-Rao / 2 for diagonal density matrices.");
}

/// Build an n x n diagonal density matrix from a probability vector.
fn diag_density(p: &[f64]) -> Mat<c64> {
    let n = p.len();
    let mut m = Mat::<c64>::zeros(n, n);
    for (i, &pi) in p.iter().enumerate() {
        m[(i, i)] = c64::new(pi, 0.0);
    }
    m
}
