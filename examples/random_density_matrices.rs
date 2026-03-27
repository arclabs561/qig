//! Random density matrices and RMT eigenvalue statistics.
//!
//! Generates random density matrices via Wishart construction (A A^H / tr(A A^H)),
//! computes fidelity/Bures distance statistics over many pairs, and compares
//! eigenvalue distributions against the Marchenko-Pastur prediction.
//!
//! Key observations:
//! - Random density matrices cluster near the maximally mixed state I/d.
//! - Eigenvalue distribution of random density matrices follows a rescaled
//!   Marchenko-Pastur law (the "induced measure" on density matrices).
//!
//! Run: cargo run --example random_density_matrices

use faer::{complex_native::c64 as Complex64, Mat};
use qig::{bures_distance, fidelity};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rmt::{empirical_spectral_density, marchenko_pastur_density, marchenko_pastur_support};

fn main() {
    println!("=== Random Density Matrices and RMT ===\n");

    let d = 8; // Hilbert space dimension
    let n_samples = 200;
    let seed = 42u64;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Generate random density matrices via Wishart: A is d x d complex Gaussian,
    // rho = A A^H / tr(A A^H).
    let matrices: Vec<Mat<Complex64>> = (0..n_samples)
        .map(|_| random_density_matrix(d, &mut rng))
        .collect();

    // ---------------------------------------------------------------
    // Part 1: Fidelity and Bures distance to I/d
    // ---------------------------------------------------------------
    println!("--- Part 1: Distance to maximally mixed state I/{d} ---\n");

    let maximally_mixed = {
        let mut m = Mat::<Complex64>::zeros(d, d);
        for i in 0..d {
            m[(i, i)] = Complex64::new(1.0 / d as f64, 0.0);
        }
        m
    };

    let fidelities_to_mixed: Vec<f64> = matrices
        .iter()
        .map(|rho| fidelity(rho.as_ref(), maximally_mixed.as_ref()).unwrap_or(0.0))
        .collect();

    let bures_to_mixed: Vec<f64> = matrices
        .iter()
        .map(|rho| bures_distance(rho.as_ref(), maximally_mixed.as_ref()).unwrap_or(0.0))
        .collect();

    let mean_fid: f64 = fidelities_to_mixed.iter().sum::<f64>() / n_samples as f64;
    let mean_bures: f64 = bures_to_mixed.iter().sum::<f64>() / n_samples as f64;
    let max_bures: f64 = bures_to_mixed.iter().cloned().fold(0.0_f64, f64::max);

    println!("  {n_samples} random {d}x{d} density matrices");
    println!("  Mean fidelity to I/{d}:     {mean_fid:.4}");
    println!("  Mean Bures distance to I/{d}: {mean_bures:.4}");
    println!("  Max Bures distance to I/{d}:  {max_bures:.4}");
    println!("  (Random states cluster near I/d as d grows)\n");

    // ---------------------------------------------------------------
    // Part 2: Pairwise fidelity statistics
    // ---------------------------------------------------------------
    println!("--- Part 2: Pairwise fidelity statistics ---\n");

    let n_pairs = 100;
    let mut pair_fids = Vec::with_capacity(n_pairs);
    let mut pair_bures = Vec::with_capacity(n_pairs);
    for i in 0..n_pairs {
        let j = (i + 1) % n_samples;
        let f = fidelity(matrices[i].as_ref(), matrices[j].as_ref()).unwrap_or(0.0);
        let b = bures_distance(matrices[i].as_ref(), matrices[j].as_ref()).unwrap_or(0.0);
        pair_fids.push(f);
        pair_bures.push(b);
    }

    let mean_pair_f: f64 = pair_fids.iter().sum::<f64>() / n_pairs as f64;
    let mean_pair_b: f64 = pair_bures.iter().sum::<f64>() / n_pairs as f64;

    println!("  {n_pairs} random pairs:");
    println!("  Mean pairwise fidelity:     {mean_pair_f:.4}");
    println!("  Mean pairwise Bures dist:   {mean_pair_b:.4}");
    println!("  (High fidelity = states are close; typical for large d)\n");

    // ---------------------------------------------------------------
    // Part 3: Eigenvalue distribution vs Marchenko-Pastur
    // ---------------------------------------------------------------
    println!("--- Part 3: Eigenvalue distribution vs Marchenko-Pastur ---\n");

    // Collect all eigenvalues across samples. For a d x d density matrix
    // generated from a d x d complex Gaussian A, the eigenvalues of rho
    // follow a distribution related to MP with ratio gamma = d/d = 1.
    // We use a rectangular A (d x 2d) to get a non-trivial ratio.
    let rect_dim = 2 * d; // A is d x rect_dim
    let rect_matrices: Vec<Vec<f64>> = (0..n_samples)
        .map(|_| {
            let rho = random_density_matrix_rect(d, rect_dim, &mut rng);
            eigenvalues_hermitian(&rho)
        })
        .collect();

    let all_eigs: Vec<f64> = rect_matrices.into_iter().flatten().collect();

    // The eigenvalues of rho = A A^H / tr(A A^H) are related to the
    // Marchenko-Pastur distribution. After rescaling by d (since tr(rho)=1),
    // the empirical eigenvalue density of d * lambda matches MP with
    // gamma = d / rect_dim.
    let gamma = d as f64 / rect_dim as f64;
    let rescaled_eigs: Vec<f64> = all_eigs.iter().map(|&e| e * d as f64).collect();

    let n_bins = 12;
    let (centers, densities) = empirical_spectral_density(&rescaled_eigs, n_bins);
    let (mp_lo, mp_hi) = marchenko_pastur_support(gamma, 1.0);

    println!("  Eigenvalue distribution of d*lambda (d={d}, gamma={gamma:.2})");
    println!("  MP support: [{mp_lo:.4}, {mp_hi:.4}]");
    println!();
    println!("  Empirical vs MP density:");
    println!("  {:>8} | {:>8} {:>8}", "lambda", "empirical", "MP");
    println!("  {:-<8}-+-{:-<8}-{:-<8}", "", "", "");

    for (c, emp_d) in centers.iter().zip(densities.iter()) {
        let mp_d = marchenko_pastur_density(*c, gamma, 1.0);
        let bar_emp = "#".repeat((emp_d * 20.0).min(40.0) as usize);
        println!("  {c:8.4} | {emp_d:8.4} {mp_d:8.4} {bar_emp}");
    }
    println!();
    println!("  The empirical distribution tracks the MP prediction.");
    println!("  Deviations are expected at d={d} (finite-size effects).\n");

    // ---------------------------------------------------------------
    // Part 4: Concentration around I/d
    // ---------------------------------------------------------------
    println!("--- Part 4: Concentration as dimension grows ---\n");

    for dim in [4, 8, 16, 32] {
        let mut rng_inner = rand::rngs::StdRng::seed_from_u64(seed);
        let mixed = {
            let mut m = Mat::<Complex64>::zeros(dim, dim);
            for i in 0..dim {
                m[(i, i)] = Complex64::new(1.0 / dim as f64, 0.0);
            }
            m
        };
        let n = 50;
        let sum_bures: f64 = (0..n)
            .map(|_| {
                let rho = random_density_matrix(dim, &mut rng_inner);
                bures_distance(rho.as_ref(), mixed.as_ref()).unwrap_or(0.0)
            })
            .sum();
        let avg = sum_bures / n as f64;
        println!("  d={dim:3}: mean Bures distance to I/d = {avg:.4}");
    }
    println!("  (Distance decreases with d: measure concentrates around I/d)");
}

/// Generate a random d x d density matrix from a square Gaussian.
fn random_density_matrix(d: usize, rng: &mut rand::rngs::StdRng) -> Mat<Complex64> {
    let mut a = Mat::<Complex64>::zeros(d, d);
    for i in 0..d {
        for j in 0..d {
            a[(i, j)] = Complex64::new(StandardNormal.sample(rng), StandardNormal.sample(rng));
        }
    }
    let mut rho = &a * a.adjoint();
    let tr: f64 = (0..d).map(|i| rho[(i, i)].re).sum();
    let inv_tr = 1.0 / tr;
    for i in 0..d {
        for j in 0..d {
            rho[(i, j)] *= inv_tr;
        }
    }
    rho
}

/// Generate a density matrix from a rectangular d x k Gaussian (k > d gives
/// a less degenerate spectrum, better matching MP predictions).
fn random_density_matrix_rect(d: usize, k: usize, rng: &mut rand::rngs::StdRng) -> Mat<Complex64> {
    let mut a = Mat::<Complex64>::zeros(d, k);
    for i in 0..d {
        for j in 0..k {
            a[(i, j)] = Complex64::new(StandardNormal.sample(rng), StandardNormal.sample(rng));
        }
    }
    let mut rho = &a * a.adjoint();
    let tr: f64 = (0..d).map(|i| rho[(i, i)].re).sum();
    let inv_tr = 1.0 / tr;
    for i in 0..d {
        for j in 0..d {
            rho[(i, j)] *= inv_tr;
        }
    }
    rho
}

/// Extract real eigenvalues of a Hermitian matrix, sorted ascending.
fn eigenvalues_hermitian(m: &Mat<Complex64>) -> Vec<f64> {
    let n = m.nrows();
    let evals = m.as_ref().selfadjoint_eigenvalues(faer::Side::Lower);
    let mut v: Vec<f64> = evals.iter().copied().take(n).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}
