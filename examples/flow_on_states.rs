//! Flow matching on quantum states via Bloch sphere (qig + flowmatch).
//!
//! Uses flowmatch's ODE integrator to interpolate between two pure quantum
//! states on the Bloch sphere. The velocity field points toward the target
//! state (linear interpolation in Bloch coordinates).
//!
//! Tracks quantum fidelity F(rho(t), rho_target) over the trajectory.
//! At t=1, fidelity should be ~1.0.
//!
//! flowmatch uses ndarray::Array1<f32>; qig uses faer::Mat<c64>.
//! Conversion happens at the boundary.

use faer::complex_native::c64;
use faer::Mat;
use flowmatch::ode::{integrate_fixed, OdeMethod};
use ndarray::Array1;
use std::f64::consts::PI;

fn main() {
    // Source and target states on the Bloch sphere [theta, phi].
    let source = [PI / 6.0, 0.0]; // near north pole
    let target = [PI / 2.0, PI / 3.0]; // on equator

    let rho_target = bloch_to_density(target[0], target[1]);

    // Velocity field: linear interpolation toward target.
    // v(x, t) = (target - x) / (1 - t + eps)
    let target_f32 = Array1::from_vec(vec![target[0] as f32, target[1] as f32]);

    let x0 = Array1::from_vec(vec![source[0] as f32, source[1] as f32]);
    let steps = 100;
    let dt = 1.0f32 / steps as f32;

    println!("Flow matching on Bloch sphere (qig + flowmatch):");
    println!("  source = [theta={:.3}, phi={:.3}]", source[0], source[1]);
    println!("  target = [theta={:.3}, phi={:.3}]", target[0], target[1]);
    println!("  steps  = {steps}, method = Euler");
    println!();

    // Track fidelity at checkpoints.
    let checkpoints = [0.0, 0.25, 0.5, 0.75, 1.0];
    let mut checkpoint_idx = 0;

    // We'll integrate manually to track intermediate states.
    let mut x = x0.clone();
    let mut t = 0.0f32;

    println!("  t      theta    phi      fidelity");
    println!("  ------ -------- -------- --------");

    // Print initial state.
    print_checkpoint(0.0, &x, &rho_target);

    for _step in 0..steps {
        // Linear velocity toward target, scaled by 1/(1-t+eps).
        let eps = 1e-4f32;
        let scale = 1.0 / (1.0 - t + eps);
        let v = Array1::from_vec(vec![
            (target_f32[0] - x[0]) * scale,
            (target_f32[1] - x[1]) * scale,
        ]);

        // Euler step
        x[0] += dt * v[0];
        x[1] += dt * v[1];
        t += dt;

        // Check if we hit a checkpoint.
        if checkpoint_idx < checkpoints.len() - 1 {
            let next_cp = checkpoints[checkpoint_idx + 1];
            if t >= next_cp - dt / 2.0 {
                print_checkpoint(t, &x, &rho_target);
                checkpoint_idx += 1;
            }
        }
    }

    // Final fidelity check.
    let rho_final = bloch_to_density(x[0] as f64, x[1] as f64);
    let final_fid = qig::fidelity(rho_final.as_ref(), rho_target.as_ref()).unwrap();
    println!();
    println!("  Final fidelity: {final_fid:.6}");
    assert!(
        final_fid > 0.99,
        "Final fidelity should be close to 1.0, got {final_fid}"
    );
    println!("  PASSED: flow reached target state.");
    println!();

    // Also demonstrate using flowmatch::ode::integrate_fixed directly.
    println!("Using flowmatch::ode::integrate_fixed (Heun method):");
    let x0_copy = Array1::from_vec(vec![source[0] as f32, source[1] as f32]);
    let target_copy = target_f32.clone();

    let result = integrate_fixed(OdeMethod::Heun, &x0_copy, 0.0, dt, steps, |x, t| {
        let eps = 1e-4f32;
        let scale = 1.0 / (1.0 - t + eps);
        Ok(Array1::from_vec(vec![
            (target_copy[0] - x[0]) * scale,
            (target_copy[1] - x[1]) * scale,
        ]))
    })
    .unwrap();

    let rho_heun = bloch_to_density(result[0] as f64, result[1] as f64);
    let heun_fid = qig::fidelity(rho_heun.as_ref(), rho_target.as_ref()).unwrap();
    println!(
        "  Heun endpoint  = [theta={:.4}, phi={:.4}]",
        result[0], result[1]
    );
    println!("  Heun fidelity  = {heun_fid:.6}");
    assert!(
        heun_fid > 0.99,
        "Heun fidelity should be close to 1.0, got {heun_fid}"
    );
    println!("  PASSED.");
}

fn print_checkpoint(t: f32, x: &Array1<f32>, rho_target: &Mat<c64>) {
    let rho = bloch_to_density(x[0] as f64, x[1] as f64);
    let fid = qig::fidelity(rho.as_ref(), rho_target.as_ref()).unwrap();
    println!("  {t:.2}     {:.4}   {:.4}   {fid:.4}", x[0], x[1]);
}

/// Convert Bloch sphere coordinates to a 2x2 pure-state density matrix.
///
/// |psi> = [cos(theta/2), e^{i*phi} * sin(theta/2)]
/// rho = |psi><psi|
fn bloch_to_density(theta: f64, phi: f64) -> Mat<c64> {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    let (cp, sp) = (phi.cos(), phi.sin());

    let mut rho = Mat::<c64>::zeros(2, 2);
    rho[(0, 0)] = c64::new(c * c, 0.0);
    rho[(0, 1)] = c64::new(c * s * cp, -c * s * sp);
    rho[(1, 0)] = c64::new(c * s * cp, c * s * sp);
    rho[(1, 1)] = c64::new(s * s, 0.0);
    rho
}
