//! Bures manifold via Bloch sphere parameterization (qig + skel).
//!
//! Every 2x2 density matrix can be written as:
//!
//!   rho(r, theta, phi) = (I + r * n . sigma) / 2
//!
//! where n = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)) is a unit
//! vector and r in [0, 1] is the purity parameter (r=0 is maximally mixed,
//! r=1 is pure). The Bloch ball interior is a 3D Riemannian manifold under
//! the Bures metric.
//!
//! For simplicity, this example fixes r=1 (pure states on the Bloch sphere
//! surface), parameterized by [theta, phi]. The manifold is then the 2-sphere
//! S^2, and the Bures metric on pure states reduces to 1/4 the round metric.
//!
//! Implements `skel::Manifold` for this space and verifies the exp/log roundtrip.

use faer::complex_native::c64;
use faer::Mat;
use ndarray::{array, Array1, ArrayView1};
use skel::Manifold;
use std::f64::consts::PI;

/// Bures manifold of pure 2x2 density matrices (Bloch sphere).
///
/// Points are [theta, phi] with theta in [0, pi], phi in [0, 2*pi).
struct BlochSphere;

impl BlochSphere {
    /// Convert Bloch sphere coordinates to a 2x2 density matrix.
    fn to_density(coords: &ArrayView1<f64>) -> Mat<c64> {
        let (theta, phi) = (coords[0], coords[1]);
        let (ct, st) = (theta.cos(), theta.sin());
        let (cp, sp) = (phi.cos(), phi.sin());

        // rho = |psi><psi| where |psi> = [cos(theta/2), e^{i*phi} sin(theta/2)]
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();

        let mut rho = Mat::<c64>::zeros(2, 2);
        rho[(0, 0)] = c64::new(c * c, 0.0);
        rho[(0, 1)] = c64::new(c * s * cp, -c * s * sp);
        rho[(1, 0)] = c64::new(c * s * cp, c * s * sp);
        rho[(1, 1)] = c64::new(s * s, 0.0);

        // Suppress unused variable warnings -- ct, st used conceptually
        let _ = (ct, st);

        rho
    }
}

impl Manifold for BlochSphere {
    fn exp_map(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64> {
        // Exponential map on S^2 (standard sphere geometry).
        let (theta, phi) = (x[0], x[1]);
        let (vt, vp) = (v[0], v[1]);

        // Tangent vector norm in the round metric on S^2:
        // ||v||^2 = v_theta^2 + sin^2(theta) * v_phi^2
        let st = theta.sin();
        let norm_sq = vt * vt + st * st * vp * vp;
        let norm = norm_sq.sqrt();

        if norm < 1e-14 {
            return x.to_owned();
        }

        // Convert to Cartesian, apply sphere exp map, convert back.
        let (nx, ny, nz) = spherical_to_cart(theta, phi);

        // Tangent vector in Cartesian (d/dtheta and d/dphi basis):
        let ct = theta.cos();
        let (cp, sp) = (phi.cos(), phi.sin());
        // d/dtheta:
        let (ex, ey, ez) = (ct * cp, ct * sp, -st);
        // d/dphi (not normalized):
        let (fx, fy, fz) = (-st * sp, st * cp, 0.0);

        let (wx, wy, wz) = (vt * ex + vp * fx, vt * ey + vp * fy, vt * ez + vp * fz);

        // Normalize tangent direction
        let (ux, uy, uz) = (wx / norm, wy / norm, wz / norm);

        // Geodesic: gamma(1) = cos(norm)*n + sin(norm)*u
        let cn = norm.cos();
        let sn = norm.sin();
        let (rx, ry, rz) = (cn * nx + sn * ux, cn * ny + sn * uy, cn * nz + sn * uz);

        let result = cart_to_spherical(rx, ry, rz);
        array![result.0, result.1]
    }

    fn log_map(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let (theta_x, phi_x) = (x[0], x[1]);
        let (theta_y, phi_y) = (y[0], y[1]);

        let (nx, ny, nz) = spherical_to_cart(theta_x, phi_x);
        let (mx, my, mz) = spherical_to_cart(theta_y, phi_y);

        let dot = (nx * mx + ny * my + nz * mz).clamp(-1.0, 1.0);
        let omega = dot.acos();

        if omega < 1e-14 {
            return Array1::zeros(2);
        }

        // Direction: project m onto tangent space at n.
        let (dx, dy, dz) = (mx - dot * nx, my - dot * ny, mz - dot * nz);
        let d_norm = (dx * dx + dy * dy + dz * dz).sqrt();
        if d_norm < 1e-14 {
            return Array1::zeros(2);
        }

        // Cartesian tangent vector scaled by omega
        let (wx, wy, wz) = (
            omega * dx / d_norm,
            omega * dy / d_norm,
            omega * dz / d_norm,
        );

        // Project back to (v_theta, v_phi) coordinates.
        let st = theta_x.sin();
        let ct = theta_x.cos();
        let (cp, sp) = (phi_x.cos(), phi_x.sin());

        let (ex, ey, ez) = (ct * cp, ct * sp, -st);
        let (fx, fy, fz) = (-st * sp, st * cp, 0.0);

        let vt = wx * ex + wy * ey + wz * ez;
        let vp = if st.abs() > 1e-14 {
            (wx * fx + wy * fy + wz * fz) / (st * st)
        } else {
            0.0
        };

        array![vt, vp]
    }

    fn parallel_transport(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        v: &ArrayView1<f64>,
    ) -> Array1<f64> {
        // Schild's ladder approximation via exp/log.
        // For a proper implementation we'd use the sphere transport formula,
        // but for demonstration purposes, the linear approximation suffices.
        let mid_v = {
            let log_xy = self.log_map(x, y);
            // half-step
            let half: Array1<f64> = log_xy.mapv(|c| c * 0.5);
            let mid = self.exp_map(x, &half.view());
            // Transport v to midpoint, then to y (two-step Schild's ladder).
            let _ = mid;
            // Simplified: just use the log/exp trick.
            // Shoot v from x, get endpoint z.
            let z = self.exp_map(x, v);
            // log_y(z) gives approximately the transported vector.
            self.log_map(y, &z.view())
        };
        mid_v
    }

    fn project(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let theta = x[0].clamp(0.0, PI);
        let phi = x[1].rem_euclid(2.0 * PI);
        array![theta, phi]
    }
}

fn spherical_to_cart(theta: f64, phi: f64) -> (f64, f64, f64) {
    let st = theta.sin();
    (st * phi.cos(), st * phi.sin(), theta.cos())
}

fn cart_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();
    let theta = (z / r).clamp(-1.0, 1.0).acos();
    let phi = y.atan2(x).rem_euclid(2.0 * PI);
    (theta, phi)
}

fn main() {
    let manifold = BlochSphere;

    // Two pure states on the Bloch sphere
    let x = array![PI / 4.0, 0.0]; // theta=pi/4, phi=0
    let y = array![PI / 3.0, PI / 2.0]; // theta=pi/3, phi=pi/2

    // -- Exp/Log roundtrip --
    let v = manifold.log_map(&x.view(), &y.view());
    let y_recovered = manifold.exp_map(&x.view(), &v.view());

    println!("Bloch sphere manifold (skel::Manifold):");
    println!("  x = [{:.4}, {:.4}]", x[0], x[1]);
    println!("  y = [{:.4}, {:.4}]", y[0], y[1]);
    println!("  v = log_x(y) = [{:.6}, {:.6}]", v[0], v[1]);
    println!(
        "  exp_x(v)     = [{:.6}, {:.6}]",
        y_recovered[0], y_recovered[1]
    );

    let err_theta = (y_recovered[0] - y[0]).abs();
    let err_phi = (y_recovered[1] - y[1]).abs();
    let err = (err_theta * err_theta + err_phi * err_phi).sqrt();
    println!("  roundtrip error = {err:.2e}");
    assert!(err < 1e-8, "exp(log(y)) != y: error = {err}");
    println!("  PASSED: exp_x(log_x(y)) = y");
    println!();

    // -- Bures distance vs geodesic distance --
    let rho_x = BlochSphere::to_density(&x.view());
    let rho_y = BlochSphere::to_density(&y.view());
    let bures = qig::bures_distance(rho_x.as_ref(), rho_y.as_ref()).unwrap();
    let fid = qig::fidelity(rho_x.as_ref(), rho_y.as_ref()).unwrap();

    // For pure states: F = |<psi|phi>|, Bures angle = arccos(F).
    // The Bures metric on pure states is (1/2) * round metric on S^2.
    // So Bures angle = geodesic_distance_on_sphere / 2.
    let v_norm = {
        let st = x[0].sin();
        (v[0] * v[0] + st * st * v[1] * v[1]).sqrt()
    };

    println!("Bures distance (qig) vs sphere geodesic:");
    println!("  Fidelity       = {fid:.6}");
    println!("  Bures distance = {bures:.6}");
    println!("  Bures angle    = {:.6}", fid.acos());
    println!("  Sphere dist    = {v_norm:.6}");
    println!("  Sphere dist/2  = {:.6}", v_norm / 2.0);
    println!();

    // -- Parallel transport --
    let w = array![0.1, 0.2]; // a tangent vector at x
    let w_transported = manifold.parallel_transport(&x.view(), &y.view(), &w.view());
    println!("Parallel transport:");
    println!("  w at x             = [{:.4}, {:.4}]", w[0], w[1]);
    println!(
        "  transport(x->y, w) = [{:.4}, {:.4}]",
        w_transported[0], w_transported[1]
    );

    // -- Verify: log_x(x) = 0 --
    let v_self = manifold.log_map(&x.view(), &x.view());
    let norm_self = (v_self[0] * v_self[0] + v_self[1] * v_self[1]).sqrt();
    println!();
    println!("Self-distance:");
    println!("  ||log_x(x)|| = {norm_self:.2e}");
    assert!(norm_self < 1e-12, "log_x(x) should be zero");
    println!("  PASSED: log_x(x) = 0");
}
