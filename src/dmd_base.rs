use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, Ix1, s};
use ndarray_linalg::{c64, Eigh, LstSq, Scalar, Solver, SVDDC, SVD, UPLO};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::rc::Rc;
use num_complex::{Complex64, ComplexFloat};

pub struct ActivationBitmaskProxy {
    original_modes: Array2<c64>,
    original_eigs: Array1<c64>,
    original_amplitudes: Array1<c64>,
    old_bitmask: Option<Array1<bool>>,
    modes: Array2<c64>,
    eigs: Array1<c64>,
    amplitudes: Array1<c64>,
}

impl ActivationBitmaskProxy {
    pub fn new(dmd_operator: &DMDOperator, amplitudes: ArrayView1<c64>) -> Self {
        let original_modes = dmd_operator.modes.clone();
        let original_eigs = dmd_operator.eigenvalues.clone().into_dimensionality::<Ix1>().unwrap();
        let original_amplitudes = amplitudes.to_owned();

        let mut proxy = ActivationBitmaskProxy {
            original_modes,
            original_eigs,
            original_amplitudes,
            old_bitmask: None,
            modes: Array2::zeros((0, 0)),
            eigs: Array1::zeros(0),
            amplitudes: Array1::zeros(0),
        };
        proxy.change_bitmask(Array1::from_elem(dmd_operator.eigenvalues.len(), true));
        proxy
    }

    pub fn change_bitmask(&mut self, value: Array1<bool>) {
        if let Some(ref old_bitmask) = self.old_bitmask {
            self.original_modes
                .slice_mut(s![.., old_bitmask])
                .assign(&self.modes);
            self.original_eigs
                .slice_mut(s![old_bitmask.clone()])
                .assign(&self.eigs);
            self.original_amplitudes
                .slice_mut(s![old_bitmask.clone()])
                .assign(&self.amplitudes);
        }

        self.modes = self
            .original_modes
            .slice(s![.., value.clone()])
            .to_owned();
        self.eigs = self.original_eigs.slice(s![value.clone()]).to_owned();
        self.amplitudes = self
            .original_amplitudes
            .slice(s![value.clone()])
            .to_owned();

        self.old_bitmask = Some(value);
    }
}

pub struct DMDBase {
    pub svd_rank: usize,
    pub tlsq_rank: usize,
    pub exact: bool,
    pub opt: bool,
    pub rescale_mode: Option<String>,
    pub forward_backward: bool,
    pub sorted_eigs: bool,
    pub tikhonov_regularization: Option<f64>,
    pub atilde: DMDOperator,
    pub original_time: Option<HashMap<String, f64>>,
    pub dmd_time: Option<HashMap<String, f64>>,
    pub b: Option<Array1<c64>>,
    pub snapshots_holder: Option<SnapshotsHolder>,
    pub modes_activation_bitmask_proxy: Option<Rc<ActivationBitmaskProxy>>,
}

impl DMDBase {
    pub fn dynamics(&self) -> Array2<Complex64> {
        let temp = self
            .eigs
            .clone()
            .into_iter()
            .map(|eig| Array1::<Complex64>::from_elem(self.dmd_timesteps().len(), eig))
            .collect::<Array2<Complex64>>();
        let tpow = self
            .dmd_timesteps()
            .mapv(|t| (t - self.original_time["t0"]) / self.original_time["dt"]);

        let tpow = self.translate_eigs_exponent(tpow);

        temp.mapv_inplace(|val| val.powf(tpow)) * self.amplitudes.clone().insert_axis(Axis(1))
    }

    pub fn reconstructed_data(&self) -> Array2<Complex64> {
        self.modes.dot(&self.dynamics())
    }

    pub fn frequency(&self) -> Array1<f64> {
        self.eigs.mapv(|eig| eig.im / (2.0 * std::f64::consts::PI * self.original_time["dt"]))
    }

    pub fn growth_rate(&self) -> Array1<f64> {
        self.eigs.mapv(|eig| eig.re / self.original_time["dt"])
    }
}


pub struct DMDOperator {
    svd_rank: usize,
    exact: bool,
    modes: Option<Array2<Complex64>>,
    eigenvectors: Option<Array2<Complex64>>,
    eigenvalues: Option<Array1<Complex64>>,
}

impl DMDOperator {
    pub fn new(svd_rank: usize, exact: bool) -> Self {
        DMDOperator {
            svd_rank,
            exact,
            modes: None,
            eigenvectors: None,
            eigenvalues: None,
        }
    }

    pub fn fit(&mut self, x: Array2<Complex64>, y: Array2<Complex64>) {
        let svd_rank = if self.svd_rank == 0 {
            std::cmp::min(x.dim().0, x.dim().1)
        } else {
            self.svd_rank
        };

        let svd = x.svd(true, true).unwrap();
        let u = svd.0.unwrap();
        let s = svd.1;
        let vh = svd.2.unwrap();

        let u_truncated = u.slice(s![.., 0..svd_rank]).to_owned();
        let s_truncated = s.slice(s![0..svd_rank]).to_owned();
        let vh_truncated = vh.slice(s![0..svd_rank, ..]).to_owned();

        let s_inv = Array1::from_iter(s_truncated.iter().map(|&x| 1.0 / x)).diag();
        let a_tilde = u_truncated
            .t()
            .dot(&y)
            .dot(&vh_truncated)
            .dot(&s_inv);

        let eig_decomposition = a_tilde.eigh(UPLO::Upper).unwrap();
        let eigenvectors = eig_decomposition.0;
        let eigenvalues = eig_decomposition.1;

        self.eigenvalues = Some(eigenvalues);
        self.eigenvectors = Some(eigenvectors.clone());

        if self.exact {
            self.modes = Some(x.dot(&y).dot(&vh).dot(&s_inv).dot(&eigenvectors));
        } else {
            self.modes = Some(u.dot(&eigenvectors));
        }
    }

    pub fn modes(&self) -> &Option<Array2<Complex64>> {
        &self.modes
    }

    pub fn eigenvectors(&self) -> &Option<Array2<Complex64>> {
        &self.eigenvectors
    }

    pub fn eigenvalues(&self) -> &Option<Array1<Complex64>> {
        &self.eigenvalues
    }
}

