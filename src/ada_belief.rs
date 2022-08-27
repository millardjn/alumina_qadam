use alumina::core::{errors::ExecError, graph::Node};
use alumina::opt::{calc_change_sqr, GradientStepper};
use indexmap::{indexmap, IndexMap};
use ndarray::{ArcArray, ArrayD, IxDyn, Zip};
use rayon::prelude::*;

#[derive(Clone, Debug)]
struct NodeState {
    momentums: ArrayD<f32>,
    curvatures: ArrayD<f32>,
}

#[derive(Clone, Debug)]
pub struct AdaBelief {
    step_count: usize,

    rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bias_correct: bool,
    states: IndexMap<Node, NodeState>,
}

impl AdaBelief {
    /// Create an optimisation problem assuming that all nodes marked `Parameter` should be optimised.
    pub fn new(rate: f32, beta1: f32, beta2: f32) -> Self {
        AdaBelief {
            step_count: 0,

            rate,
            beta1,
            beta2,
            epsilon: 1e-7,
            bias_correct: true,
            states: indexmap![],
        }
    }

    // /// Learning rate, α
    // ///
    // ///
    // /// θ = θ - α ∇f(θ)
    // ///
    // /// Default: 1e-3
    // pub fn rate(&mut self, rate: f32) -> &mut Self {
    // 	self.rate = rate;
    // 	self
    // }

    // /// Momentum coefficient, β
    // ///
    // /// If not `None`, the following update is used:
    // /// m = β m + ∇f(θ)
    // /// θ = θ - α m
    // ///
    // /// Default: None
    // pub fn momentum<O: Into<Option<f32>>>(&mut self, momentum: O) -> &mut Self {
    // 	self.momentum = momentum.into();
    // 	self
    // }

    /// Learning rate, α
    pub fn rate(&mut self, rate: f32) -> &mut Self {
        self.rate = rate;
        self
    }

    /// Momentum coefficient, β1
    ///
    /// Default: 0.9
    pub fn beta1(&mut self, beta1: f32) -> &mut Self {
        self.beta1 = beta1;
        self
    }

    /// Momentum coefficient, β2
    ///
    /// Default: 0.995
    pub fn beta2(&mut self, beta2: f32) -> &mut Self {
        self.beta2 = beta2;
        self
    }

    /// Fuzz Factor, eps
    ///
    /// Sometimes worth increasing, according to google.
    /// Default: 1e-7
    pub fn epsilon(&mut self, epsilon: f32) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Should bias correction be performed for the momentum vector.
    ///
    /// Note: the curvature vector is always corrected.
    ///
    /// Default: true
    pub fn bias_correct(&mut self, bias_correct: bool) -> &mut Self {
        self.bias_correct = bias_correct;
        self
    }
}

impl GradientStepper for AdaBelief {
    fn step_count(&self) -> usize {
        self.step_count
    }

    fn step(
        &mut self,
        mut parameters_and_grad_values: IndexMap<Node, ArcArray<f32, IxDyn>>,
        calc_change: bool,
    ) -> Result<f32, ExecError> {

        let rate = self.rate;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;
        //let calc_change = options.calc_change;
        let bias_correct = self.bias_correct;
        let momentum_correction = if bias_correct {1.0 / (1.0 - self.beta1.powi(self.step_count as i32 + 1))} else {1.0};
        let curv_correction = 1.0 / (1.0 - self.beta2.powi(self.step_count as i32 + 1));

        let change_sqr: f32 = {
            for param in parameters_and_grad_values.keys() {
                self.states.entry(param.clone()).or_insert_with(|| {
                    let shape = param
                        .shape()
                        .to_data_shape()
                        .expect("Parameters must have fixed shapes");
                    NodeState {
                        momentums: ArrayD::zeros(shape.clone()),
                        curvatures: ArrayD::zeros(shape),
                    }
                });
            }

            // write new parameter array into grad array
            self.states
                .iter_mut()
                .filter_map(|(param, state)| {
                    parameters_and_grad_values
                        .swap_remove(param)
                        .map(|grad_arr| (param, state, grad_arr.to_owned()))
                })
                .par_bridge()
                .map(|(param, state, mut grad_arr)| {
                    let param_arr = param.value().unwrap();

                    Zip::from(&param_arr)
                        .and(&mut state.momentums)
                        .and(&mut state.curvatures)
                        .and(grad_arr.view_mut())
                        .for_each(|param, momentum, curv, grad| {
                            *momentum = *momentum * beta1 + (1.0 - beta1) * *grad;
                            *curv = *curv * beta2
                                + (1.0 - beta1) * (*grad - *momentum) * (*grad - *momentum);
                            *grad = param
                                - rate * (*momentum) * momentum_correction
                                    / ((*curv * curv_correction).sqrt() + epsilon);
                        });
                 
                    let change_sqr = if calc_change {
                        calc_change_sqr(param_arr.view(), grad_arr.view())
                    } else {
                        0.0
                    };
                    param.set_value(grad_arr);
                    change_sqr
                })
                .sum()
        };

        self.step_count += 1;

        //Ok(StepData {
        //	loss,
        //	step: self.step_count,
        Ok(change_sqr.sqrt())
        //})
    }
}
