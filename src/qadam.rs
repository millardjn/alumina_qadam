use alumina::core::{errors::ExecError, graph::Node};
use alumina::opt::{calc_change_sqr, GradientStepper};
use indexmap::{indexmap, IndexMap};
use ndarray::{ArcArray, ArrayD, IxDyn, Zip};
use rayon::prelude::*;

#[derive(Clone, Debug)]
struct NodeState {
    momentums: ArrayD<f32>,
    correlated_variance: ArrayD<f32>,
    uncorrelated_variance: ArrayD<f32>,
    prev_grad: Option<ArrayD<f32>>,

    deferred_update: ArrayD<f32>,
    
    



}

#[derive(Clone, Debug)]
pub struct QAdam {
    step_count: usize,

    rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,

    update_limit: f32,

    bias_correct: bool,
    states: IndexMap<Node, NodeState>,

    preconditioners: IndexMap<Node, f32>,
}

impl QAdam {
    /// Create an optimisation problem assuming that all nodes marked `Parameter` should be optimised.
    pub fn new(rate: f32, beta1: f32, beta2: f32) -> Self {
        QAdam {
            step_count: 0,

            rate,
            beta1,
            beta2,
            epsilon: 1e-8,

            update_limit: 2.0,

            bias_correct: true,
            states: indexmap![],
            preconditioners: indexmap![],
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
        assert!(rate > 0.0);
        self.rate = rate;
        self
    }

    /// Momentum coefficient, β1
    ///
    /// Default: 0.9
    pub fn beta1(&mut self, beta1: f32) -> &mut Self {
        assert!(beta1 >= 0.0);
        assert!(beta1 < 1.0);
        self.beta1 = beta1;
        self
    }

    /// Momentum coefficient, β2
    ///
    /// Default: 0.995
    pub fn beta2(&mut self, beta2: f32) -> &mut Self {
        assert!(beta2 >= 0.0);
        assert!(beta2 < 1.0);
        self.beta2 = beta2;
        self
    }

    /// Fuzz Factor, eps
    ///
    /// Sometimes worth increasing, according to google.
    /// Default: 1e-7
    pub fn epsilon(&mut self, epsilon: f32) -> &mut Self {
        assert!(epsilon > 0.0);
        self.epsilon = epsilon;
        self
    }

    /// Update Limit
    ///
    /// The maximum change in a parameter value allowed in a single update is limited to update_limit*learning_rate*node_preconditioner.
    /// If a larger change would occur, the change is stored and dispensed in later steps.
    /// 
    /// Default: 2.0
    pub fn update_limit(&mut self, update_limit: f32) -> &mut Self {
        assert!(update_limit > 0.0);
        self.update_limit = update_limit;
        self
    }

    // stores the parameter scale of the Node
    // updates are multiplied by this scale
    // the change_limit is multiplied by this scale
    pub fn preconditioner(&mut self, preconditioners: IndexMap<Node, f32>) -> &mut Self {
        self.preconditioners.extend(preconditioners);
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

impl GradientStepper for QAdam {
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
        let update_limit = self.update_limit;
        let bias_correct = self.bias_correct;
        let preconditioners = &self.preconditioners;
        let momentum_correction = if bias_correct {1.0 / (1.0 - self.beta1.powi(self.step_count as i32 + 1))} else {1.0};
        let curv_correction = 1.0 / (1.0 - self.beta2.powi(self.step_count as i32 + 1));
        let crossover = self.beta2.powi(self.step_count as i32 + 1);

        let change_sqr: f32 = {
            for param in parameters_and_grad_values.keys() {
                self.states.entry(param.clone()).or_insert_with(|| {
                    let shape = param
                        .shape()
                        .to_data_shape()
                        .expect("Parameters must have fixed shapes");
                    NodeState {
                        momentums: ArrayD::zeros(shape.clone()),
                        correlated_variance: ArrayD::zeros(shape.clone()),//ArrayD::from_elem(shape.clone(), std::f32::EPSILON),
                        uncorrelated_variance: ArrayD::zeros(shape.clone()),//ArrayD::from_elem(shape, std::f32::EPSILON),

                        deferred_update: ArrayD::zeros(shape.clone()),
                        prev_grad: None,


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
                    if let Some(prev_grad) = state.prev_grad.take() {
                        let param_arr = param.value().unwrap();
                        let preconditioner = *preconditioners.get(param).unwrap_or(&1.0);
                        
                        Zip::from(&mut state.momentums)
                            .and(&mut state.correlated_variance)
                            .and(&mut state.uncorrelated_variance)
                            .and(&prev_grad)
                            .and(grad_arr.view_mut())
                            .for_each(
                                |momentum, corr_var, uncorr_var, prev_grad, grad| {
                                   
                                    let avg = (*prev_grad+*grad)*0.5;
                                    // standard deviation of the average around the true value
                                    let std_dev = (*grad - *prev_grad)* 0.5;
                                    

                                    // let avg = (*prev_grad+*grad)*0.5;
                                    // let avg_var = *mini_var*0.5;
                                    // let estimate_var = avg_var**true_var/(avg_var + *true_var);
                                    // let estimate = avg*estimate_var/avg_var;
                                    // let variance_correction = (avg_var + *true_var)/ *true_var;
                                    // *true_var =
                                    //     *true_var * beta2 + (1.0 - beta2) * estimate * estimate * variance_correction;

                                    //     *true_var = *true_var * beta2 + (1.0 - beta2) * avg * avg * *true_var / (avg_var + *true_var) ;




   
                                    // let safe_corr_var = *corr_var * (1.0-crossover)* curv_correction + ((*prev_grad**prev_grad + *grad**grad)*0.25)*crossover + epsilon*epsilon;
                                    // *corr_var =
                                    //     *corr_var * beta2 + (1.0 - beta2) * avg * avg * safe_corr_var / (*uncorr_var* curv_correction + safe_corr_var) ;


                                    // having this after the corr_var update helps slightly when sudden large values are encountered
                                    // *uncorr_var = *uncorr_var * beta2 + (1.0 - beta2) * std_dev * std_dev ;




                                    *uncorr_var = *uncorr_var * beta2 + (1.0 - beta2) * std_dev * std_dev ;

                                    let safe_corr_var =  (*corr_var * (1.0-crossover) + *uncorr_var*crossover)* curv_correction + epsilon*epsilon;
                                    *corr_var =
                                        *corr_var * beta2 + (1.0 - beta2) * avg * avg * safe_corr_var / (*uncorr_var + safe_corr_var) ;





                                    *momentum = *momentum * beta1 + (1.0 - beta1) * avg;


                                    //*grad = param - rate * momentum_correction * (*momentum) / safe_corr_var.sqrt();
                                    *grad = - rate*preconditioner * momentum_correction * (*momentum) / safe_corr_var.sqrt();


                                    
                                },


                            );

                            
                            Zip::from(&param_arr)
                            .and(&mut state.deferred_update)
                            .and(grad_arr.view_mut())
                            .for_each(
                                |param, deferred_update, grad| {
                                    let limit = update_limit*rate*preconditioner;

                                    // Dispense from deferred update, up to limit
                                    let mut update = deferred_update.clamp(-limit, limit);
                                    *deferred_update -= update;

                                    // Add new update, and then defer any excess
                                    update += *grad;
                                    let limited_update = update.clamp(-limit, limit);
                                    *deferred_update += update - limited_update;

                                    // test
                                    *grad = *param + limited_update;
                                });
                        
                        let change_sqr = if calc_change {
                            calc_change_sqr(param_arr.view(), grad_arr.view())
                        } else {
                            0.0
                        };
                        param.set_value(grad_arr);
                        change_sqr
                    } else {
                        state.prev_grad = Some(grad_arr);
                        0.0
                    }
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

// //#[inline(never)]
// fn estimate_true_grad(true_var: f32, mini_var: f32, prev_grad: f32, grad: f32) -> f32 {
//     if true_var*mini_var == 0.0 {

//         let true_grad_estimate = (prev_grad + grad)*0.1;
            
//         true_grad_estimate
//     } else {
//         let estimate_var = true_var * mini_var 
//         / (2.0 * mini_var * true_var + mini_var * mini_var) * mini_var;
    
//         let true_grad_estimate = (0.0 / true_var
//             + prev_grad / mini_var
//             + grad / mini_var)
//             * estimate_var;

//         true_grad_estimate
//     }

    
// }