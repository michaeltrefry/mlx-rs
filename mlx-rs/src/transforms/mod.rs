//! Function transforms
//!
//! This mod provides functions for automatic differentiation and other
//! transformations on functions.
//!
//! **WARN**: Because function transforms including compilation works on
//! the computation graph, the user must ensure that all `Array`s are passed
//! as inputs to the function/closure. Closures with captured `Array`s may
//! not work as expected and may lead to undefined behavior.
//!
//! # Automatic Differentiation
//!
//! Automatic differentiation in MLX works on functions rather than on implicit
//! graphs.
//!
//! **NOTE**: If you are coming to MLX from PyTorch, you no longer need
//! functions like backward, zero_grad, and detach, or properties like
//! requires_grad.
//!
//! You can use the [`grad()`] and [`value_and_grad()`] function to compute
//! gradients of more complex functions. These functions compute the gradient
//! with respect to the first argument, in order to manually specify the the
//! argument to compute the gradient with respect to, use
//! [`grad_with_argnums()`] or [`value_and_grad_with_argnums()`].
//!
//! TODO: update the example once https://github.com/oxideai/mlx-rs/pull/218 is merged
//!
//! ```rust,ignore
//! use mlx_rs::{Array, error::Result, transforms::grad};
//!
//! fn f(x: &Array) -> Result<Array> {
//!     x.square()
//! }
//!
//! fn calculate_grad(func: impl Fn(&Array) -> Result<Array>, arg: &Array) -> Result<Array> {
//!     grad(&func, &[0])(arg)
//! }
//!
//! let x = Array::from(1.5);
//!
//! let dfdx = calculate_grad(f, &x).unwrap();
//! assert_eq!(dfdx.item::<f32>(), 2.0 * 1.5);
//!
//! let dfdx2 = calculate_grad(|args| calculate_grad(f, args), &x).unwrap();
//! assert_eq!(dfdx2.item::<f32>(), 2.0);
//! ```

use mlx_sys::mlx_closure_value_and_grad;

use crate::{
    error::{get_and_clear_closure_error, Result},
    module::ModuleParamRef,
    utils::{guard::Guarded, Closure, VectorArray, SUCCESS},
    Array,
};

pub mod compile;
mod grad;
mod keyed_value_and_grad;
mod value_and_grad;

pub use grad::*;
pub use keyed_value_and_grad::*;
pub use value_and_grad::*;

/// Evaluate an iterator of [`Array`]s.
pub fn eval<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    let vec = VectorArray::try_from_iter(outputs.into_iter())?;
    <() as Guarded>::try_from_op(|_| unsafe { mlx_sys::mlx_eval(vec.as_ptr()) })
}

/// Evaluate a module's parameters.
///
/// This is a convenience function that flattens the parameters and evaluates them.
pub fn eval_params(params: ModuleParamRef<'_>) -> Result<()> {
    eval(params.flatten().values().copied())
}

/// Asynchronously evaluate an iterator of [`Array`]s.
///
/// Please note that this is not a rust async function.
pub fn async_eval<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    let vec = VectorArray::try_from_iter(outputs.into_iter())?;
    <() as Guarded>::try_from_op(|_| unsafe { mlx_sys::mlx_async_eval(vec.as_ptr()) })
}

/// Asynchronously evaluate a module's parameters.
///
/// This is a convenience function that flattens the parameters and evaluates them.
pub fn async_eval_params(params: ModuleParamRef<'_>) -> Result<()> {
    async_eval(params.flatten().values().copied())
}

/// Gradient (activation) checkpointing.
///
/// Returns a function equivalent to `f` in the forward pass, but whose **backward** pass recomputes
/// `f` from its inputs instead of retaining `f`'s intermediate activations on the autograd tape.
/// Apply it per-segment (e.g. once per transformer block) inside a function being differentiated by
/// [`value_and_grad`]/[`keyed_value_and_grad`] to bound the reverse-mode working set to a single
/// segment's activations, at the cost of one recomputation of `f` during the backward pass.
///
/// Gradients flow to the arrays passed as **inputs** to the returned function. Per the module-level
/// note, pass every array you need a gradient for as an input; arrays merely *captured* by `f` are
/// baked into its graph as constants and receive no gradient (which is exactly what you want for
/// frozen base weights — pass only the trainable parameters + activations as inputs).
///
/// Wraps `mlx::core::checkpoint`.
pub fn checkpoint<'a, F>(f: F) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let inner = Closure::new_fallible(f);
    move |inputs: &[Array]| -> Result<Vec<Array>> {
        // Wrap the inner closure as a checkpointed closure, then apply it to `inputs`. Building the
        // wrapper per call is cheap (it only re-references `inner`, no compute) and keeps `inner`
        // borrowed for the lifetime of the returned closure.
        let ckpt =
            Closure::try_from_op(|res| unsafe { mlx_sys::mlx_checkpoint(res, inner.as_ptr()) })?;
        let c_inputs = VectorArray::try_from_iter(inputs.iter())?;
        let outputs = VectorArray::try_from_op(|res| unsafe {
            mlx_sys::mlx_closure_apply(res, ckpt.as_ptr(), c_inputs.as_ptr())
        })
        .map_err(|e| get_and_clear_closure_error().unwrap_or(e))?;
        let values: Vec<Array> = outputs.try_into_values()?;
        Ok(values)
    }
}

#[inline]
fn jvp_inner(
    closure: Closure<'_>,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    let c_primals = VectorArray::try_from_iter(primals.iter())?;
    let c_tangents = VectorArray::try_from_iter(tangents.iter())?;

    <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_jvp(
            res_0,
            res_1,
            closure.as_ptr(),
            c_primals.as_ptr(),
            c_tangents.as_ptr(),
        )
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })
}

/// Compute the Jacobian-vector product.
///
/// This computes the product of the Jacobian of a function `f` evaluated at
/// `primals` with the `tangents`.
///
/// # Params:
///
/// - `f`: function which takes an array of `Array` and returns an array of
///   `Array`
/// - `primals`: array of `Array` at which to evaluate the Jacobian
/// - `tangents`: array of `Array` which are the "vector" in the Jacobian-vector
///   product.  The `tangents` should be the same in number, shape and type as
///   the inputs of `f`, e.g. the `primals`
///
/// # Returns:
///
/// Array of the Jacobian-vector products which is the same in number, shape and
/// type of the outputs of `f`
pub fn jvp<'a, F>(f: F, primals: &[Array], tangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    jvp_inner(closure, primals, tangents)
}

/// Similar to [`jvp`] but handles closures that can return an error.
pub fn fallible_jvp<'a, F>(
    f: F,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new_fallible(f);
    jvp_inner(closure, primals, tangents)
}

#[inline]
fn vjp_inner(
    closure: Closure<'_>,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    let c_primals = VectorArray::try_from_iter(primals.iter())?;
    let c_cotangents = VectorArray::try_from_iter(cotangents.iter())?;

    <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_vjp(
            res_0,
            res_1,
            closure.as_ptr(),
            c_primals.as_ptr(),
            c_cotangents.as_ptr(),
        )
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })
}

/// Compute the vector-Jacobian product.
///
/// Computes the product of the `cotangents` with the Jacobian of a function `f` evaluated at
/// `primals`.
///
/// # Params:
///
/// - f: function which takes an array of `Array` and returns an array of `Array`
/// - primals: array of `Array` at which to evaluate the Jacobian
/// - cotangents: array of `Array` which are the "vector" in the vector-Jacobian product. The
///   `cotangents` should be the same in number, shape and type as the outputs of `f`
///
/// # Returns:
///
/// array of the vector-Jacobian products which is the same in number, shape and type of the outputs
/// of `f`
pub fn vjp<'a, F>(f: F, primals: &[Array], cotangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    vjp_inner(closure, primals, cotangents)
}

/// Similar to [`vjp`] but handles closures that can return an error.
pub fn fallible_vjp<'a, F>(
    f: F,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new_fallible(f);
    vjp_inner(closure, primals, cotangents)
}

pub(crate) struct ClosureValueAndGrad {
    pub(crate) c_closure_value_and_grad: mlx_closure_value_and_grad,
}

impl ClosureValueAndGrad {
    pub fn as_ptr(&self) -> mlx_closure_value_and_grad {
        self.c_closure_value_and_grad
    }
}

impl Drop for ClosureValueAndGrad {
    fn drop(&mut self) {
        let status =
            unsafe { mlx_sys::mlx_closure_value_and_grad_free(self.c_closure_value_and_grad) };
        debug_assert_eq!(status, SUCCESS);
    }
}

fn value_and_gradient(
    value_and_grad: mlx_closure_value_and_grad,
    arrays: impl Iterator<Item = impl AsRef<Array>>,
) -> Result<(Vec<Array>, Vec<Array>)> {
    let input_vector = VectorArray::try_from_iter(arrays)?;

    <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_closure_value_and_grad_apply(
            res_0,
            res_1,
            value_and_grad,
            input_vector.as_ptr(),
        )
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })
}

#[cfg(test)]
mod tests {

    use crate::{
        array,
        transforms::{jvp, vjp},
        Array,
    };

    use super::*;

    // The unit tests below are adapted from the mlx c++ codebase

    // sc-4874: gradient (activation) checkpointing — used to bound the reverse-mode working set in
    // the mlx-gen LoRA trainers. The decisive property: gradients w.r.t. the *explicit inputs* of a
    // checkpointed segment must match the non-checkpointed graph exactly (to fp tolerance), even
    // when the segment also reads a *captured constant* (the frozen base weight) that gets no grad.
    #[test]
    fn test_checkpoint_grad_matches_non_checkpointed() {
        use crate::ops::tanh;
        use crate::random;

        let key = random::key(0).unwrap();
        let wc = random::normal::<f32>(&[8, 8], None, None, Some(&key)).unwrap(); // captured constant
        let x = random::normal::<f32>(&[4, 8], None, None, Some(&key)).unwrap(); // captured constant

        // layer(h, p) = tanh(h @ wc + p), with `wc` and `x` captured; `p` is the trainable input.
        let plain = |args: &[Array]| -> Result<Vec<Array>> {
            let mut h = x.clone();
            for p in args {
                h = tanh(&(h.matmul(&wc)?.add(p)?))?;
            }
            Ok(vec![h.sum(None)?])
        };
        let ckpt = |args: &[Array]| -> Result<Vec<Array>> {
            let mut h = x.clone();
            for p in args {
                let mut seg = checkpoint(|inp: &[Array]| -> Result<Vec<Array>> {
                    Ok(vec![tanh(&(inp[0].matmul(&wc)?.add(&inp[1])?))?])
                });
                h = seg(&[h.clone(), p.clone()])?.into_iter().next().unwrap();
            }
            Ok(vec![h.sum(None)?])
        };

        let p0 = random::normal::<f32>(&[8], None, None, Some(&key)).unwrap();
        let p1 = random::normal::<f32>(&[8], None, None, Some(&key)).unwrap();
        let args = &[p0, p1];
        let argnums = &[0, 1];

        let (v_plain, g_plain) = value_and_grad_with_argnums(plain, argnums)(args).unwrap();
        let (v_ckpt, g_ckpt) = value_and_grad_with_argnums(ckpt, argnums)(args).unwrap();

        // Forward value matches.
        assert!(
            (v_plain[0].item::<f32>() - v_ckpt[0].item::<f32>()).abs() < 1e-4,
            "forward value differs: {} vs {}",
            v_plain[0].item::<f32>(),
            v_ckpt[0].item::<f32>()
        );
        // Gradients w.r.t. each trainable input match (the property the trainer relies on).
        for (gp, gc) in g_plain.iter().zip(g_ckpt.iter()) {
            let d = gp.subtract(gc).unwrap().abs().unwrap().max(None).unwrap();
            assert!(
                d.item::<f32>() < 1e-4,
                "checkpoint grad diverged: max|Δ| = {}",
                d.item::<f32>()
            );
        }
    }

    // Per-segment checkpointing must reduce the backward-pass peak memory on a deep chain (the whole
    // point). Build a chain deep/wide enough that retained activations dominate, and compare peaks.
    #[test]
    fn test_checkpoint_reduces_peak_memory() {
        use crate::memory::{clear_cache, get_peak_memory, reset_peak_memory};
        use crate::ops::tanh;
        use crate::random;
        use crate::transforms::eval;

        // Deep stack with large activations and small weights, so the RETAINED forward activations
        // dominate the peak — the regime checkpointing targets (the z-image 1024 case: ~105 GB of
        // activations). At shallow depth / small activations, weight-grad + matmul temporaries
        // dominate instead and checkpoint's recompute overhead loses; that is expected and is why
        // this test is sized so activations dominate.
        let key = random::key(1).unwrap();
        let (n, d, hidden) = (256i32, 1024i32, 4096i32); // wide FFN: the [n,hidden] intermediate dominates
        let depth = 96usize;
        // Captured frozen "base weights" — the up/down projections of each segment's FFN.
        let w_up = random::normal::<f32>(&[d, hidden], None, None, Some(&key)).unwrap();
        let w_down = random::normal::<f32>(&[hidden, d], None, None, Some(&key)).unwrap();
        let x = random::normal::<f32>(&[n, d], None, None, Some(&key)).unwrap();
        let p = random::normal::<f32>(&[d], None, None, Some(&key)).unwrap();

        // A representative block: up-project to a wide hidden, gelu, down-project, residual + bias.
        // The wide [n,hidden] activation is exactly the kind of intermediate a real DiT block retains
        // for the backward — and what per-segment checkpointing recomputes instead of storing.
        let run = |use_ckpt: bool| -> usize {
            let f = |args: &[Array]| -> Result<Vec<Array>> {
                let mut h = x.clone();
                for _ in 0..depth {
                    if use_ckpt {
                        let mut seg = checkpoint(|inp: &[Array]| -> Result<Vec<Array>> {
                            let up = tanh(&inp[0].matmul(&w_up)?)?;
                            let down = up.matmul(&w_down)?;
                            Ok(vec![tanh(&down.add(&inp[1])?)?])
                        });
                        h = seg(&[h.clone(), args[0].clone()])?
                            .into_iter()
                            .next()
                            .unwrap();
                    } else {
                        let up = tanh(&h.matmul(&w_up)?)?;
                        let down = up.matmul(&w_down)?;
                        h = tanh(&down.add(&args[0])?)?;
                    }
                }
                Ok(vec![h.sum(None)?])
            };
            clear_cache();
            reset_peak_memory();
            let (_v, g) = value_and_grad_with_argnums(f, &[0])(&[p.clone()]).unwrap();
            eval(g.iter()).unwrap();
            get_peak_memory()
        };

        let peak_plain = run(false);
        let peak_ckpt = run(true);
        eprintln!(
            "[checkpoint] peak plain {:.1} MB  ckpt {:.1} MB  ({:.0}% reduction)",
            peak_plain as f64 / 1e6,
            peak_ckpt as f64 / 1e6,
            100.0 * (1.0 - peak_ckpt as f64 / peak_plain as f64)
        );
        assert!(
            peak_ckpt < peak_plain,
            "checkpointing must reduce backward peak memory: plain {peak_plain} vs ckpt {peak_ckpt}"
        );
    }

    #[test]
    fn test_jvp() {
        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] + &inputs[1]] };
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let (out, dout) = jvp(f, &[x, y], &[array!(1.0f32), array!(3.0f32)]).unwrap();
        assert_eq!(out[0].item::<f32>(), 2.0f32);
        assert_eq!(dout[0].item::<f32>(), 4.0f32);
    }

    #[test]
    fn test_jvp_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>> {
            inputs[0].add(&inputs[1]).map(|res| vec![res])
        };

        // Success case
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let (out, dout) = fallible_jvp(f, &[x, y], &[array!(1.0f32), array!(3.0f32)]).unwrap();
        assert_eq!(out[0].item::<f32>(), 2.0f32);
        assert_eq!(dout[0].item::<f32>(), 4.0f32);

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let result = fallible_jvp(f, &[a, b], &[array!(1.0f32), array!(3.0f32)]);
        assert!(result.is_err());

        // Check that the error is not just "mlx_closure returned a non-zero value"
        let err = result.unwrap_err();
        assert!(!err.what().contains("non-zero value"))
    }

    #[test]
    fn test_vjp() {
        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] + &inputs[1]] };
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let primals = vec![x, y];
        let cotangents = vec![array!(1.0f32)];
        let (out, dout) = vjp(f, &primals, &cotangents).unwrap();
        assert_eq!(out[0].item::<f32>(), 2.0f32);
        assert_eq!(dout[0].item::<f32>(), 1.0f32);
    }

    #[test]
    fn test_vjp_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>> {
            inputs[0].add(&inputs[1]).map(|res| vec![res])
        };

        // Success case
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let primals = vec![x, y];
        let cotangents = vec![array!(1.0f32)];
        let (out, dout) = fallible_vjp(f, &primals, &cotangents).unwrap();
        assert_eq!(out[0].item::<f32>(), 2.0f32);
        assert_eq!(dout[0].item::<f32>(), 1.0f32);

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let result = fallible_vjp(f, &[a, b], &[array!(1.0f32)]);
        assert!(result.is_err());

        // Check that the error is not just "mlx_closure returned a non-zero value"
        let err = result.unwrap_err();
        assert!(!err.what().contains("non-zero value"))
    }
}
