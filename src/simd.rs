//! Context-sensitive SIMD operations
//!
//! These aren't exactly numeric-specific, but they can make use of SIMD instructions.

use core::mem::ManuallyDrop;

use super::*;

use generic_array::internals::{ArrayBuilder, ArrayConsumer};

/// Selects elements from one array or another using `self` as a mask.
pub unsafe trait Select<T, N: ArrayLength> {
    /// Selects elements from one array or another using `self` as a mask.
    ///
    /// Example:
    ///
    /// ```ignore
    /// use numeric_array::simd::Select;
    ///
    /// let mask = narr![bool; true, false, false, true];
    ///
    /// let a = narr![i32; 1, 2, 3, 4];
    /// let b = narr![i32; 5, 6, 7, 8];
    ///
    /// // Compiles to vblendvps on my machine
    /// let selected = mask.select(a, b);
    ///
    /// assert_eq!(selected, narr![i32; 1, 6, 7, 4]);
    /// ```
    fn select(self, true_values: NumericArray<T, N>, false_values: NumericArray<T, N>) -> NumericArray<T, N>;
}

/// Rearranges one numeric array into another using the supplied indices
pub unsafe trait Permute<T, N: ArrayLength> {
    /// Performs the permutation
    fn permute(self, values: &NumericArray<T, N>) -> NumericArray<T, N>;
}

unsafe impl<T, N: ArrayLength> Permute<T, N> for NumericArray<usize, N>
where
    T: Clone,
{
    #[inline]
    fn permute(self, values: &NumericArray<T, N>) -> NumericArray<T, N> {
        NumericArray::from_iter(self.iter().map(|index| unsafe {
            values.get_unchecked(*index).clone() //
        }))
    }
}

unsafe impl<T, N: ArrayLength> Select<T, N> for NumericArray<bool, N> {
    #[inline]
    fn select(self, true_values: NumericArray<T, N>, false_values: NumericArray<T, N>) -> NumericArray<T, N> {
        if core::mem::needs_drop::<T>() {
            unsafe {
                let mut true_values = ArrayConsumer::new(true_values.0);
                let mut false_values = ArrayConsumer::new(false_values.0);

                let (true_values_iter, true_values_position) = true_values.iter_position();
                let (false_values_iter, false_values_position) = false_values.iter_position();

                NumericArray::from_iter(self.0.iter().zip(true_values_iter.zip(false_values_iter)).map(|(m, (t, f))| {
                    let t = ptr::read(t);
                    let f = ptr::read(f);

                    *true_values_position += 1;
                    *false_values_position = *true_values_position;

                    match *m {
                        true => t,
                        false => f,
                    }
                }))
            }
        } else {
            let true_values = ManuallyDrop::new(true_values);
            let false_values = ManuallyDrop::new(false_values);

            NumericArray::from_iter(self.iter().zip(true_values.iter().zip(false_values.iter())).map(|(mask, (t, f))| unsafe {
                match *mask {
                    true => ptr::read(t),
                    false => ptr::read(f),
                }
            }))
        }
    }
}
