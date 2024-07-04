//! Context-sensitive SIMD operations
//!
//! These aren't exactly numeric-specific, but they can make use of SIMD instructions.

use core::mem::ManuallyDrop;

use super::*;

use generic_array::internals::ArrayConsumer;

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
    ///
    /// NOTE: The default implementation of this will clamp the index values to the length of the array.
    fn select(self, true_values: NumericArray<T, N>, false_values: NumericArray<T, N>) -> NumericArray<T, N>;
}

/// Rearranges one numeric array into another using the supplied indices
pub unsafe trait Permute<T, N: ArrayLength> {
    /// Performs the permutation
    fn permute(self, values: &NumericArray<T, N>) -> NumericArray<T, N>;
}

unsafe impl<T, I, N: ArrayLength> Permute<T, N> for NumericArray<I, N>
where
    T: Clone,
    I: Copy + Into<usize>,
{
    #[inline]
    fn permute(self, values: &NumericArray<T, N>) -> NumericArray<T, N> {
        NumericArray::from_iter(self.iter().map(|index| unsafe {
            let index: usize = (*index).into();

            values.get_unchecked(index.min(N::to_usize())).clone()
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
            let mut values = ManuallyDrop::new(false_values);

            for (mask, (v, t)) in self.iter().zip(values.iter_mut().zip(true_values.iter())) {
                if *mask {
                    unsafe { ptr::copy_nonoverlapping(t, v, 1) };
                }
            }

            ManuallyDrop::into_inner(values)
        }
    }
}
