//! Context-sensitive SIMD operations
//!
//! These aren't exactly numeric-specific, but they can make use of SIMD instructions.

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
        unsafe {
            let mut destination = ArrayBuilder::new();

            destination.extend(self.0.iter().map(|index| values.get_unchecked(*index).clone()));

            NumericArray::new(destination.assume_init())
        }
    }
}

unsafe impl<T, N: ArrayLength> Select<T, N> for NumericArray<bool, N> {
    #[inline]
    fn select(self, true_values: NumericArray<T, N>, false_values: NumericArray<T, N>) -> NumericArray<T, N> {
        unsafe {
            let mut true_values = ArrayConsumer::new(true_values.0);
            let mut false_values = ArrayConsumer::new(false_values.0);

            let (true_values_iter, true_values_position) = true_values.iter_position();
            let (false_values_iter, false_values_position) = false_values.iter_position();

            let mut destination = ArrayBuilder::new();

            destination.extend(self.0.iter().zip(true_values_iter.zip(false_values_iter)).map(|(m, (t, f))| {
                let t = ptr::read(t);
                let f = ptr::read(f);

                *true_values_position += 1;
                *false_values_position = *true_values_position;

                match *m {
                    true => t,
                    false => f,
                }
            }));

            NumericArray::new(destination.assume_init())
        }
    }
}
