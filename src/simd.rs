//! Context-sensitive SIMD operations
//!
//! These aren't exactly numeric-specific, but they can make use of SIMD instructions.

use super::*;

use generic_array::{ArrayBuilder, ArrayConsumer};

/// Selects elements from one array or another using `self` as a mask.
pub unsafe trait Select<T, N: ArrayLength<T>> {
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
pub unsafe trait Permute<T, N: ArrayLength<T>> {
    /// Performs the permutation
    fn permute(self, values: &NumericArray<T, N>) -> NumericArray<T, N>;
}

unsafe impl<T, N: ArrayLength<T>> Permute<T, N> for NumericArray<usize, N>
where
    N: ArrayLength<usize>,
    T: Clone,
{
    fn permute(self, values: &NumericArray<T, N>) -> NumericArray<T, N> {
        unsafe {
            let mut indices = ArrayConsumer::new(self.0);

            let (index_iter, index_position) = indices.iter_position();

            let mut destination = ArrayBuilder::new();

            {
                let (destination_iter, destination_position) = destination.iter_position();

                destination_iter.zip(index_iter).for_each(|(dst, index)| {
                    let value = values.get_unchecked(*index).clone();

                    *index_position += 1;

                    ptr::write(dst, value);

                    *destination_position += 1;
                });
            }

            NumericArray::new(destination.into_inner())
        }
    }
}

unsafe impl<T, N: ArrayLength<T>> Select<T, N> for NumericArray<bool, N>
where
    N: ArrayLength<bool>,
{
    fn select(self, true_values: NumericArray<T, N>, false_values: NumericArray<T, N>) -> NumericArray<T, N> {
        unsafe {
            let mut mask = ArrayConsumer::new(self.0);
            let mut true_values = ArrayConsumer::new(true_values.0);
            let mut false_values = ArrayConsumer::new(false_values.0);

            let (mask_iter, mask_position) = mask.iter_position();
            let (true_values_iter, true_values_position) = true_values.iter_position();
            let (false_values_iter, false_values_position) = false_values.iter_position();

            let mut destination = ArrayBuilder::new();

            {
                let (destination_iter, destination_position) = destination.iter_position();

                destination_iter
                    .zip(mask_iter.zip(true_values_iter.zip(false_values_iter)))
                    .for_each(|(dst, (m, (t, f)))| {
                        let t = ptr::read(t);
                        let f = ptr::read(f);
                        let m = ptr::read(m);

                        *mask_position += 1;
                        *true_values_position += 1;
                        *false_values_position += 1;

                        ptr::write(dst, if m { t } else { f });

                        *destination_position += 1;
                    });
            }

            NumericArray::new(destination.into_inner())
        }
    }
}
