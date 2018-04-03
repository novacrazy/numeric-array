//! Context-sensitive SIMD operations
//!
//! These aren't exactly numeric-specific, but they can make use of SIMD instructions.

use super::*;

/// Selects elements from one array or another using `self` as a mask.
pub unsafe trait Select<T, N: ArrayLength<T>> {
    /// Selects elements from one array or another using `self` as a mask.
    ///
    /// Example:
    ///
    /// ```ignore
    /// use simd::SimdSelect;
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

unsafe impl<T, N: ArrayLength<T>> Select<T, N> for NumericArray<bool, N>
where
    N: ArrayLength<bool>,
{
    fn select(self, true_values: NumericArray<T, N>, false_values: NumericArray<T, N>) -> NumericArray<T, N> {
        let mut mask = ArrayConsumer::new(self.0);
        let mut true_values = ArrayConsumer::new(true_values.0);
        let mut false_values = ArrayConsumer::new(false_values.0);

        let mut destination = ArrayBuilder::new();

        for (dst, (m, (t, f))) in destination.array.iter_mut().zip(mask.array.iter().zip(true_values.array.iter().zip(false_values.array.iter()))) {
            unsafe {
                let t = ptr::read(t);
                let f = ptr::read(f);
                let m = ptr::read(m);

                mask.position += 1;
                true_values.position += 1;
                false_values.position += 1;

                ptr::write(dst, if m { t } else { f });

                destination.position += 1;
            }
        }

        NumericArray::new(destination.into_inner())
    }
}
