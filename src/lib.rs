//! # `numeric-array`
//!
//! [![crates.io](https://img.shields.io/crates/v/numeric-array.svg)](https://crates.io/crates/numeric-array)
//! [![Documentation](https://docs.rs/numeric-array/badge.svg)](https://docs.rs/numeric-array)
//! [![MIT/Apache-2 licensed](https://img.shields.io/crates/l/numeric-array.svg)](./LICENSE-Apache)
//!
//! `numeric-array` is a wrapper around
//! [`generic-array`](https://github.com/fizyk20/generic-array)
//! that adds efficient numeric trait implementations, designed
//! to encourage LLVM to autovectorize expressions into SIMD
//! instructions and perform compile-time evaluation.
//!
//! All stable `core::ops` traits are implemented for `NumericArray` itself,
//! plus the thin `NumericConstant` type, which is required to
//! differeniate constant values from `NumericArray` itself.
//!
//! Additionally, most of `num_traits` are implemented,
//! including `Num` itself. So you can even use a whole array as a generic number.
//!
//! Example:
//!
//! ```rust
//! use num_traits::Float;
//! use numeric_array::{NumericArray, narr};
//!
//! # #[cfg(feature = "std")]
//! fn main() {
//!     let a = narr![1.0, 2.0, 3.0, 4.0];
//!     let b = narr![5.0, 6.0, 7.0, 8.0];
//!     let c = narr![9.0, 1.0, 2.0, 3.0];
//!
//!     // Compiles to a single vfmadd213ps instruction on my machine
//!     let d = a.mul_add(b, c);
//!
//!     assert_eq!(d, narr![14.0, 13.0, 23.0, 35.0]);
//! }
//!
//! # #[cfg(not(feature = "std"))] fn main() {}
//! ```
//!
//! When used with `RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"`,
//! then Rust and LLVM are smart enough to autovectorize almost all operations
//! into SIMD instructions, or even just evaluate them at compile time.
//! The above example is actually evaluated at compile time, so if you
//! were to view the assembly it would show the result only.
//!
//! This is ideal for situations where simple component-wise operations are required for arrays.
//!

#![deny(missing_docs)]
#![no_std]

extern crate num_traits;

pub extern crate generic_array;

pub use generic_array::{typenum, ArrayLength};

#[cfg(feature = "serde1")]
extern crate serde;

use core::{cmp, ptr, slice};

use core::borrow::{Borrow, BorrowMut};
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::ops::{Range, RangeFrom, RangeFull, RangeTo};

use core::iter::FromIterator;

use core::fmt::{Debug, Formatter, Result as FmtResult};

use generic_array::functional::*;
use generic_array::sequence::*;
use generic_array::{GenericArray, GenericArrayIter};

#[cfg(feature = "serde1")]
mod impl_serde;

pub mod geometry;
pub mod impls;
pub mod simd;

/// A numeric wrapper for a `GenericArray`, allowing for easy numerical operations
/// on the whole sequence.
///
/// This has the added bonus of allowing SIMD optimizations for almost all operations
/// when compiled with `RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"`
///
/// For example, adding together four-element `NumericArray`'s will result
/// in a single SIMD instruction for all elements at once.
#[repr(transparent)]
pub struct NumericArray<T, N: ArrayLength>(GenericArray<T, N>);

/// Sugar for `NumericArray::new(arr![...])`
///
/// ```ignore
/// #[macro_use]
/// extern crate generic_array;
/// ```
///
/// is required to use this, as it still uses the `arr!` macro internally.
#[macro_export]
macro_rules! narr {
    ($($t:tt)*) => {
        $crate::NumericArray::new($crate::generic_array::arr!($($t)*))
    }
}

unsafe impl<T, N: ArrayLength> GenericSequence<T> for NumericArray<T, N> {
    type Length = N;
    type Sequence = Self;

    #[inline(always)]
    fn generate<F>(f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        NumericArray(GenericArray::generate(f))
    }
}

/// This is required to allow `NumericArray` to be operated on by both other `NumericArray`
/// instances and constants, with generic types,
/// because some type `U` supercedes `NumericArray<U, N>`
///
/// As a result, constants must be wrapped in this totally
/// transparent wrapper type to differentiate the types to Rust.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct NumericConstant<T>(pub T);

/// Creates a new `NumericConstant` from the given expression.
#[macro_export]
macro_rules! nconstant {
    ($value:expr) => {
        $crate::NumericConstant($value)
    };
}

impl<T> Deref for NumericConstant<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for NumericConstant<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Debug, N: ArrayLength> Debug for NumericArray<T, N> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.debug_tuple("NumericArray").field(&self.0).finish()
    }
}

impl<X, T, N: ArrayLength> From<X> for NumericArray<T, N>
where
    X: Into<GenericArray<T, N>>,
{
    fn from(x: X) -> NumericArray<T, N> {
        NumericArray::new(x.into())
    }
}

impl<T: Clone, N: ArrayLength> Clone for NumericArray<T, N> {
    fn clone(&self) -> NumericArray<T, N> {
        NumericArray(self.0.clone())
    }
}

impl<T: Copy, N: ArrayLength> Copy for NumericArray<T, N> where N::ArrayType<T>: Copy {}

impl<T, N: ArrayLength> Deref for NumericArray<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, N: ArrayLength> DerefMut for NumericArray<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T, U, N: ArrayLength> PartialEq<NumericArray<U, N>> for NumericArray<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, rhs: &NumericArray<U, N>) -> bool {
        **self == **rhs
    }
}

impl<T, U, N: ArrayLength> PartialEq<GenericArray<U, N>> for NumericArray<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, rhs: &GenericArray<U, N>) -> bool {
        **self == **rhs
    }
}

impl<T, N: ArrayLength> cmp::Eq for NumericArray<T, N> where T: cmp::Eq {}

impl<T, N: ArrayLength> PartialOrd<Self> for NumericArray<T, N>
where
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, rhs: &Self) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&self.0, &rhs.0)
    }

    #[inline]
    fn lt(&self, rhs: &Self) -> bool {
        PartialOrd::lt(&self.0, &rhs.0)
    }

    #[inline]
    fn le(&self, rhs: &Self) -> bool {
        PartialOrd::le(&self.0, &rhs.0)
    }

    #[inline]
    fn gt(&self, rhs: &Self) -> bool {
        PartialOrd::gt(&self.0, &rhs.0)
    }

    #[inline]
    fn ge(&self, rhs: &Self) -> bool {
        PartialOrd::ge(&self.0, &rhs.0)
    }
}

impl<T, N: ArrayLength> PartialOrd<GenericArray<T, N>> for NumericArray<T, N>
where
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, rhs: &GenericArray<T, N>) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&self.0, rhs)
    }

    #[inline]
    fn lt(&self, rhs: &GenericArray<T, N>) -> bool {
        PartialOrd::lt(&self.0, rhs)
    }

    #[inline]
    fn le(&self, rhs: &GenericArray<T, N>) -> bool {
        PartialOrd::le(&self.0, rhs)
    }

    #[inline]
    fn gt(&self, rhs: &GenericArray<T, N>) -> bool {
        PartialOrd::gt(&self.0, rhs)
    }

    #[inline]
    fn ge(&self, rhs: &GenericArray<T, N>) -> bool {
        PartialOrd::ge(&self.0, rhs)
    }
}

impl<T, N: ArrayLength> cmp::Ord for NumericArray<T, N>
where
    T: cmp::Ord,
{
    #[inline]
    fn cmp(&self, rhs: &Self) -> cmp::Ordering {
        cmp::Ord::cmp(&self.0, &rhs.0)
    }
}

impl<T, N: ArrayLength> NumericArray<T, N> {
    /// Creates a new `NumericArray` instance from a `GenericArray` instance.
    ///
    /// Example:
    ///
    /// ```
    /// #[macro_use]
    /// extern crate generic_array;
    /// extern crate numeric_array;
    ///
    /// use numeric_array::NumericArray;
    ///
    /// fn main() {
    ///     let arr = NumericArray::new(arr![1, 2, 3, 4]);
    ///
    ///     println!("{:?}", arr); // Prints 'NumericArray([1, 2, 3, 4])'
    /// }
    /// ```
    #[inline]
    pub fn new(arr: GenericArray<T, N>) -> NumericArray<T, N> {
        NumericArray(arr)
    }

    /// Creates a new array filled with a single value.
    ///
    /// Example:
    ///
    /// ```ignore
    /// let a = NumericArray::new(arr![5, 5, 5, 5]);
    /// let b = NumericArray::splat(5);
    ///
    /// assert_eq!(a, b);
    /// ```
    #[inline]
    pub fn splat(t: T) -> NumericArray<T, N>
    where
        T: Clone,
    {
        NumericArray(GenericArray::generate(|_| t.clone()))
    }

    /// Convert all elements of the `NumericArray` to another `NumericArray` using `From`
    pub fn convert<U: From<T>>(self) -> NumericArray<U, N> {
        self.0.map(From::from).into()
    }

    /// Consumes self and returns the internal `GenericArray` instance
    #[inline]
    pub fn into_array(self) -> GenericArray<T, N> {
        self.0
    }

    /// Get reference to underlying `GenericArray` instance.
    #[inline]
    pub fn as_array(&self) -> &GenericArray<T, N> {
        &self.0
    }

    /// Get mutable reference to underlying `GenericArray` instance.
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut GenericArray<T, N> {
        &mut self.0
    }

    /// Extracts a slice containing the entire array.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Extracts a mutable slice containing the entire array.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    /// Converts slice to a numeric array reference with inferred length;
    ///
    /// Length of the slice must be equal to the length of the array.
    #[inline(always)]
    pub fn from_slice(slice: &[T]) -> &NumericArray<T, N> {
        slice.into()
    }

    /// Converts mutable slice to a mutable numeric array reference
    ///
    /// Length of the slice must be equal to the length of the array.
    #[inline(always)]
    pub fn from_mut_slice(slice: &mut [T]) -> &mut NumericArray<T, N> {
        slice.into()
    }
}

use core::ops::Sub;
use typenum::{bit::B1 as True, Diff, IsGreaterOrEqual};

impl<T, N: ArrayLength> NumericArray<T, N> {
    /// Offset the numeric array and cast it into a shorter array
    #[inline(always)]
    pub fn offset<V: ArrayLength, O: ArrayLength>(&self) -> &NumericArray<T, V>
    where
        N: Sub<O>,
        Diff<N, O>: IsGreaterOrEqual<V, Output = True>,
    {
        unsafe { &*((self as *const _ as *const T).add(O::USIZE) as *const NumericArray<T, V>) }
    }

    /// Offset the numeric array and cast it into a shorter array
    #[inline(always)]
    pub fn offset_mut<V: ArrayLength, O: ArrayLength>(&mut self) -> &mut NumericArray<T, V>
    where
        N: Sub<O>,
        Diff<N, O>: IsGreaterOrEqual<V, Output = True>,
    {
        unsafe { &mut *((self as *mut _ as *mut T).add(O::USIZE) as *mut NumericArray<T, V>) }
    }
}

impl<'a, T, N: ArrayLength> From<&'a [T]> for &'a NumericArray<T, N> {
    /// Converts slice to a numeric array reference with inferred length;
    ///
    /// Length of the slice must be equal to the length of the array.
    #[inline(always)]
    fn from(slice: &[T]) -> &NumericArray<T, N> {
        debug_assert_eq!(slice.len(), N::to_usize());

        unsafe { &*(slice.as_ptr() as *const NumericArray<T, N>) }
    }
}

impl<'a, T, N: ArrayLength> From<&'a mut [T]> for &'a mut NumericArray<T, N> {
    /// Converts mutable slice to a mutable numeric array reference
    ///
    /// Length of the slice must be equal to the length of the array.
    #[inline(always)]
    fn from(slice: &mut [T]) -> &mut NumericArray<T, N> {
        debug_assert_eq!(slice.len(), N::to_usize());

        unsafe { &mut *(slice.as_mut_ptr() as *mut NumericArray<T, N>) }
    }
}

impl<T, N: ArrayLength> AsRef<[T]> for NumericArray<T, N> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T, N: ArrayLength> Borrow<[T]> for NumericArray<T, N> {
    #[inline(always)]
    fn borrow(&self) -> &[T] {
        self
    }
}

impl<T, N: ArrayLength> AsMut<[T]> for NumericArray<T, N> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T, N: ArrayLength> BorrowMut<[T]> for NumericArray<T, N> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T, N: ArrayLength> Index<usize> for NumericArray<T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &T {
        self.0.index(index)
    }
}

impl<T, N: ArrayLength> IndexMut<usize> for NumericArray<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.0.index_mut(index)
    }
}

impl<T, N: ArrayLength> Index<Range<usize>> for NumericArray<T, N> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, index: Range<usize>) -> &[T] {
        self.0.index(index)
    }
}

impl<T, N: ArrayLength> IndexMut<Range<usize>> for NumericArray<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
        self.0.index_mut(index)
    }
}

impl<T, N: ArrayLength> Index<RangeTo<usize>> for NumericArray<T, N> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, index: RangeTo<usize>) -> &[T] {
        self.0.index(index)
    }
}

impl<T, N: ArrayLength> IndexMut<RangeTo<usize>> for NumericArray<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
        self.0.index_mut(index)
    }
}

impl<T, N: ArrayLength> Index<RangeFrom<usize>> for NumericArray<T, N> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, index: RangeFrom<usize>) -> &[T] {
        self.0.index(index)
    }
}

impl<T, N: ArrayLength> IndexMut<RangeFrom<usize>> for NumericArray<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
        self.0.index_mut(index)
    }
}

impl<T, N: ArrayLength> Index<RangeFull> for NumericArray<T, N> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, _index: RangeFull) -> &[T] {
        self
    }
}

impl<T, N: ArrayLength> IndexMut<RangeFull> for NumericArray<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        self
    }
}

impl<'a, T, N: ArrayLength> IntoIterator for &'a NumericArray<T, N> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, N: ArrayLength> IntoIterator for &'a mut NumericArray<T, N> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, N: ArrayLength> IntoIterator for NumericArray<T, N> {
    type Item = T;
    type IntoIter = GenericArrayIter<T, N>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T, N: ArrayLength> FromIterator<T> for NumericArray<T, N> {
    #[inline(always)]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        NumericArray(GenericArray::from_iter(iter))
    }
}

impl<T, N: ArrayLength> Default for NumericArray<T, N>
where
    T: Default,
{
    #[inline(always)]
    fn default() -> Self {
        NumericArray(GenericArray::default())
    }
}

#[cfg(test)]
pub mod tests {
    use num_traits::float::FloatCore;

    // This stops the compiler from optimizing based on known data, only data types.
    #[inline(never)]
    pub fn black_box<T>(val: T) -> T {
        use core::{mem, ptr};

        let ret = unsafe { ptr::read_volatile(&val) };
        mem::forget(val);
        ret
    }

    #[test]
    fn test_ops() {
        let a = black_box(narr![1, 3, 5, 7]);
        let b = black_box(narr![2, 4, 6, 8]);

        let c = a + b;
        let d = c * nconstant!(black_box(5));
        let e = d << nconstant!(1_usize);

        assert_eq!(e, narr![30, 70, 110, 150])
    }

    #[test]
    fn test_constants() {
        let a = black_box(narr![1, 3, 5, 7]);
        let b = black_box(narr![2, 4, 6, 8]);

        let c = a + b * nconstant!(2);

        assert_eq!(c, narr![5, 11, 17, 23]);
    }

    #[test]
    fn test_floats() {
        let a = black_box(narr![1.0f32, 3.0, 5.0, 7.0]);
        let b = black_box(narr![2.0f32, 4.0, 6.0, 8.0]);

        let c = a + b;

        black_box(c);
    }

    #[test]
    fn test_other() {
        use num_traits::Saturating;

        let a = black_box(narr![1, 3, 5, 7]);
        let b = black_box(narr![2, 4, 6, 8]);

        let c = a.saturating_add(b);

        black_box(c);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_atan2() {
        use num_traits::Float;

        let a = black_box(narr![1.0f32, 2.0, 3.0, 4.0]);
        let b = black_box(narr![2.0f32, 3.0, 4.0, 5.0]);

        let c = a.atan2(b);

        assert_eq!(c, narr![0.4636476, 0.5880026, 0.6435011, 0.67474097]);
    }

    #[test]
    fn test_classify() {
        use core::num::FpCategory;

        let nan = f32::nan();
        let infinity = f32::infinity();

        let any_nan = black_box(narr![1.0, 2.0, nan, 0.0]);
        let any_infinite = black_box(narr![1.0, infinity, 2.0, 3.0]);
        let any_mixed = black_box(narr![1.0, infinity, nan, 0.0]);
        let all_normal = black_box(narr![1.0, 2.0, 3.0, 4.0]);
        let all_zero = black_box(narr![0.0, 0.0, 0.0, 0.0]);

        let non_zero = black_box(narr![0.0f32, 1.0, 0.0, 0.0]);

        assert_eq!(any_nan.classify(), FpCategory::Nan);
        assert_eq!(any_mixed.classify(), FpCategory::Nan);
        assert_eq!(any_infinite.classify(), FpCategory::Infinite);
        assert_eq!(all_normal.classify(), FpCategory::Normal);
        assert_eq!(all_zero.classify(), FpCategory::Zero);

        assert_eq!(non_zero.classify(), FpCategory::Normal);

        assert!(!any_nan.is_infinite());
        assert!(any_mixed.is_infinite());
        assert!(any_nan.is_nan());
        assert!(any_mixed.is_nan());
        assert!(!any_infinite.is_nan());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_tanh() {
        use num_traits::Float;

        let a = black_box(narr![1.0f32, 2.0, 3.0, 4.0]);

        black_box(a.tanh());
    }

    #[cfg(feature = "std")]
    #[test]
    pub fn test_madd() {
        use num_traits::Float;

        let a = black_box(narr![1.0f32, 2.0, 3.0, 4.0]);
        let b = black_box(narr![5.0f32, 6.0, 7.0, 8.0]);
        let c = black_box(narr![9.0f32, 1.0, 2.0, 3.0]);

        let d = a.mul_add(b, c);

        assert_eq!(d, narr![14.0, 13.0, 23.0, 35.0]);
    }

    #[test]
    #[no_mangle]
    pub fn test_select() {
        use crate::simd::Select;

        let mask = black_box(narr![true, false, false, true]);

        let a = black_box(narr![1, 2, 3, 4]);
        let b = black_box(narr![5, 6, 7, 8]);

        // Compiles to vblendvps
        let selected = mask.select(a, b);

        assert_eq!(selected, narr![1, 6, 7, 4]);
    }
}
