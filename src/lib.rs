//! `numeric-array` is a wrapper around
//! [`generic-array`](https://github.com/fizyk20/generic-array) that adds
//! efficient numeric trait implementations, often times making use of
//! SIMD instructions and compile-time evaluations.
//!
//! All stable `std::ops` traits are implemented for `NumericArray` itself,
//! plus the thin `NumericConstant` type, which is required to
//! differeniate constant values from `NumericArray` itself.
//!
//! Additionally, most of `num_traits` are implemented,
//! including `Num` itself. So you can even use a whole array as a generic number.
//!
//! Example:
//!
//! ```rust
//! extern crate num_traits;
//! #[macro_use]
//! extern crate generic_array;
//! #[macro_use]
//! extern crate numeric_array;
//!
//! use num_traits::Float;
//! use numeric_array::NumericArray;
//!
//! fn main() {
//!     let a = narr![f32; 1, 2, 3, 4];
//!     let b = narr![f32; 5, 6, 7, 8];
//!     let c = narr![f32; 9, 1, 2, 3];
//!
//!     // Compiles to a single vfmadd213ps instruction on my machine
//!     let d = a.mul_add(b, c);
//!
//!     assert_eq!(d, narr![f32; 14, 13, 23, 35]);
//! }
//! ```
//!
//! When used with `RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"`,
//! then Rust and LLVM are smart enough to turn almost all operations
//! into SIMD instructions, or even just evaluate them at compile time.
//! The above example is actually evaluated at compile time,
//! so if you were to view the assembly it would show the result only.
//! Rust is pretty smart.
//!
//! Therefore, this is ideal for situations where simple component-wise
//! operations are required for arrays.
//!

#![deny(missing_docs)]

extern crate num_traits;

#[cfg(test)]
extern crate typenum;

#[cfg_attr(test, macro_use)]
extern crate generic_array;

use std::{cmp, mem, ptr, slice};

use std::mem::ManuallyDrop;

use std::borrow::{Borrow, BorrowMut};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use std::iter::FromIterator;

use std::fmt::{Debug, Formatter, Result as FmtResult};

use generic_array::functional::*;
use generic_array::sequence::*;
use generic_array::{ArrayLength, GenericArray, GenericArrayIter};

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
#[repr(C)]
pub struct NumericArray<T, N: ArrayLength<T>>(GenericArray<T, N>);

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
        $crate::NumericArray::new(arr!($($t)*))
    }
}

unsafe impl<T, N: ArrayLength<T>> GenericSequence<T> for NumericArray<T, N> {
    type Length = N;
    type Sequence = Self;

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

impl<T: Debug, N: ArrayLength<T>> Debug for NumericArray<T, N> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.debug_tuple("NumericArray").field(&self.0).finish()
    }
}

impl<X, T, N: ArrayLength<T>> From<X> for NumericArray<T, N>
where
    X: Into<GenericArray<T, N>>,
{
    fn from(x: X) -> NumericArray<T, N> {
        NumericArray::new(x.into())
    }
}

impl<T: Clone, N: ArrayLength<T>> Clone for NumericArray<T, N> {
    fn clone(&self) -> NumericArray<T, N> {
        NumericArray(self.0.clone())
    }
}

impl<T: Copy, N: ArrayLength<T>> Copy for NumericArray<T, N>
where
    N::ArrayType: Copy,
{
}

impl<T, N: ArrayLength<T>> Deref for NumericArray<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, N: ArrayLength<T>> DerefMut for NumericArray<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T, U, N: ArrayLength<T> + ArrayLength<U>> PartialEq<NumericArray<U, N>> for NumericArray<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, rhs: &NumericArray<U, N>) -> bool {
        **self == **rhs
    }
}

impl<T, U, N: ArrayLength<T> + ArrayLength<U>> PartialEq<GenericArray<U, N>> for NumericArray<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, rhs: &GenericArray<U, N>) -> bool {
        **self == **rhs
    }

    fn ne(&self, rhs: &GenericArray<U, N>) -> bool {
        **self == **rhs
    }
}

impl<T, N: ArrayLength<T>> cmp::Eq for NumericArray<T, N>
where
    T: cmp::Eq,
{
}

impl<T, N: ArrayLength<T>> PartialOrd<Self> for NumericArray<T, N>
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

impl<T, N: ArrayLength<T>> PartialOrd<GenericArray<T, N>> for NumericArray<T, N>
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

impl<T, N: ArrayLength<T>> cmp::Ord for NumericArray<T, N>
where
    T: cmp::Ord,
{
    #[inline]
    fn cmp(&self, rhs: &Self) -> cmp::Ordering {
        cmp::Ord::cmp(&self.0, &rhs.0)
    }
}

impl<T, N: ArrayLength<T>> NumericArray<T, N> {
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
    ///     let arr = NumericArray::new(arr![i32; 1, 2, 3, 4]);
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
    /// let a = NumericArray::new(arr![i32; 5, 5, 5, 5]);
    /// let b = NumericArray::from_element(5);
    ///
    /// assert_eq!(a, b);
    /// ```
    #[inline]
    pub fn from_element(t: T) -> NumericArray<T, N>
    where
        T: Clone,
    {
        NumericArray(GenericArray::generate(|_| t.clone()))
    }

    /// Convert all elements of the `NumericArray` to another `NumericArray` using `From`
    pub fn convert<U: From<T>>(self) -> NumericArray<U, N>
    where
        N: ArrayLength<U>,
    {
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
    #[inline]
    pub fn as_mut_array(&mut self) -> &mut GenericArray<T, N> {
        &mut self.0
    }

    /// Extracts a slice containing the entire array.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Extracts a mutable slice containing the entire array.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T, N: ArrayLength<T>> AsRef<[T]> for NumericArray<T, N> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T, N: ArrayLength<T>> Borrow<[T]> for NumericArray<T, N> {
    fn borrow(&self) -> &[T] {
        self
    }
}

impl<T, N: ArrayLength<T>> AsMut<[T]> for NumericArray<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T, N: ArrayLength<T>> BorrowMut<[T]> for NumericArray<T, N> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T, N: ArrayLength<T>> Index<usize> for NumericArray<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &T {
        &(**self)[index]
    }
}

impl<T, N: ArrayLength<T>> IndexMut<usize> for NumericArray<T, N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut (**self)[index]
    }
}

impl<T, N: ArrayLength<T>> Index<Range<usize>> for NumericArray<T, N> {
    type Output = [T];

    #[inline]
    fn index(&self, index: Range<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}

impl<T, N: ArrayLength<T>> Index<RangeTo<usize>> for NumericArray<T, N> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeTo<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}

impl<T, N: ArrayLength<T>> Index<RangeFrom<usize>> for NumericArray<T, N> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeFrom<usize>) -> &[T] {
        Index::index(&**self, index)
    }
}

impl<T, N: ArrayLength<T>> Index<RangeFull> for NumericArray<T, N> {
    type Output = [T];

    #[inline]
    fn index(&self, _index: RangeFull) -> &[T] {
        self
    }
}

impl<'a, T, N: ArrayLength<T>> IntoIterator for &'a NumericArray<T, N> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, N: ArrayLength<T>> IntoIterator for &'a mut NumericArray<T, N> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, N: ArrayLength<T>> IntoIterator for NumericArray<T, N> {
    type Item = T;
    type IntoIter = GenericArrayIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T, N: ArrayLength<T>> FromIterator<T> for NumericArray<T, N> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        NumericArray(GenericArray::from_iter(iter))
    }
}

impl<T, N: ArrayLength<T>> Default for NumericArray<T, N>
where
    T: Default,
{
    fn default() -> Self {
        NumericArray(GenericArray::default())
    }
}

// Just copied from `generic-array` internals
struct ArrayBuilder<T, N: ArrayLength<T>> {
    array: ManuallyDrop<GenericArray<T, N>>,
    position: usize,
}

impl<T, N: ArrayLength<T>> ArrayBuilder<T, N> {
    fn new() -> ArrayBuilder<T, N> {
        ArrayBuilder {
            array: ManuallyDrop::new(unsafe { mem::uninitialized() }),
            position: 0,
        }
    }

    fn into_inner(self) -> GenericArray<T, N> {
        let array = unsafe { ptr::read(&self.array) };

        mem::forget(self);

        ManuallyDrop::into_inner(array)
    }
}

impl<T, N: ArrayLength<T>> Drop for ArrayBuilder<T, N> {
    fn drop(&mut self) {
        for value in self.array.iter_mut().take(self.position) {
            unsafe {
                ptr::drop_in_place(value);
            }
        }
    }
}

struct ArrayConsumer<T, N: ArrayLength<T>> {
    array: ManuallyDrop<GenericArray<T, N>>,
    position: usize,
}

impl<T, N: ArrayLength<T>> ArrayConsumer<T, N> {
    fn new(array: GenericArray<T, N>) -> ArrayConsumer<T, N> {
        ArrayConsumer {
            array: ManuallyDrop::new(array),
            position: 0,
        }
    }
}

impl<T, N: ArrayLength<T>> Drop for ArrayConsumer<T, N> {
    fn drop(&mut self) {
        for i in self.position..N::to_usize() {
            unsafe {
                ptr::drop_in_place(self.array.get_unchecked_mut(i));
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    // This stops the compiler from optimizing based on known data, only data types.
    #[inline(never)]
    pub fn black_box<T>(val: T) -> T {
        use std::{mem, ptr};

        let ret = unsafe { ptr::read_volatile(&val) };
        mem::forget(val);
        ret
    }

    #[test]
    fn test_ops() {
        let a = black_box(narr![i32; 1, 3, 5, 7]);
        let b = black_box(narr![i32; 2, 4, 6, 8]);

        let c = a + b;
        let d = c * nconstant!(black_box(5));
        let e = d << nconstant!(1_usize);

        assert_eq!(e, narr![i32; 30, 70, 110, 150])
    }

    #[test]
    fn test_constants() {
        let a = black_box(narr![i32; 1, 3, 5, 7]);
        let b = black_box(narr![i32; 2, 4, 6, 8]);

        let c = a + b * nconstant!(2);

        assert_eq!(c, narr![i32; 5, 11, 17, 23]);
    }

    #[test]
    fn test_floats() {
        let a = black_box(narr![f32; 1.0, 3.0, 5.0, 7.0]);
        let b = black_box(narr![f32; 2.0, 4.0, 6.0, 8.0]);

        let c = a + b;

        black_box(c);
    }

    #[test]
    fn test_other() {
        use num_traits::Saturating;

        let a = black_box(narr![i32; 1, 3, 5, 7]);
        let b = black_box(narr![i32; 2, 4, 6, 8]);

        let c = a.saturating_add(b);

        black_box(c);
    }

    #[test]
    fn test_atan2() {
        use num_traits::Float;

        let a = black_box(narr![f32; 1, 2, 3, 4]);
        let b = black_box(narr![f32; 2, 3, 4, 5]);

        let c = a.atan2(b);

        assert_eq!(c, narr![f32; 0.4636476, 0.5880026, 0.6435011, 0.67474097]);
    }

    #[test]
    fn test_classify() {
        use num_traits::Float;
        use std::num::FpCategory;

        let nan = f32::nan();
        let infinity = f32::infinity();

        let any_nan = black_box(narr![f32; 1, 2, nan, 0]);
        let any_infinite = black_box(narr![f32; 1, infinity, 2, 3]);
        let any_mixed = black_box(narr![f32; 1, infinity, nan, 0]);
        let all_normal = black_box(narr![f32; 1, 2, 3, 4]);
        let all_zero = black_box(narr![f32; 0, 0, 0, 0]);

        let non_zero = black_box(narr![f32; 0, 1, 0, 0]);

        assert_eq!(any_nan.classify(), FpCategory::Nan);
        assert_eq!(any_mixed.classify(), FpCategory::Nan);
        assert_eq!(any_infinite.classify(), FpCategory::Infinite);
        assert_eq!(all_normal.classify(), FpCategory::Normal);
        assert_eq!(all_zero.classify(), FpCategory::Zero);

        assert_eq!(non_zero.classify(), FpCategory::Normal);

        assert_eq!(any_nan.is_infinite(), false);
        assert_eq!(any_mixed.is_infinite(), true);
        assert_eq!(any_nan.is_nan(), true);
        assert_eq!(any_mixed.is_nan(), true);
        assert_eq!(any_infinite.is_nan(), false);
    }

    #[test]
    fn test_tanh() {
        use num_traits::Float;

        let a = black_box(narr![f32; 1, 2, 3, 4]);

        let b = a.tanh();

        println!("{:?}", b);
    }

    #[test]
    pub fn test_madd() {
        use num_traits::Float;

        let a = black_box(narr![f32; 1, 2, 3, 4]);
        let b = black_box(narr![f32; 5, 6, 7, 8]);
        let c = black_box(narr![f32; 9, 1, 2, 3]);

        let d = a.mul_add(b, c);

        assert_eq!(d, narr![f32; 14, 13, 23, 35]);
    }

    #[test]
    #[no_mangle]
    pub fn test_select() {
        use simd::Select;

        let mask = black_box(narr![bool; true, false, false, true]);

        let a = black_box(narr![i32; 1, 2, 3, 4]);
        let b = black_box(narr![i32; 5, 6, 7, 8]);

        // Compiles to vblendvps
        let selected = mask.select(a, b);

        assert_eq!(selected, narr![i32; 1, 6, 7, 4]);
    }
}
