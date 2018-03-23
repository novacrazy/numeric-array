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

use generic_array::{ArrayLength, GenericArray, GenericArrayIter};
use generic_array::sequence::*;

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
    ($value: expr) => {
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

impl<T, N: ArrayLength<T>> From<GenericArray<T, N>> for NumericArray<T, N> {
    fn from(arr: GenericArray<T, N>) -> NumericArray<T, N> {
        NumericArray(arr)
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

macro_rules! impl_unary_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, N: ArrayLength<T>> $op_trait for NumericArray<T, N>
            where
                T: $op_trait,
                N: ArrayLength<<T as $op_trait>::Output>
            {
                type Output = NumericArray<<T as $op_trait>::Output, N>;

                fn $op(self) -> Self::Output {
                    NumericArray(self.0.map($op_trait::$op))
                }
            }

            impl<'a, T: Clone, N: ArrayLength<T>> $op_trait for &'a NumericArray<T, N>
            where
                T: $op_trait,
                N: ArrayLength<<T as $op_trait>::Output>
            {
                type Output = NumericArray<<T as $op_trait>::Output, N>;

                fn $op(self) -> Self::Output {
                    NumericArray((&self.0).map(|x| $op_trait::$op(x.clone())))
                }
            }
        )*
    }
}

macro_rules! impl_binary_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, U, N: ArrayLength<T> + ArrayLength<U>> $op_trait<NumericArray<U, N>> for NumericArray<T, N>
            where
                T: $op_trait<U>,
                N: ArrayLength<<T as $op_trait<U>>::Output>
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                fn $op(self, rhs: NumericArray<U, N>) -> Self::Output {
                    NumericArray(self.0.zip(rhs.0, $op_trait::$op))
                }
            }

            impl<'a, T, U: Clone, N: ArrayLength<T> + ArrayLength<U>> $op_trait<&'a NumericArray<U, N>> for NumericArray<T, N>
            where
                T: $op_trait<U>,
                N: ArrayLength<<T as $op_trait<U>>::Output>
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                fn $op(self, rhs: &'a NumericArray<U, N>) -> Self::Output {
                    NumericArray(self.0.zip(&rhs.0, |l, r| $op_trait::$op(l, r.clone())))
                }
            }

            impl<'a, T: Clone, U, N: ArrayLength<T> + ArrayLength<U>> $op_trait<NumericArray<U, N>> for &'a NumericArray<T, N>
            where
                T: $op_trait<U>,
                N: ArrayLength<<T as $op_trait<U>>::Output>
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                fn $op(self, rhs: NumericArray<U, N>) -> Self::Output {
                    NumericArray((&self.0).zip(rhs.0, |l, r| $op_trait::$op(l.clone(), r)))
                }
            }

            impl<'a, 'b, T: Clone, U: Clone, N: ArrayLength<T> + ArrayLength<U>> $op_trait<&'b NumericArray<U, N>> for &'a NumericArray<T, N>
            where
                T: $op_trait<U>,
                N: ArrayLength<<T as $op_trait<U>>::Output>
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                fn $op(self, rhs: &'b NumericArray<U, N>) -> Self::Output {
                    NumericArray((&self.0).zip(&rhs.0, |l, r| $op_trait::$op(l.clone(), r.clone())))
                }
            }

            impl<T, U: Clone, N: ArrayLength<T>> $op_trait<NumericConstant<U>> for NumericArray<T, N>
            where
                T: $op_trait<U>,
                N: ArrayLength<<T as $op_trait<U>>::Output>
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                fn $op(self, rhs: NumericConstant<U>) -> Self::Output {
                    NumericArray(self.0.map(|l| $op_trait::$op(l, rhs.0.clone())))
                }
            }

            impl<'a, T: Clone, U: Clone, N: ArrayLength<T>> $op_trait<NumericConstant<U>> for &'a NumericArray<T, N>
            where
                T: $op_trait<U>,
                N: ArrayLength<<T as $op_trait<U>>::Output>
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                fn $op(self, rhs: NumericConstant<U>) -> Self::Output {
                    NumericArray((&self.0).map(|l| $op_trait::$op(l.clone(), rhs.0.clone())))
                }
            }
        )*
    }
}

macro_rules! impl_assign_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, U, N: ArrayLength<T> + ArrayLength<U>> $op_trait<NumericArray<U, N>> for NumericArray<T, N>
            where
                T: $op_trait<U>
            {
                fn $op(&mut self, rhs: NumericArray<U, N>) {
                    let mut consumer = ArrayConsumer::new(rhs.0);

                    for (lhs, rhs) in self.iter_mut().zip(consumer.array.iter()) {
                        $op_trait::$op(lhs, unsafe { ptr::read(rhs) });

                        consumer.position += 1;
                    }
                }
            }

            impl<'a, T, U: Clone, N: ArrayLength<T> + ArrayLength<U>> $op_trait<&'a NumericArray<U, N>> for NumericArray<T, N>
            where
                T: $op_trait<U>
            {
                fn $op(&mut self, rhs: &'a NumericArray<U, N>) {
                    for (lhs, rhs) in self.iter_mut().zip(rhs.iter()) {
                        $op_trait::$op(lhs, rhs.clone());
                    }
                }
            }

            impl<T, U: Clone, N: ArrayLength<T>> $op_trait<NumericConstant<U>> for NumericArray<T, N>
            where
                T: $op_trait<U>
            {
                fn $op(&mut self, rhs: NumericConstant<U>) {
                    for lhs in self.iter_mut() {
                        $op_trait::$op(lhs, rhs.0.clone());
                    }
                }
            }
        )*
    }
}

macro_rules! impl_wrapping_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, N: ArrayLength<T>> num_traits::$op_trait for NumericArray<T, N>
            where
                T: num_traits::$op_trait
            {
                fn $op(&self, rhs: &Self) -> Self {
                    NumericArray((&self.0).zip(&rhs.0, num_traits::$op_trait::$op))
                }
            }
        )*
    }
}

macro_rules! impl_checked_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, N: ArrayLength<T>> $op_trait for NumericArray<T, N>
            where
                T: $op_trait
            {
                fn $op(&self, rhs: &Self) -> Option<Self> {
                    let mut builder = ArrayBuilder::new();

                    for (dst, (lhs, rhs)) in builder.array.iter_mut().zip(self.iter().zip(rhs.iter())) {
                        if let Some(value) = $op_trait::$op(lhs, rhs) {
                            unsafe {
                                ptr::write(dst, value);
                            }

                            builder.position += 1;
                        } else {
                            return None;
                        }
                    }

                    Some(NumericArray(builder.into_inner()))
                }
            }
        )*
    }
}

macro_rules! impl_float_const {
    ($($f:ident),*) => {
        impl<T, N: ArrayLength<T>> FloatConst for NumericArray<T, N>
        where
            T: FloatConst
        {
            $(
                fn $f() -> Self {
                    NumericArray(GenericArray::generate(|_| <T as FloatConst>::$f()))
                }
            )*
        }
    }
}

pub mod impls {
    //! Implementation notes
    //!
    //! For any method that returns a single element, like `ToPrimitive` methods and `Float::integer_decode`,
    //! the first element of the array will be used. If the array length is zero, `None` or zero is returned.
    //!
    //! For any method that accepts a single value, the same value is used across the entire operation. For example,
    //! `powi` will raise the power of every element by the given value.
    //!
    //! All floating point classification functions such as `is_finite`, `is_nan`, and `classify`, among others,
    //! follow a rule of highest importance. `NaN`s take precedence over `Infinite`, `Infinite` takes precedence over
    //! `Subnormal`, `Subnormal` takes precedence over `Normal`, `Normal` takes precedence over `Zero`
    //!
    //! This means that `classify` and `is_nan` will return `NaN`/true if any values are `NaN`,
    //! and `is_normal` will only return true when ALL values are normal.
    //!
    //! Additionally, similar rules are implemented for `is_sign_positive`/`is_positive` and `is_sign_negative`/`is_negative`, where `is_sign_positive` is
    //! true if all values are positive, but `is_sign_negative` is true when any value is negative.

    use super::*;

    use generic_array::functional::*;

    use std::ops::*;
    use std::num::FpCategory;

    use num_traits::*;

    impl_unary_ops! {
        Inv::inv, // num_traits
        Neg::neg,
        Not::not
    }

    impl_binary_ops! {
        Pow::pow, // num_traits
        Add::add,
        Sub::sub,
        Mul::mul,
        Div::div,
        Rem::rem,
        BitAnd::bitand,
        BitOr::bitor,
        BitXor::bitxor,
        Shr::shr,
        Shl::shl
    }

    impl_assign_ops! {
        AddAssign::add_assign,
        SubAssign::sub_assign,
        MulAssign::mul_assign,
        DivAssign::div_assign,
        RemAssign::rem_assign,
        BitAndAssign::bitand_assign,
        BitOrAssign::bitor_assign,
        BitXorAssign::bitxor_assign,
        ShrAssign::shr_assign,
        ShlAssign::shl_assign
    }

    impl_wrapping_ops! {
        WrappingAdd::wrapping_add,
        WrappingSub::wrapping_sub,
        WrappingMul::wrapping_mul
    }

    impl_checked_ops! {
        CheckedAdd::checked_add,
        CheckedSub::checked_sub,
        CheckedMul::checked_mul,
        CheckedDiv::checked_div
    }

    impl<T, N: ArrayLength<T>> CheckedShl for NumericArray<T, N>
    where
        T: CheckedShl,
        Self: Shl<u32, Output = Self>,
    {
        fn checked_shl(&self, rhs: u32) -> Option<Self> {
            let mut builder = ArrayBuilder::new();

            for (dst, lhs) in builder.array.iter_mut().zip(self.iter()) {
                if let Some(value) = CheckedShl::checked_shl(lhs, rhs) {
                    unsafe {
                        ptr::write(dst, value);
                    }

                    builder.position += 1;
                } else {
                    return None;
                }
            }

            Some(NumericArray(builder.into_inner()))
        }
    }

    impl<T, N: ArrayLength<T>> CheckedShr for NumericArray<T, N>
    where
        T: CheckedShr,
        Self: Shr<u32, Output = Self>,
    {
        fn checked_shr(&self, rhs: u32) -> Option<Self> {
            let mut builder = ArrayBuilder::new();

            for (dst, lhs) in builder.array.iter_mut().zip(self.iter()) {
                if let Some(value) = CheckedShr::checked_shr(lhs, rhs) {
                    unsafe {
                        ptr::write(dst, value);
                    }

                    builder.position += 1;
                } else {
                    return None;
                }
            }

            Some(NumericArray(builder.into_inner()))
        }
    }

    impl_float_const!(
        E,
        FRAC_1_PI,
        FRAC_1_SQRT_2,
        FRAC_2_PI,
        FRAC_2_SQRT_PI,
        FRAC_PI_2,
        FRAC_PI_3,
        FRAC_PI_4,
        FRAC_PI_6,
        FRAC_PI_8,
        LN_10,
        LN_2,
        LOG10_E,
        LOG2_E,
        PI,
        SQRT_2
    );

    impl<T, N: ArrayLength<T>> Zero for NumericArray<T, N>
    where
        T: Zero,
    {
        fn zero() -> Self {
            NumericArray(GenericArray::generate(|_| <T as Zero>::zero()))
        }

        fn is_zero(&self) -> bool {
            self.iter().all(|x| x.is_zero())
        }
    }

    impl<T, N: ArrayLength<T>> One for NumericArray<T, N>
    where
        T: One,
    {
        fn one() -> Self {
            NumericArray(GenericArray::generate(|_| <T as One>::one()))
        }
    }

    impl<T, N: ArrayLength<T>> Saturating for NumericArray<T, N>
    where
        T: Saturating,
    {
        fn saturating_add(self, rhs: Self) -> Self {
            NumericArray(self.0.zip(rhs.0, Saturating::saturating_add))
        }

        fn saturating_sub(self, rhs: Self) -> Self {
            NumericArray(self.0.zip(rhs.0, Saturating::saturating_sub))
        }
    }

    impl<T: Clone, N: ArrayLength<T>> Num for NumericArray<T, N>
    where
        T: Num,
    {
        type FromStrRadixErr = <T as Num>::FromStrRadixErr;

        fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
            <T as Num>::from_str_radix(str, radix).map(Self::from_element)
        }
    }

    impl<T: Clone, N: ArrayLength<T>> Signed for NumericArray<T, N>
    where
        T: Signed,
    {
        fn abs(&self) -> Self {
            NumericArray((&self.0).map(Signed::abs))
        }

        fn abs_sub(&self, rhs: &Self) -> Self {
            NumericArray((&self.0).zip(&rhs.0, Signed::abs_sub))
        }

        fn signum(&self) -> Self {
            NumericArray((&self.0).map(Signed::signum))
        }

        fn is_positive(&self) -> bool {
            self.iter().all(Signed::is_positive)
        }

        fn is_negative(&self) -> bool {
            self.iter().any(Signed::is_negative)
        }
    }

    impl<T: Clone, N: ArrayLength<T>> Unsigned for NumericArray<T, N>
    where
        T: Unsigned,
    {
    }

    impl<T, N: ArrayLength<T>> Bounded for NumericArray<T, N>
    where
        T: Bounded,
    {
        fn min_value() -> Self {
            NumericArray(GenericArray::generate(|_| <T as Bounded>::min_value()))
        }

        fn max_value() -> Self {
            NumericArray(GenericArray::generate(|_| <T as Bounded>::max_value()))
        }
    }

    impl<T, N: ArrayLength<T>> ToPrimitive for NumericArray<T, N>
    where
        T: ToPrimitive,
    {
        #[inline]
        fn to_i64(&self) -> Option<i64> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_i64())
            }
        }

        #[inline]
        fn to_u64(&self) -> Option<u64> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_u64())
            }
        }

        #[inline]
        fn to_isize(&self) -> Option<isize> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_isize())
            }
        }

        #[inline]
        fn to_i8(&self) -> Option<i8> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_i8())
            }
        }

        #[inline]
        fn to_i16(&self) -> Option<i16> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_i16())
            }
        }

        #[inline]
        fn to_i32(&self) -> Option<i32> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_i32())
            }
        }

        #[inline]
        fn to_usize(&self) -> Option<usize> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_usize())
            }
        }

        #[inline]
        fn to_u8(&self) -> Option<u8> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_u8())
            }
        }

        #[inline]
        fn to_u16(&self) -> Option<u16> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_u16())
            }
        }

        #[inline]
        fn to_u32(&self) -> Option<u32> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_u32())
            }
        }

        #[inline]
        fn to_f32(&self) -> Option<f32> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_f32())
            }
        }

        #[inline]
        fn to_f64(&self) -> Option<f64> {
            if N::to_usize() == 0 {
                None
            } else {
                self.first().and_then(|x| x.to_f64())
            }
        }
    }

    impl<T, N: ArrayLength<T>> NumCast for NumericArray<T, N>
    where
        T: NumCast + Clone,
    {
        fn from<P: ToPrimitive>(n: P) -> Option<Self> {
            T::from(n).map(|t| Self::from_element(t))
        }
    }

    impl<T, N: ArrayLength<T>> Float for NumericArray<T, N>
    where
        T: Float + Copy,
        Self: Copy,
    {
        #[inline]
        fn nan() -> Self {
            Self::from_element(Float::nan())
        }

        #[inline]
        fn infinity() -> Self {
            Self::from_element(Float::infinity())
        }

        #[inline]
        fn neg_infinity() -> Self {
            Self::from_element(Float::neg_infinity())
        }

        #[inline]
        fn neg_zero() -> Self {
            Self::from_element(Float::neg_zero())
        }

        #[inline]
        fn min_value() -> Self {
            Self::from_element(Float::min_value())
        }

        #[inline]
        fn min_positive_value() -> Self {
            Self::from_element(Float::min_positive_value())
        }

        #[inline]
        fn max_value() -> Self {
            Self::from_element(Float::max_value())
        }

        fn is_nan(self) -> bool {
            self.iter().any(|x| Float::is_nan(*x))
        }

        fn is_infinite(self) -> bool {
            self.iter().any(|x| Float::is_infinite(*x))
        }

        fn is_finite(self) -> bool {
            self.iter().all(|x| Float::is_finite(*x))
        }

        fn is_normal(self) -> bool {
            self.iter().all(|x| Float::is_normal(*x))
        }

        fn classify(self) -> FpCategory {
            let mut ret = FpCategory::Zero;

            for x in self.iter() {
                match Float::classify(*x) {
                    // If NaN is found, return NaN immediately
                    FpCategory::Nan => return FpCategory::Nan,
                    // If infinite, set infinite
                    FpCategory::Infinite => ret = FpCategory::Infinite,
                    // If Subnormal and not infinite, set subnormal
                    FpCategory::Subnormal if ret != FpCategory::Infinite => {
                        ret = FpCategory::Subnormal;
                    }
                    // If normal and zero, upgrade to normal
                    FpCategory::Normal if ret == FpCategory::Zero => {
                        ret = FpCategory::Normal;
                    }
                    _ => {}
                }
            }

            ret
        }

        fn floor(self) -> Self {
            self.0.map(Float::floor).into()
        }

        fn ceil(self) -> Self {
            self.0.map(Float::ceil).into()
        }

        fn round(self) -> Self {
            self.0.map(Float::round).into()
        }

        fn trunc(self) -> Self {
            self.0.map(Float::trunc).into()
        }

        fn fract(self) -> Self {
            self.0.map(Float::fract).into()
        }

        fn abs(self) -> Self {
            self.0.map(Float::abs).into()
        }

        fn signum(self) -> Self {
            self.0.map(Float::signum).into()
        }

        fn is_sign_positive(self) -> bool {
            self.iter().all(|x| Float::is_sign_positive(*x))
        }

        fn is_sign_negative(self) -> bool {
            self.iter().any(|x| Float::is_sign_negative(*x))
        }

        fn mul_add(self, a: Self, b: Self) -> Self {
            let mut left = ArrayConsumer::new(self.0);
            let mut a_arr = ArrayConsumer::new(a.0);
            let mut b_arr = ArrayConsumer::new(b.0);

            let mut destination = ArrayBuilder::new();

            for (dst, (l, (a, b))) in destination.array.iter_mut().zip(
                left.array
                    .iter()
                    .zip(a_arr.array.iter().zip(b_arr.array.iter())),
            ) {
                unsafe {
                    ptr::write(
                        dst,
                        Float::mul_add(ptr::read(l), ptr::read(a), ptr::read(b)),
                    )
                }

                left.position += 1;
                a_arr.position += 1;
                b_arr.position += 1;
                destination.position += 1;
            }

            NumericArray::new(destination.into_inner())
        }

        fn recip(self) -> Self {
            self.0.map(Float::recip).into()
        }

        fn powi(self, n: i32) -> Self {
            self.0.map(|x| Float::powi(x, n)).into()
        }

        fn powf(self, n: Self) -> Self {
            self.0.zip(n.0, Float::powf).into()
        }

        fn sqrt(self) -> Self {
            self.0.map(Float::sqrt).into()
        }

        fn exp(self) -> Self {
            self.0.map(Float::exp).into()
        }

        fn exp2(self) -> Self {
            self.0.map(Float::exp2).into()
        }

        fn ln(self) -> Self {
            self.0.map(Float::ln).into()
        }

        fn log(self, base: Self) -> Self {
            self.0.zip(base.0, Float::log).into()
        }

        fn log2(self) -> Self {
            self.0.map(Float::log2).into()
        }

        fn log10(self) -> Self {
            self.0.map(Float::log10).into()
        }

        fn max(self, other: Self) -> Self {
            self.0.zip(other.0, Float::max).into()
        }

        fn min(self, other: Self) -> Self {
            self.0.zip(other.0, Float::min).into()
        }

        fn abs_sub(self, other: Self) -> Self {
            self.0.zip(other.0, Float::abs_sub).into()
        }

        fn cbrt(self) -> Self {
            self.0.map(Float::cbrt).into()
        }

        fn hypot(self, other: Self) -> Self {
            self.0.zip(other.0, Float::hypot).into()
        }

        fn sin(self) -> Self {
            self.0.map(Float::sin).into()
        }

        fn cos(self) -> Self {
            self.0.map(Float::cos).into()
        }

        fn tan(self) -> Self {
            self.0.map(Float::tan).into()
        }

        fn asin(self) -> Self {
            self.0.map(Float::asin).into()
        }

        fn acos(self) -> Self {
            self.0.map(Float::acos).into()
        }

        fn atan(self) -> Self {
            self.0.map(Float::atan).into()
        }

        fn atan2(self, other: Self) -> Self {
            self.0.zip(other.0, Float::atan2).into()
        }

        fn sin_cos(self) -> (Self, Self) {
            let mut source = ArrayConsumer::new(self.0);
            let mut sin_destination = ArrayBuilder::new();
            let mut cos_destination = ArrayBuilder::new();

            for ((sin, cos), src) in sin_destination
                .array
                .iter_mut()
                .zip(cos_destination.array.iter_mut())
                .zip(source.array.iter())
            {
                unsafe {
                    let (s, c) = Float::sin_cos(ptr::read(src));

                    ptr::write(sin, s);
                    ptr::write(cos, c);
                }

                source.position += 1;
                sin_destination.position += 1;
                cos_destination.position += 1;
            }

            (
                NumericArray::new(sin_destination.into_inner()),
                NumericArray::new(cos_destination.into_inner()),
            )
        }

        fn exp_m1(self) -> Self {
            self.0.map(Float::exp_m1).into()
        }

        fn ln_1p(self) -> Self {
            self.0.map(Float::ln_1p).into()
        }

        fn sinh(self) -> Self {
            self.0.map(Float::sinh).into()
        }

        fn cosh(self) -> Self {
            self.0.map(Float::cosh).into()
        }

        fn tanh(self) -> Self {
            self.0.map(Float::tanh).into()
        }

        fn asinh(self) -> Self {
            self.0.map(Float::asinh).into()
        }

        fn acosh(self) -> Self {
            self.0.map(Float::acosh).into()
        }

        fn atanh(self) -> Self {
            self.0.map(Float::atanh).into()
        }

        fn integer_decode(self) -> (u64, i16, i8) {
            if N::to_usize() == 0 {
                (0, 0, 0)
            } else {
                self.first().unwrap().integer_decode()
            }
        }

        #[inline]
        fn epsilon() -> Self {
            Self::from_element(Float::epsilon())
        }

        fn to_degrees(self) -> Self {
            self.0.map(Float::to_degrees).into()
        }

        fn to_radians(self) -> Self {
            self.0.map(Float::to_radians).into()
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

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
        use std::num::FpCategory;
        use num_traits::Float;

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
}
