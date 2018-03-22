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
//! #[macro_use]
//! extern crate generic_array;
//! #[macro_use]
//! extern crate numeric_array;
//!
//! use numeric_array::NumericArray;
//!
//! fn main() {
//!     let a = narr![i32; 1, 3, 5, 7];
//!     let b = narr![i32; 2, 4, 6, 8];
//!
//!     let c = a + b * nconstant!(2);
//!
//!     assert_eq!(c, narr![i32; 5, 11, 17, 23]);
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
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};

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
        F: FnMut(usize) -> T
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
    ($value:expr) => { $crate::NumericConstant($value) }
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
    T: PartialEq<U>
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
                    for (lhs, rhs) in self.iter_mut().zip(rhs.iter()) {
                        $op_trait::$op(lhs, unsafe { ptr::read(rhs) });
                    }

                    mem::forget(rhs);
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

// std::ops and num_traits impementations in their own module for wildcard imports
mod impls {
    use super::*;

    use generic_array::functional::*;

    use std::ops::*;
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
        Self: Shl<u32, Output=Self>,
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
        Self: Shr<u32, Output=Self>,
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
            NumericArray(GenericArray::generate(
                |_| <T as Bounded>::min_value(),
            ))
        }

        fn max_value() -> Self {
            NumericArray(GenericArray::generate(
                |_| <T as Bounded>::max_value(),
            ))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

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
    fn test_readme_example() {
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
}
