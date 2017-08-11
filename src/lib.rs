//! `numeric-array` is a wrapper around [`generic-array`](https://github.com/fizyk20/generic-array) that adds efficient numeric trait implementations, often times making use of SIMD instructions and compile-time evaluations.
//!
//! All stable `std::ops` traits are implemented for `NumericArray` itself, plus the thin `NumericConstant` type, which is required to differeniate constant values from `NumericArray` itself.
//!
//! Additionally, most of `num_traits` are implemented, including `Num` itself. So you can even use a whole array as a generic number.
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
//! When used with `RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"`, then Rust and LLVM are smart enough to turn almost all operations into SIMD instructions, or even just evaluate them at compile time. The above example is actually evaluated at compile time, so if you were to view the assembly it would show the result only. Rust is pretty smart.
//!
//! Therefore, this is ideal for situations where simple component-wise operations are required for arrays.
//!
#![deny(missing_docs)]

extern crate num_traits;
extern crate typenum;

#[cfg_attr(test, macro_use)]
extern crate generic_array;
extern crate nodrop;

use std::{mem, ptr, slice};

use std::borrow::{Borrow, BorrowMut};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use std::ops::{Add, Sub};

use std::fmt::{Debug, Formatter, Result as FmtResult};

use nodrop::NoDrop;
use typenum::*;
use generic_array::{ArrayLength, GenericArray, GenericArrayIter};

/// Defines any `NumericArray` type as its element and length types
///
/// This is useful for situations where the element type and length can be variable,
/// but you need a generic type bound and don't want to deal with
/// `T` and `N: ArrayLength<T>` from `NumericArray<T, N>` directly.
pub trait NumericSequence<T> {
    /// Array length type
    type Length: ArrayLength<T>;
}

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

impl<T, N: ArrayLength<T>> NumericSequence<T> for NumericArray<T, N> {
    type Length = N;
}

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

/// This is required to allow `NumericArray` to be operated on by both other `NumericArray`
/// instances and constants, with generic types, because some type `U` supercedes `NumericArray<U, N>`
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
        self.map(|x| x.clone())
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
        &self.0
    }
}

impl<T, N: ArrayLength<T>> DerefMut for NumericArray<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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

impl<T, U, N: ArrayLength<T> + ArrayLength<U>> PartialEq<GenericArray<U, N>> for NumericArray<T, N> where T: PartialEq<U> {
    fn eq(&self, rhs: &GenericArray<U, N>) -> bool {
        **self == **rhs
    }

    fn ne(&self, rhs: &GenericArray<U, N>) -> bool {
        **self == **rhs
    }
}

impl<T, N: ArrayLength<T>> ::std::cmp::Eq for NumericArray<T, N>
where
    T: ::std::cmp::Eq,
{
}

impl<T, N: ArrayLength<T>> PartialOrd<Self> for NumericArray<T, N>
where
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, rhs: &Self) -> Option<::std::cmp::Ordering> {
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
    fn partial_cmp(&self, rhs: &GenericArray<T, N>) -> Option<::std::cmp::Ordering> {
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

impl<T, N: ArrayLength<T>> ::std::cmp::Ord for NumericArray<T, N>
where
    T: ::std::cmp::Ord,
{
    #[inline]
    fn cmp(&self, rhs: &Self) -> ::std::cmp::Ordering {
        ::std::cmp::Ord::cmp(&self.0, &rhs.0)
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
        Self::generate(|| t.clone())
    }

    /// Generates a new array using the given generator function to create each value.
    ///
    /// The generator function is called `N` times, once for each element.
    pub fn generate<G>(g: G) -> NumericArray<T, N>
    where
        G: Fn() -> T,
    {
        let mut res: NoDrop<GenericArray<T, N>> = NoDrop::new(unsafe { mem::uninitialized() });

        for dst in res.iter_mut() {
            unsafe {
                ptr::write(dst, g());
            }
        }

        NumericArray(res.into_inner())
    }

    /// Moves all but the last element into a `NumericArray` with one
    /// less element than the current one.
    ///
    /// The last element is dropped.
    ///
    /// Example:
    ///
    /// ```ignore
    /// let a = NumericArray::new(arr![i32; 1, 2, 3, 4]);
    /// let b = NumericArray::new(arr![i32; 1, 2, 3]);
    ///
    /// assert_eq!(a.shorten(), b);
    /// ```
    pub fn shorten(self) -> NumericArray<T, Sub1<N>>
    where
        N: Sub<B1>,
        Sub1<N>: ArrayLength<T>,
    {
        use std::{mem, ptr};

        let mut shorter: GenericArray<T, Sub1<N>> = unsafe { mem::uninitialized() };

        for (dst, src) in shorter.iter_mut().zip(self.iter()) {
            unsafe {
                ptr::write(dst, ptr::read(src));
            }
        }

        let _last = unsafe { ptr::read(&self.0[N::to_usize() - 1]) };

        mem::forget(self);

        NumericArray(shorter)
    }

    /// Moves all the current elements into a new array with one more element than the current one.
    ///
    /// The last element of the new array is set to `last`
    ///
    /// Example:
    ///
    /// ```ignore
    /// let a = NumericArray::new(arr![i32; 1, 2, 3, 4]);
    /// let b = NumericArray::new(arr![i32; 1, 2, 3]);
    ///
    /// assert_eq!(a, b.lengthen(4));
    /// ```
    pub fn lengthen(self, last: T) -> NumericArray<T, Add1<N>>
    where
        N: Add<B1>,
        Add1<N>: ArrayLength<T>,
    {
        use std::{mem, ptr};

        let mut longer: GenericArray<T, Add1<N>> = unsafe { mem::uninitialized() };

        for (dst, src) in longer.iter_mut().zip(self.iter()) {
            unsafe {
                ptr::write(dst, ptr::read(src));
            }
        }

        unsafe {
            ptr::write(&mut longer[N::to_usize()], last);
        }

        mem::forget(self);

        NumericArray(longer)
    }

    /// Maps the current array to a new one of the same size using the given function.
    pub fn map<U, F>(&self, f: F) -> NumericArray<U, N>
    where
        N: ArrayLength<U>,
        F: Fn(&T) -> U,
    {
        let mut res: NoDrop<GenericArray<U, N>> = NoDrop::new(unsafe { mem::uninitialized() });

        for (dst, src) in res.iter_mut().zip(self.iter()) {
            unsafe {
                ptr::write(dst, f(src));
            }
        }

        NumericArray(res.into_inner())
    }

    /// Same as `map`, but the values are moved rather than referenced.
    pub fn map_move<U, F>(self, f: F) -> NumericArray<U, N>
    where
        N: ArrayLength<U>,
        F: Fn(T) -> U,
    {
        let mut res: NoDrop<GenericArray<U, N>> = NoDrop::new(unsafe { mem::uninitialized() });

        for (dst, src) in res.iter_mut().zip(self.iter()) {
            unsafe {
                ptr::write(dst, f(ptr::read(src)));
            }
        }

        mem::forget(self);

        NumericArray(res.into_inner())
    }

    /// Combines two same-length arrays and maps both values to a new array using the given function.
    pub fn zip<V, U, F>(&self, rhs: &NumericArray<V, N>, f: F) -> NumericArray<U, N>
    where
        N: ArrayLength<V> + ArrayLength<U>,
        F: Fn(&T, &V) -> U,
    {
        let mut res: NoDrop<GenericArray<U, N>> = NoDrop::new(unsafe { mem::uninitialized() });

        for (dst, (lhs, rhs)) in res.iter_mut().zip(self.iter().zip(rhs.iter())) {
            unsafe {
                ptr::write(dst, f(lhs, rhs));
            }
        }

        NumericArray(res.into_inner())
    }

    /// Same as `zip`, but `self` values are moved. The `rhs` array is still accessed by reference.
    pub fn zip_move<V, U, F>(self, rhs: &NumericArray<V, N>, f: F) -> NumericArray<U, N>
    where
        N: ArrayLength<V> + ArrayLength<U>,
        F: Fn(T, &V) -> U,
    {
        let mut res: NoDrop<GenericArray<U, N>> = NoDrop::new(unsafe { mem::uninitialized() });

        for (dst, (lhs, rhs)) in res.iter_mut().zip(self.iter().zip(rhs.iter())) {
            unsafe {
                ptr::write(dst, f(ptr::read(lhs), rhs));
            }
        }

        mem::forget(self);

        NumericArray(res.into_inner())
    }

    /// Like `zip` and `zip_move`, but moves both `self` and the `rhs` array.
    pub fn zip_move_both<V, U, F>(self, rhs: NumericArray<V, N>, f: F) -> NumericArray<U, N>
    where
        N: ArrayLength<V> + ArrayLength<U>,
        F: Fn(T, V) -> U,
    {
        let mut res: NoDrop<GenericArray<U, N>> = NoDrop::new(unsafe { mem::uninitialized() });

        for (dst, (lhs, rhs)) in res.iter_mut().zip(self.iter().zip(rhs.iter())) {
            unsafe {
                ptr::write(dst, f(ptr::read(lhs), ptr::read(rhs)));
            }
        }

        mem::forget(self);
        mem::forget(rhs);

        NumericArray(res.into_inner())
    }

    /// Convert one `NumericArray` to another using `From` for each element.
    pub fn convert<U>(self) -> NumericArray<U, N>
    where
        N: ArrayLength<U>,
        U: From<T>,
    {
        self.map_move(From::from)
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

impl<T, N: ArrayLength<T>> Default for NumericArray<T, N>
where
    T: Default,
{
    fn default() -> Self {
        Self::generate(|| Default::default())
    }
}

impl<T, N: ArrayLength<T>> num_traits::Zero for NumericArray<T, N>
where
    T: num_traits::Zero,
{
    fn zero() -> Self {
        Self::generate(|| <T as num_traits::Zero>::zero())
    }

    fn is_zero(&self) -> bool {
        self.iter().all(|x| x.is_zero())
    }
}

impl<T, N: ArrayLength<T>> num_traits::One for NumericArray<T, N>
where
    T: num_traits::One,
{
    fn one() -> Self {
        Self::generate(|| <T as num_traits::One>::one())
    }
}

macro_rules! impl_unary_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, N: ArrayLength<T>> ::std::ops::$op_trait for NumericArray<T, N>
            where
                T: ::std::ops::$op_trait,
                N: ArrayLength<<T as ::std::ops::$op_trait>::Output>
            {
                type Output = NumericArray<<T as ::std::ops::$op_trait>::Output, N>;

                fn $op(self) -> Self::Output {
                    self.map_move(::std::ops::$op_trait::$op)
                }
            }
        )*
    }
}

macro_rules! impl_binary_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, U, N: ArrayLength<T> + ArrayLength<U>> ::std::ops::$op_trait<NumericArray<U, N>> for NumericArray<T, N>
            where
                T: ::std::ops::$op_trait<U>,
                N: ArrayLength<<T as ::std::ops::$op_trait<U>>::Output>
            {
                type Output = NumericArray<<T as ::std::ops::$op_trait<U>>::Output, N>;

                fn $op(self, rhs: NumericArray<U, N>) -> Self::Output {
                    self.zip_move_both(rhs, ::std::ops::$op_trait::$op)
                }
            }

            impl<T, U: Clone, N: ArrayLength<T>> ::std::ops::$op_trait<NumericConstant<U>> for NumericArray<T, N>
            where
                T: ::std::ops::$op_trait<U>,
                N: ArrayLength<<T as ::std::ops::$op_trait<U>>::Output>
            {
                type Output = NumericArray<<T as ::std::ops::$op_trait<U>>::Output, N>;

                fn $op(self, rhs: NumericConstant<U>) -> Self::Output {
                    self.map_move(|l| ::std::ops::$op_trait::$op(l, rhs.0.clone()))
                }
            }
        )*
    }
}

macro_rules! impl_assign_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, U, N: ArrayLength<T> + ArrayLength<U>> ::std::ops::$op_trait<NumericArray<U, N>> for NumericArray<T, N>
            where
                T: ::std::ops::$op_trait<U>
            {
                fn $op(&mut self, rhs: NumericArray<U, N>) {
                    for (lhs, rhs) in self.iter_mut().zip(rhs.iter()) {
                        ::std::ops::$op_trait::$op(lhs, unsafe { ptr::read(rhs) });
                    }

                    mem::forget(rhs);
                }
            }

            impl<T, U: Clone, N: ArrayLength<T>> ::std::ops::$op_trait<NumericConstant<U>> for NumericArray<T, N>
            where
                T: ::std::ops::$op_trait<U>
            {
                fn $op(&mut self, rhs: NumericConstant<U>) {
                    for lhs in self.iter_mut() {
                        ::std::ops::$op_trait::$op(lhs, rhs.0.clone());
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
                    self.zip(rhs, num_traits::$op_trait::$op)
                }
            }
        )*
    }
}

macro_rules! impl_checked_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, N: ArrayLength<T>> num_traits::$op_trait for NumericArray<T, N>
            where
                T: num_traits::$op_trait
            {
                fn $op(&self, rhs: &Self) -> Option<Self> {
                    let mut res: NoDrop<GenericArray<T, N>> = NoDrop::new(unsafe { mem::uninitialized() });

                    for (dst, (lhs, rhs)) in res.iter_mut().zip(self.iter().zip(rhs.iter())) {
                        if let Some(value) = num_traits::$op_trait::$op(lhs, rhs) {
                            unsafe {
                                ptr::write(dst, value);
                            }
                        } else {
                            return None;
                        }
                    }

                    Some(NumericArray(res.into_inner()))
                }
            }
        )*
    }
}

impl_unary_ops!(Neg::neg, Not::not);

impl_binary_ops! {
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

impl<T, N: ArrayLength<T>> num_traits::Saturating for NumericArray<T, N>
where
    T: num_traits::Saturating,
{
    fn saturating_add(self, rhs: Self) -> Self {
        self.zip_move_both(rhs, num_traits::Saturating::saturating_add)
    }

    fn saturating_sub(self, rhs: Self) -> Self {
        self.zip_move_both(rhs, num_traits::Saturating::saturating_sub)
    }
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

impl<T: Clone, N: ArrayLength<T>> num_traits::Num for NumericArray<T, N>
where
    T: num_traits::Num,
{
    type FromStrRadixErr = <T as num_traits::Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        <T as num_traits::Num>::from_str_radix(str, radix).map(Self::from_element)
    }
}

impl<T: Clone, N: ArrayLength<T>> num_traits::Signed for NumericArray<T, N>
where
    T: num_traits::Signed,
{
    fn abs(&self) -> Self {
        self.map(num_traits::Signed::abs)
    }

    fn abs_sub(&self, rhs: &Self) -> Self {
        self.zip(rhs, num_traits::Signed::abs_sub)
    }

    fn signum(&self) -> Self {
        self.map(num_traits::Signed::signum)
    }

    fn is_positive(&self) -> bool {
        self.iter().all(num_traits::Signed::is_positive)
    }

    fn is_negative(&self) -> bool {
        self.iter().any(num_traits::Signed::is_negative)
    }
}

impl<T: Clone, N: ArrayLength<T>> num_traits::Unsigned for NumericArray<T, N>
where
    T: num_traits::Unsigned,
{
}

macro_rules! impl_float_const {
    ($($f:ident),*) => {
        impl<T, N: ArrayLength<T>> num_traits::FloatConst for NumericArray<T, N>
        where
            T: num_traits::FloatConst
        {
            $(
                fn $f() -> Self {
                    Self::generate(|| <T as num_traits::FloatConst>::$f())
                }
            )*
        }
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

impl<T, N: ArrayLength<T>> num_traits::Bounded for NumericArray<T, N>
where
    T: num_traits::Bounded,
{
    fn min_value() -> Self {
        Self::generate(|| <T as num_traits::Bounded>::min_value())
    }

    fn max_value() -> Self {
        Self::generate(|| <T as num_traits::Bounded>::max_value())
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
    fn shorten() {
        let a = NumericArray::new(arr![i32; 1, 2, 3, 4]);
        let b = NumericArray::new(arr![i32; 1, 2, 3]);

        assert_eq!(a.shorten(), b);
    }

    #[test]
    fn lengthen() {
        let a = NumericArray::new(arr![i32; 1, 2, 3, 4]);
        let b = NumericArray::new(arr![i32; 1, 2, 3]);

        assert_eq!(a, b.lengthen(4));
    }

    #[test]
    fn ops() {
        let a = black_box(NumericArray::new(arr![i32; 1, 3, 5, 7]));
        let b = black_box(NumericArray::new(arr![i32; 2, 4, 6, 8]));

        let c = a + b;
        let d = c * nconstant!(black_box(5));
        let e = d << nconstant!(1_usize);

        assert_eq!(e, NumericArray::new(arr![i32; 30, 70, 110, 150]))
    }

    #[test]
    fn readme_example() {
        let a = narr![i32; 1, 3, 5, 7];
        let b = narr![i32; 2, 4, 6, 8];

        let c = a + b * nconstant!(2);

        assert_eq!(c, narr![i32; 5, 11, 17, 23]);
    }
}
