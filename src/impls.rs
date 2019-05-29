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

use std::num::FpCategory;
use std::ops::*;

use num_traits::*;

use generic_array::{ArrayBuilder, ArrayConsumer};

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
                    unsafe {
                        let mut right = ArrayConsumer::new(rhs.0);

                        let (right_iter, right_position) = right.iter_position();

                        self.iter_mut().zip(right_iter).for_each(|(lhs, rhs)| {
                            $op_trait::$op(lhs, ptr::read(rhs));

                            *right_position += 1;
                        });
                    }
                }
            }

            impl<'a, T, U: Clone, N: ArrayLength<T> + ArrayLength<U>> $op_trait<&'a NumericArray<U, N>> for NumericArray<T, N>
            where
                T: $op_trait<U>
            {
                fn $op(&mut self, rhs: &'a NumericArray<U, N>) {
                    self.iter_mut().zip(rhs.iter()).for_each(|(lhs, rhs)| {
                        $op_trait::$op(lhs, rhs.clone());
                    });
                }
            }

            impl<T, U: Clone, N: ArrayLength<T>> $op_trait<NumericConstant<U>> for NumericArray<T, N>
            where
                T: $op_trait<U>
            {
                fn $op(&mut self, rhs: NumericConstant<U>) {
                    self.iter_mut().for_each(|lhs| {
                        $op_trait::$op(lhs, rhs.0.clone());
                    });
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
                    unsafe {
                        let mut builder = ArrayBuilder::new();

                        {
                            let (array_iter, position) = builder.iter_position();

                            for (dst, (lhs, rhs)) in array_iter.zip(self.iter().zip(rhs.iter())) {
                                if let Some(value) = $op_trait::$op(lhs, rhs) {
                                    ptr::write(dst, value);

                                    *position += 1;
                                } else {
                                    return None;
                                }
                            }
                        }

                        Some(NumericArray(builder.into_inner()))
                    }
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
        unsafe {
            let mut builder = ArrayBuilder::new();

            {
                let (builder_iter, builder_position) = builder.iter_position();

                for (dst, lhs) in builder_iter.zip(self.iter()) {
                    if let Some(value) = CheckedShl::checked_shl(lhs, rhs) {
                        ptr::write(dst, value);

                        *builder_position += 1;
                    } else {
                        return None;
                    }
                }
            }

            Some(NumericArray(builder.into_inner()))
        }
    }
}

impl<T, N: ArrayLength<T>> CheckedShr for NumericArray<T, N>
where
    T: CheckedShr,
    Self: Shr<u32, Output = Self>,
{
    fn checked_shr(&self, rhs: u32) -> Option<Self> {
        unsafe {
            let mut builder = ArrayBuilder::new();

            {
                let (builder_iter, builder_position) = builder.iter_position();

                for (dst, lhs) in builder_iter.zip(self.iter()) {
                    if let Some(value) = CheckedShr::checked_shr(lhs, rhs) {
                        ptr::write(dst, value);

                        *builder_position += 1;
                    } else {
                        return None;
                    }
                }
            }

            Some(NumericArray(builder.into_inner()))
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
        <T as Num>::from_str_radix(str, radix).map(Self::splat)
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

impl<T: Clone, N: ArrayLength<T>> Unsigned for NumericArray<T, N> where T: Unsigned {}

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

macro_rules! impl_to_primitive {
    ($($to:ident => $prim:ty),*) => {
        impl<T, N: ArrayLength<T>> ToPrimitive for NumericArray<T, N>
        where
            T: ToPrimitive,
        {
            $(
                #[inline]
                fn $to(&self) -> Option<$prim> {
                    if N::to_usize() == 0 { None } else {
                        self.first().and_then(ToPrimitive::$to)
                    }
                }
            )*
        }
    }
}

impl_to_primitive! {
    to_i8       => i8,
    to_i16      => i16,
    to_i32      => i32,
    to_i64      => i64,
    to_i128     => i128,
    to_isize    => isize,

    to_u8       => u8,
    to_u16      => u16,
    to_u32      => u32,
    to_u64      => u64,
    to_u128     => u128,
    to_usize    => usize,

    to_f32      => f32,
    to_f64      => f64
}

impl<T, N: ArrayLength<T>> NumCast for NumericArray<T, N>
where
    T: NumCast + Clone,
{
    fn from<P: ToPrimitive>(n: P) -> Option<Self> {
        T::from(n).map(Self::splat)
    }
}

impl<T, N: ArrayLength<T>> Float for NumericArray<T, N>
where
    T: Float + Copy,
    Self: Copy,
{
    #[inline]
    fn nan() -> Self {
        Self::splat(Float::nan())
    }

    #[inline]
    fn infinity() -> Self {
        Self::splat(Float::infinity())
    }

    #[inline]
    fn neg_infinity() -> Self {
        Self::splat(Float::neg_infinity())
    }

    #[inline]
    fn neg_zero() -> Self {
        Self::splat(Float::neg_zero())
    }

    #[inline]
    fn min_value() -> Self {
        Self::splat(Float::min_value())
    }

    #[inline]
    fn min_positive_value() -> Self {
        Self::splat(Float::min_positive_value())
    }

    #[inline]
    fn max_value() -> Self {
        Self::splat(Float::max_value())
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
        unsafe {
            let mut left = ArrayConsumer::new(self.0);
            let mut a_arr = ArrayConsumer::new(a.0);
            let mut b_arr = ArrayConsumer::new(b.0);

            let (left_iter, left_position) = left.iter_position();
            let (a_arr_iter, a_arr_position) = a_arr.iter_position();
            let (b_arr_iter, b_arr_position) = b_arr.iter_position();

            let mut destination = ArrayBuilder::new();

            {
                let (destination_iter, destination_position) = destination.iter_position();

                destination_iter
                    .zip(left_iter.zip(a_arr_iter.zip(b_arr_iter)))
                    .for_each(|(dst, (l, (a, b)))| {
                        let l = ptr::read(l);
                        let a = ptr::read(a);
                        let b = ptr::read(b);

                        *left_position += 1;
                        *a_arr_position += 1;
                        *b_arr_position += 1;

                        ptr::write(dst, Float::mul_add(l, a, b));

                        *destination_position += 1;
                    });
            }

            NumericArray::new(destination.into_inner())
        }
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
        unsafe {
            let mut source = ArrayConsumer::new(self.0);

            let (source_iter, source_position) = source.iter_position();

            let mut sin_destination = ArrayBuilder::new();
            let mut cos_destination = ArrayBuilder::new();

            {
                let (sin_destination_iter, sin_destination_position) = sin_destination.iter_position();
                let (cos_destination_iter, cos_destination_position) = cos_destination.iter_position();

                sin_destination_iter
                    .zip(cos_destination_iter)
                    .zip(source_iter)
                    .for_each(|((sin, cos), src)| {
                        let x = ptr::read(src);

                        *source_position += 1;

                        let (s, c) = Float::sin_cos(x);

                        ptr::write(sin, s);
                        ptr::write(cos, c);

                        *sin_destination_position += 1;
                        *cos_destination_position += 1;
                    });
            }

            (
                NumericArray::new(sin_destination.into_inner()),
                NumericArray::new(cos_destination.into_inner()),
            )
        }
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
        Self::splat(Float::epsilon())
    }

    fn to_degrees(self) -> Self {
        self.0.map(Float::to_degrees).into()
    }

    fn to_radians(self) -> Self {
        self.0.map(Float::to_radians).into()
    }
}
