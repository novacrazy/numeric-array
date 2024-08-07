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

use core::mem::{self, ManuallyDrop};
use core::num::FpCategory;
use core::ops::*;

use num_traits::{float::FloatCore, *};

use generic_array::internals::{ArrayConsumer, IntrusiveArrayBuilder};

macro_rules! impl_unary_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, N: ArrayLength> $op_trait for NumericArray<T, N>
            where
                T: $op_trait,
            {
                type Output = NumericArray<<T as $op_trait>::Output, N>;

                #[inline(always)]
                fn $op(self) -> Self::Output {
                    NumericArray(self.0.map($op_trait::$op))
                }
            }

            impl<'a, T: Clone, N: ArrayLength> $op_trait for &'a NumericArray<T, N>
            where
                T: $op_trait,
            {
                type Output = NumericArray<<T as $op_trait>::Output, N>;

                #[inline(always)]
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
            impl<T, U, N: ArrayLength> $op_trait<NumericArray<U, N>> for NumericArray<T, N>
            where
                T: $op_trait<U>,
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                #[inline(always)]
                fn $op(self, rhs: NumericArray<U, N>) -> Self::Output {
                    NumericArray(self.0.zip(rhs.0, $op_trait::$op))
                }
            }

            impl<'a, T, U: Clone, N: ArrayLength> $op_trait<&'a NumericArray<U, N>> for NumericArray<T, N>
            where
                T: $op_trait<U>,
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                #[inline(always)]
                fn $op(self, rhs: &'a NumericArray<U, N>) -> Self::Output {
                    NumericArray(self.0.zip(&rhs.0, |l, r| $op_trait::$op(l, r.clone())))
                }
            }

            impl<'a, T: Clone, U, N: ArrayLength> $op_trait<NumericArray<U, N>> for &'a NumericArray<T, N>
            where
                T: $op_trait<U>,
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                #[inline(always)]
                fn $op(self, rhs: NumericArray<U, N>) -> Self::Output {
                    NumericArray((&self.0).zip(rhs.0, |l, r| $op_trait::$op(l.clone(), r)))
                }
            }

            impl<'a, 'b, T: Clone, U: Clone, N: ArrayLength> $op_trait<&'b NumericArray<U, N>> for &'a NumericArray<T, N>
            where
                T: $op_trait<U>,
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                #[inline(always)]
                fn $op(self, rhs: &'b NumericArray<U, N>) -> Self::Output {
                    NumericArray((&self.0).zip(&rhs.0, |l, r| $op_trait::$op(l.clone(), r.clone())))
                }
            }

            impl<T, U: Clone, N: ArrayLength> $op_trait<NumericConstant<U>> for NumericArray<T, N>
            where
                T: $op_trait<U>,
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                #[inline(always)]
                fn $op(self, rhs: NumericConstant<U>) -> Self::Output {
                    NumericArray(self.0.map(|l| $op_trait::$op(l, rhs.0.clone())))
                }
            }

            impl<'a, T: Clone, U: Clone, N: ArrayLength> $op_trait<NumericConstant<U>> for &'a NumericArray<T, N>
            where
                T: $op_trait<U>,
            {
                type Output = NumericArray<<T as $op_trait<U>>::Output, N>;

                #[inline(always)]
                fn $op(self, rhs: NumericConstant<U>) -> Self::Output {
                    NumericArray((&self.0).map(|l| $op_trait::$op(l.clone(), rhs.0.clone())))
                }
            }

            impl<T, U: Clone, N: ArrayLength> $op_trait<NumericArray<T, N>> for NumericConstant<U>
            where
                U: $op_trait<T>,
            {
                type Output = NumericArray<<U as $op_trait<T>>::Output, N>;

                #[inline(always)]
                fn $op(self, rhs: NumericArray<T, N>) -> Self::Output {
                    NumericArray(rhs.0.map(|r| $op_trait::$op(self.0.clone(), r)))
                }
            }

            impl<'a, T: Clone, U: Clone, N: ArrayLength> $op_trait<&'a NumericArray<T, N>> for NumericConstant<U>
            where
                U: $op_trait<T>,
            {
                type Output = NumericArray<<U as $op_trait<T>>::Output, N>;

                #[inline(always)]
                fn $op(self, rhs: &'a NumericArray<T, N>) -> Self::Output {
                    NumericArray((&rhs.0).map(|r| $op_trait::$op(self.0.clone(), r.clone())))
                }
            }
        )*
    }
}

macro_rules! impl_assign_ops {
    ($($op_trait:ident::$op:ident),*) => {
        $(
            impl<T, U, N: ArrayLength> $op_trait<NumericArray<U, N>> for NumericArray<T, N>
            where
                T: $op_trait<U>,
            {
                fn $op(&mut self, rhs: NumericArray<U, N>) {
                    if mem::needs_drop::<U>() {
                        unsafe {
                            let mut right = ArrayConsumer::new(rhs.0);

                            let (right_iter, right_position) = right.iter_position();

                            self.iter_mut().zip(right_iter).for_each(|(lhs, rhs)| {
                                $op_trait::$op(lhs, ptr::read(rhs));

                                *right_position += 1;
                            });
                        }
                    } else {
                        let right = ManuallyDrop::new(rhs);

                        self.iter_mut().zip(right.iter()).for_each(|(lhs, rhs)| unsafe {
                            $op_trait::$op(lhs, ptr::read(rhs));
                        });
                    }
                }
            }

            impl<'a, T, U: Clone, N: ArrayLength> $op_trait<&'a NumericArray<U, N>> for NumericArray<T, N>
            where
                T: $op_trait<U>,
            {
                fn $op(&mut self, rhs: &'a NumericArray<U, N>) {
                    self.iter_mut().zip(rhs.iter()).for_each(|(lhs, rhs)| {
                        $op_trait::$op(lhs, rhs.clone());
                    });
                }
            }

            impl<T, U: Clone, N: ArrayLength> $op_trait<NumericConstant<U>> for NumericArray<T, N>
            where
                T: $op_trait<U>,
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
            impl<T, N: ArrayLength> num_traits::$op_trait for NumericArray<T, N>
            where
                T: num_traits::$op_trait,
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
            impl<T, N: ArrayLength> $op_trait for NumericArray<T, N>
            where
                T: $op_trait,
            {
                fn $op(&self, rhs: &Self) -> Option<Self> {
                    unsafe {
                        let mut array = GenericArray::uninit();
                        let mut builder = IntrusiveArrayBuilder::new(&mut array);

                        {
                            let (array_iter, position) = builder.iter_position();

                            for (dst, (lhs, rhs)) in array_iter.zip(self.iter().zip(rhs.iter())) {
                                if let Some(value) = $op_trait::$op(lhs, rhs) {
                                    dst.write(value);
                                    *position += 1;
                                } else {
                                    return None;
                                }
                            }
                        }

                        Some(NumericArray({
                            builder.finish();
                            IntrusiveArrayBuilder::array_assume_init(array)
                        }))
                    }
                }
            }
        )*
    }
}

macro_rules! impl_float_const {
    ($($f:ident),*) => {
        impl<T, N: ArrayLength> FloatConst for NumericArray<T, N>
        where
            T: FloatConst,
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

impl<T, N: ArrayLength> CheckedShl for NumericArray<T, N>
where
    T: CheckedShl,
    Self: Shl<u32, Output = Self>,
{
    fn checked_shl(&self, rhs: u32) -> Option<Self> {
        unsafe {
            let mut array = GenericArray::uninit();
            let mut builder = IntrusiveArrayBuilder::new(&mut array);

            {
                let (builder_iter, builder_position) = builder.iter_position();

                for (dst, lhs) in builder_iter.zip(self.iter()) {
                    if let Some(value) = CheckedShl::checked_shl(lhs, rhs) {
                        dst.write(value);
                        *builder_position += 1;
                    } else {
                        return None;
                    }
                }
            }

            Some(NumericArray({
                builder.finish();
                IntrusiveArrayBuilder::array_assume_init(array)
            }))
        }
    }
}

impl<T, N: ArrayLength> CheckedShr for NumericArray<T, N>
where
    T: CheckedShr,
    Self: Shr<u32, Output = Self>,
{
    fn checked_shr(&self, rhs: u32) -> Option<Self> {
        unsafe {
            let mut array = GenericArray::uninit();
            let mut builder = IntrusiveArrayBuilder::new(&mut array);

            {
                let (builder_iter, builder_position) = builder.iter_position();

                for (dst, lhs) in builder_iter.zip(self.iter()) {
                    if let Some(value) = CheckedShr::checked_shr(lhs, rhs) {
                        dst.write(value);
                        *builder_position += 1;
                    } else {
                        return None;
                    }
                }
            }

            Some(NumericArray({
                builder.finish();
                IntrusiveArrayBuilder::array_assume_init(array)
            }))
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

impl<T, N: ArrayLength> Zero for NumericArray<T, N>
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

impl<T, N: ArrayLength> One for NumericArray<T, N>
where
    T: One,
{
    fn one() -> Self {
        NumericArray(GenericArray::generate(|_| <T as One>::one()))
    }
}

impl<T, N: ArrayLength> Saturating for NumericArray<T, N>
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

impl<T: Clone, N: ArrayLength> Num for NumericArray<T, N>
where
    T: Num,
{
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        <T as Num>::from_str_radix(str, radix).map(Self::splat)
    }
}

impl<T: Clone, N: ArrayLength> Signed for NumericArray<T, N>
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

impl<T: Clone, N: ArrayLength> Unsigned for NumericArray<T, N> where T: Unsigned {}

impl<T, N: ArrayLength> Bounded for NumericArray<T, N>
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
        impl<T, N: ArrayLength> ToPrimitive for NumericArray<T, N>
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

impl<T, N: ArrayLength> NumCast for NumericArray<T, N>
where
    T: NumCast + Clone,
{
    fn from<P: ToPrimitive>(n: P) -> Option<Self> {
        T::from(n).map(Self::splat)
    }
}

macro_rules! impl_float {
    ($float:ident { $($extra:tt)* }) => {
        impl<T, N: ArrayLength> $float for NumericArray<T, N>
        where
            T: $float,
            Self: Copy,
        {
            $($extra)*

            #[inline]
            fn nan() -> Self {
                Self::splat($float::nan())
            }

            #[inline]
            fn infinity() -> Self {
                Self::splat($float::infinity())
            }

            #[inline]
            fn neg_infinity() -> Self {
                Self::splat($float::neg_infinity())
            }

            #[inline]
            fn neg_zero() -> Self {
                Self::splat($float::neg_zero())
            }

            #[inline]
            fn min_value() -> Self {
                Self::splat($float::min_value())
            }

            #[inline]
            fn min_positive_value() -> Self {
                Self::splat($float::min_positive_value())
            }

            #[inline]
            fn max_value() -> Self {
                Self::splat($float::max_value())
            }

            #[inline]
            fn is_nan(self) -> bool {
                self.iter().any(|x| $float::is_nan(*x))
            }

            #[inline]
            fn is_infinite(self) -> bool {
                self.iter().any(|x| $float::is_infinite(*x))
            }

            #[inline]
            fn is_finite(self) -> bool {
                self.iter().all(|x| $float::is_finite(*x))
            }

            #[inline]
            fn is_normal(self) -> bool {
                self.iter().all(|x| $float::is_normal(*x))
            }

            #[inline]
            fn epsilon() -> Self {
                Self::splat($float::epsilon())
            }

            #[inline]
            fn to_degrees(self) -> Self {
                self.0.map($float::to_degrees).into()
            }

            #[inline]
            fn to_radians(self) -> Self {
                self.0.map($float::to_radians).into()
            }

            #[inline]
            fn integer_decode(self) -> (u64, i16, i8) {
                if N::to_usize() == 0 {
                    (0, 0, 0)
                } else {
                    self.first().unwrap().integer_decode()
                }
            }

            fn classify(self) -> FpCategory {
                let mut ret = FpCategory::Zero;

                for x in self.iter() {
                    match $float::classify(*x) {
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

            #[inline]
            fn floor(self) -> Self {
                self.0.map($float::floor).into()
            }

            #[inline]
            fn ceil(self) -> Self {
                self.0.map($float::ceil).into()
            }

            #[inline]
            fn round(self) -> Self {
                self.0.map($float::round).into()
            }

            #[inline]
            fn trunc(self) -> Self {
                self.0.map($float::trunc).into()
            }

            #[inline]
            fn fract(self) -> Self {
                self.0.map($float::fract).into()
            }

            #[inline]
            fn abs(self) -> Self {
                self.0.map($float::abs).into()
            }

            #[inline]
            fn signum(self) -> Self {
                self.0.map($float::signum).into()
            }

            #[inline]
            fn is_sign_positive(self) -> bool {
                self.iter().all(|x| $float::is_sign_positive(*x))
            }

            #[inline]
            fn is_sign_negative(self) -> bool {
                self.iter().any(|x| $float::is_sign_negative(*x))
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                self.0.zip(other.0, $float::max).into()
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                self.0.zip(other.0, $float::min).into()
            }

            #[inline]
            fn recip(self) -> Self {
                self.0.map($float::recip).into()
            }

            #[inline]
            fn powi(self, n: i32) -> Self {
                self.0.map(|x| $float::powi(x, n)).into()

                // This was a prototype with the best performance that
                // still fell short of whatever the compiler does, sadly.
                //
                // let mut e = n as i64;
                // let mut x = self;
                // let mut res = Self::one();
                // if e < 0 {
                //     x = x.recip();
                //     e = -e;
                // }
                // while e != 0 {
                //     if e & 1 != 0 {
                //         res *= x;
                //     }
                //     e >>= 1;
                //     x *= x;
                // }
                // res
            }
        }
    };
}

impl_float!(FloatCore {});

#[cfg(feature = "std")]
impl_float!(Float {
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        if mem::needs_drop::<T>() {
            unsafe {
                let mut left = ArrayConsumer::new(self.0);
                let mut a_arr = ArrayConsumer::new(a.0);
                let mut b_arr = ArrayConsumer::new(b.0);

                let (left_iter, left_position) = left.iter_position();
                let (a_arr_iter, a_arr_position) = a_arr.iter_position();
                let (b_arr_iter, b_arr_position) = b_arr.iter_position();

                NumericArray::from_iter(left_iter.zip(a_arr_iter.zip(b_arr_iter)).map(|(l, (a, b))| {
                    let l = ptr::read(l);
                    let a = ptr::read(a);
                    let b = ptr::read(b);

                    *left_position += 1;
                    *a_arr_position = *left_position;
                    *b_arr_position = *left_position;

                    Float::mul_add(l, a, b)
                }))
            }
        } else {
            let left = ManuallyDrop::new(self);
            let a = ManuallyDrop::new(a);
            let b = ManuallyDrop::new(b);

            NumericArray::from_iter(left.iter().zip(a.iter()).zip(b.iter()).map(|((l, a), b)| unsafe {
                Float::mul_add(ptr::read(l), ptr::read(a), ptr::read(b)) //
            }))
        }
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        self.0.zip(n.0, Float::powf).into()
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.0.map(Float::sqrt).into()
    }

    #[inline]
    fn exp(self) -> Self {
        self.0.map(Float::exp).into()
    }

    #[inline]
    fn exp2(self) -> Self {
        self.0.map(Float::exp2).into()
    }

    #[inline]
    fn ln(self) -> Self {
        self.0.map(Float::ln).into()
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        self.0.zip(base.0, Float::log).into()
    }

    #[inline]
    fn log2(self) -> Self {
        self.0.map(Float::log2).into()
    }

    #[inline]
    fn log10(self) -> Self {
        self.0.map(Float::log10).into()
    }

    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        self.0.zip(other.0, Float::abs_sub).into()
    }

    #[inline]
    fn cbrt(self) -> Self {
        self.0.map(Float::cbrt).into()
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        self.0.zip(other.0, Float::hypot).into()
    }

    #[inline]
    fn sin(self) -> Self {
        self.0.map(Float::sin).into()
    }

    #[inline]
    fn cos(self) -> Self {
        self.0.map(Float::cos).into()
    }

    #[inline]
    fn tan(self) -> Self {
        self.0.map(Float::tan).into()
    }

    #[inline]
    fn asin(self) -> Self {
        self.0.map(Float::asin).into()
    }

    #[inline]
    fn acos(self) -> Self {
        self.0.map(Float::acos).into()
    }

    #[inline]
    fn atan(self) -> Self {
        self.0.map(Float::atan).into()
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        self.0.zip(other.0, Float::atan2).into()
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let mut sin_array = GenericArray::uninit();
        let mut cos_array = GenericArray::uninit();

        let mut sin_destination = IntrusiveArrayBuilder::new(&mut sin_array);
        let mut cos_destination = IntrusiveArrayBuilder::new(&mut cos_array);

        if mem::needs_drop::<T>() {
            unsafe {
                let mut source = ArrayConsumer::new(self.0);

                let (source_iter, source_position) = source.iter_position();

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

                            sin.write(s);
                            cos.write(c);

                            *sin_destination_position = *source_position;
                            *cos_destination_position = *source_position;
                        });
                }
            }
        } else {
            unsafe {
                let (sin_destination_iter, _) = sin_destination.iter_position();
                let (cos_destination_iter, _) = cos_destination.iter_position();

                sin_destination_iter
                    .zip(cos_destination_iter)
                    .zip(self.iter())
                    .for_each(|((sin, cos), src)| {
                        let (s, c) = Float::sin_cos(ptr::read(src));

                        sin.write(s);
                        cos.write(c);
                    });
            }
        }

        (
            NumericArray::new(unsafe { sin_destination.finish(); IntrusiveArrayBuilder::array_assume_init(sin_array) }),
            NumericArray::new(unsafe { cos_destination.finish(); IntrusiveArrayBuilder::array_assume_init(cos_array) }),
        )
    }

    #[inline]
    fn exp_m1(self) -> Self {
        self.0.map(Float::exp_m1).into()
    }

    #[inline]
    fn ln_1p(self) -> Self {
        self.0.map(Float::ln_1p).into()
    }

    #[inline]
    fn sinh(self) -> Self {
        self.0.map(Float::sinh).into()
    }

    #[inline]
    fn cosh(self) -> Self {
        self.0.map(Float::cosh).into()
    }

    #[inline]
    fn tanh(self) -> Self {
        self.0.map(Float::tanh).into()
    }

    #[inline]
    fn asinh(self) -> Self {
        self.0.map(Float::asinh).into()
    }

    #[inline]
    fn acosh(self) -> Self {
        self.0.map(Float::acosh).into()
    }

    #[inline]
    fn atanh(self) -> Self {
        self.0.map(Float::atanh).into()
    }
});
