#![allow(missing_docs)]

use std::ops::{Add, Mul};

use num_traits::{Signed, Zero};

use super::*;

pub trait Geometric<T> {
    fn scalar_product(&self, other: &Self) -> T;

    fn abs_scalar_product(&self, other: &Self) -> T
    where
        T: Signed;

    fn norm_squared(&self) -> T;
}

impl<T, N: ArrayLength<T>> Geometric<T> for NumericArray<T, N>
where
    T: Add<Output = T> + Mul<Output = T> + Zero + Copy,
{
    #[inline(always)]
    fn scalar_product(&self, other: &Self) -> T {
        (self * other).0.fold(T::zero(), Add::add)
    }

    #[inline(always)]
    fn abs_scalar_product(&self, other: &Self) -> T
    where
        T: Signed,
    {
        (self * other).0.fold(T::zero(), |sum, x| sum + x.abs())
    }

    #[inline(always)]
    fn norm_squared(&self) -> T {
        self.scalar_product(self)
    }
}
