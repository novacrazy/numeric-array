#![allow(unused_imports)]

use std::marker::PhantomData;

use super::*;

use generic_array::functional::*;

pub mod expr_ops;

pub trait Expression {
    type Output;

    type Iter: Iterator;

    fn iter(self) -> Self::Iter;

    fn eval(self) -> Self::Output;
}

pub type ExpressionItem<E> = <<E as Expression>::Iter as Iterator>::Item;

impl<'a, T, N: ArrayLength<T>> Expression for &'a NumericArray<T, N> {
    type Output = Self;

    type Iter = <&'a NumericArray<T, N> as IntoIterator>::IntoIter;

    fn iter(self) -> Self::Iter {
        self.as_slice().iter()
    }

    fn eval(self) -> Self { self }
}

pub trait UnaryOp<T> {
    type Output;

    fn op(T) -> Self::Output;
}

pub trait BinaryOp<A, B> {
    type Output;

    fn op(A, B) -> Self::Output;
}

pub struct UnaryExpression<T, O: UnaryOp<T>, E: Expression> {
    operand: E,
    _op: PhantomData<(T, O)>
}

impl<T, O: UnaryOp<T>, E: Expression> Expression for UnaryExpression<T, O, E> {

}

pub mod ops {
    use super::*;

    use std::ops::*;
    use num_traits::*;

    macro_rules! impl_unary_ops_tags {
        ($($trait:ident::$op:ident => $tag:ident),*) => {
            $(
                pub struct $tag;

                impl<T> UnaryOp<T> for $tag
                where
                    T: $trait,
                {
                    type Output = <T as $trait>::Output;

                    fn op(value: T) -> Self::Output {
                        $trait::$op(value)
                    }
                }
            )*
        }
    }

    impl_unary_ops_tags! {
        Inv::inv => InvTag,
        Neg::neg => NegTag,
        Not::not => NotTag
    }
}