numeric-array
=============

[![crates.io](https://img.shields.io/crates/v/numeric-array.svg)](https://crates.io/crates/numeric-array)
[![Documentation](https://docs.rs/numeric-array/badge.svg)](https://docs.rs/numeric-array)
[![MIT/Apache-2 licensed](https://img.shields.io/crates/l/numeric-array.svg)](./LICENSE-Apache)

`numeric-array` is a wrapper around [`generic-array`](https://github.com/fizyk20/generic-array) that adds efficient numeric trait implementations, designed to encourage LLVM to autovectorize expressions into SIMD instructions and perform compile-time evaluation.

All stable `core::ops` traits are implemented for `NumericArray` itself, plus the thin `NumericConstant` type, which is required to differentiate constant values from `NumericArray` itself.

Additionally, most of `num_traits` are implemented, including `Num` itself. So you can even use a whole array as a generic number.

Example:

```rust
use num_traits::Float;
use numeric_array::{NumericArray, narr};

fn main() {
    let a = narr![f32; 1, 2, 3, 4];
    let b = narr![f32; 5, 6, 7, 8];
    let c = narr![f32; 9, 1, 2, 3];

    // Compiles to a single vfmadd213ps instruction on my machine
    let d = a.mul_add(b, c);

    assert_eq!(d, narr![f32; 14, 13, 23, 35]);
}
```

When used with `RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"`, then Rust and LLVM are smart enough to autovectorize almost all operations into SIMD instructions, or even just evaluate them at compile time. The above example is actually evaluated at compile time, so if you were to view the assembly it would show the result only.

This is ideal for situations where simple component-wise operations are required for arrays.