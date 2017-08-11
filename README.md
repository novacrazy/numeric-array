numeric-array
=============

`numeric-array` is a wrapper around [`generic-array`](https://github.com/fizyk20/generic-array) that adds efficient numeric trait implementations, often times making use of SIMD instructions and compile-time evaluations.

All stable `std::ops` traits are implemented for `NumericArray` itself, plus the thin `NumericConstant` type, which is required to differeniate constant values from `NumericArray` itself.

Additionally, most of `num_traits` are implemented, including `Num` itself. So you can even use a whole array as a generic number.

Example:

```rust
#[macro_use]
extern crate generic_array;
#[macro_use]
extern crate numeric_array;

use numeric_array::NumericArray;

fn main() {
    let a = narr![i32; 1, 3, 5, 7];
    let b = narr![i32; 2, 4, 6, 8];

    let c = a + b * nconstant!(2);

    assert_eq!(c, narr![i32; 5, 11, 17, 23]);
}
```

When used with `RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"`, then Rust and LLVM are smart enough to turn almost all operations into SIMD instructions, or even just evaluate them at compile time. The above example is actually evaluated at compile time, so if you were to view the assembly it would show the result only. Rust is pretty smart.

Therefore, this is ideal for situations where simple component-wise operations are required for arrays.