use serde::{
    de::{Deserialize, Deserializer},
    ser::{Serialize, Serializer},
};

use generic_array::{ArrayLength, GenericArray};

use super::{NumericArray, NumericConstant};

impl<T: Serialize, N: ArrayLength> Serialize for NumericArray<T, N> {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>, N: ArrayLength> Deserialize<'de> for NumericArray<T, N>
where
    T: Default,
{
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        GenericArray::deserialize(deserializer).map(NumericArray)
    }
}

impl<T: Serialize> Serialize for NumericConstant<T> {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for NumericConstant<T> {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(NumericConstant)
    }
}

#[cfg(test)]
mod test {
    extern crate serde_json;

    use self::serde_json::{from_str, to_string};

    use super::*;

    #[test]
    fn serialize() {
        let value = NumericArray::from(arr![i32; 1, 2, 3, 4]);

        assert_eq!(to_string(&value).unwrap(), "[1,2,3,4]");
    }

    #[test]
    fn deserialize() {
        let value: NumericArray<i32, _> = from_str("[4, 3, 2, 1]").unwrap();

        assert_eq!(value, NumericArray::from(arr![i32; 4, 3, 2, 1]));
    }
}
