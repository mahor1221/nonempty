#[cfg(feature = "serialize")]
use serde::{
    de,
    ser::{SerializeSeq, Serializer},
    Deserialize, Serialize,
};
use std::cmp::Ordering;
use std::mem;
use std::{iter, vec};

#[cfg(feature = "arbitrary")]
use proptest_derive::Arbitrary;

#[macro_export]
macro_rules! nonempty_boxed {
    ($h:expr, $($x:expr),+ $(,)?) => {{
        let tail = vec![$($x),*];
        $crate::boxed::NonEmpty{ head: Box::new($h), tail }
    }};
    ($h:expr) => {
        $crate::boxed::NonEmpty{
            head: Box::new($h),
            tail: Vec::new(),
        }
    };
}

/// Non-empty vector.
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NonEmpty<T> {
    pub head: Box<T>,
    pub tail: Vec<T>,
}

#[cfg(feature = "serialize")]
impl<'de, T> Deserialize<'de> for NonEmpty<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match Vec::<T>::deserialize(deserializer)? {
            v if v.is_empty() => Err(de::Error::custom(
                "the vector provided was empty, NonEmpty needs at least one element",
            )),
            v => Ok(NonEmpty::from_vec(v).unwrap_or_else(|| unreachable!())),
        }
    }
}

// Nb. `Serialize` is implemented manually, as serde's `into` container attribute
// requires a `T: Clone` bound which we'd like to avoid.
#[cfg(feature = "serialize")]
impl<T> Serialize for NonEmpty<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for e in self {
            seq.serialize_element(e)?;
        }
        seq.end()
    }
}

pub struct Iter<'a, T> {
    head: Option<&'a T>,
    tail: &'a [T],
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(value) = self.head.take() {
            Some(value)
        } else if let Some((first, rest)) = self.tail.split_first() {
            self.tail = rest;
            Some(first)
        } else {
            None
        }
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some((last, rest)) = self.tail.split_last() {
            self.tail = rest;
            Some(last)
        } else if let Some(first_value) = self.head.take() {
            Some(first_value)
        } else {
            None
        }
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {
    fn len(&self) -> usize {
        self.tail.len() + self.head.map_or(0, |_| 1)
    }
}

impl<'a, T> core::iter::FusedIterator for Iter<'a, T> {}

impl<T> NonEmpty<T> {
    /// Alias for [`NonEmpty::singleton`].
    pub fn new(e: T) -> Self {
        Self::singleton(e)
    }

    /// Attempt to convert an iterator into a `NonEmpty` vector.
    /// Returns `None` if the iterator was empty.
    pub fn collect<I>(iter: I) -> Option<NonEmpty<T>>
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let head = iter.next()?;
        Some(Self {
            head: Box::new(head),
            tail: iter.collect(),
        })
    }

    /// Create a new non-empty list with an initial element.
    pub fn singleton(head: T) -> Self {
        NonEmpty {
            head: Box::new(head),
            tail: Vec::new(),
        }
    }

    /// Always returns false.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Get the first element. Never fails.
    pub const fn first(&self) -> &T {
        &self.head
    }

    /// Get the mutable reference to the first element. Never fails.
    pub fn first_mut(&mut self) -> &mut T {
        &mut self.head
    }

    /// Get the possibly-empty tail of the list.
    pub fn tail(&self) -> &[T] {
        &self.tail
    }

    /// Push an element to the end of the list.
    pub fn push(&mut self, e: T) {
        self.tail.push(e)
    }

    /// Pop an element from the end of the list.
    pub fn pop(&mut self) -> Option<T> {
        self.tail.pop()
    }

    /// Inserts an element at position index within the vector, shifting all elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if index > len.
    pub fn insert(&mut self, index: usize, element: T) {
        let len = self.len();
        assert!(index <= len);

        if index == 0 {
            let head = mem::replace(&mut self.head, Box::new(element));
            self.tail.insert(0, *head);
        } else {
            self.tail.insert(index - 1, element);
        }
    }

    /// Get the length of the list.
    pub fn len(&self) -> usize {
        self.tail.len() + 1
    }

    /// Get the capacity of the list.
    pub fn capacity(&self) -> usize {
        self.tail.capacity() + 1
    }

    /// Get the last element. Never fails.
    pub fn last(&self) -> &T {
        match self.tail.last() {
            None => &self.head,
            Some(e) => e,
        }
    }

    /// Get the last element mutably.
    pub fn last_mut(&mut self) -> &mut T {
        match self.tail.last_mut() {
            None => &mut self.head,
            Some(e) => e,
        }
    }

    /// Check whether an element is contained in the list.
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.iter().any(|e| e == x)
    }

    /// Get an element by index.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index == 0 {
            Some(&self.head)
        } else {
            self.tail.get(index - 1)
        }
    }

    /// Get an element by index, mutably.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index == 0 {
            Some(&mut self.head)
        } else {
            self.tail.get_mut(index - 1)
        }
    }

    /// Truncate the list to a certain size. Must be greater than `0`.
    pub fn truncate(&mut self, len: usize) {
        assert!(len >= 1);
        self.tail.truncate(len - 1);
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            head: Some(&self.head),
            tail: &self.tail,
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        iter::once(&mut *self.head).chain(self.tail.iter_mut())
    }

    /// Often we have a `Vec` (or slice `&[T]`) but want to ensure that it is `NonEmpty` before
    /// proceeding with a computation. Using `from_slice` will give us a proof
    /// that we have a `NonEmpty` in the `Some` branch, otherwise it allows
    /// the caller to handle the `None` case.
    pub fn from_slice(slice: &[T]) -> Option<NonEmpty<T>>
    where
        T: Clone,
    {
        slice.split_first().map(|(h, t)| NonEmpty {
            head: Box::new(h.clone()),
            tail: t.into(),
        })
    }

    /// Often we have a `Vec` (or slice `&[T]`) but want to ensure that it is `NonEmpty` before
    /// proceeding with a computation. Using `from_vec` will give us a proof
    /// that we have a `NonEmpty` in the `Some` branch, otherwise it allows
    /// the caller to handle the `None` case.
    ///
    /// This version will consume the `Vec` you pass in. If you would rather pass the data as a
    /// slice then use `NonEmpty::from_slice`.
    pub fn from_vec(mut vec: Vec<T>) -> Option<NonEmpty<T>> {
        if vec.is_empty() {
            None
        } else {
            let head = Box::new(vec.remove(0));
            Some(NonEmpty { head, tail: vec })
        }
    }

    /// Deconstruct a `NonEmpty` into its head and tail.
    /// This operation never fails since we are guranteed
    /// to have a head element.
    pub fn split_first(&self) -> (&T, &[T]) {
        (&self.head, &self.tail)
    }

    /// Deconstruct a `NonEmpty` into its first, last, and
    /// middle elements, in that order.
    ///
    /// If there is only one element then first == last.
    pub fn split(&self) -> (&T, &[T], &T) {
        match self.tail.split_last() {
            None => (&self.head, &[], &self.head),
            Some((last, middle)) => (&self.head, middle, last),
        }
    }

    /// Append a `Vec` to the tail of the `NonEmpty`.
    pub fn append(&mut self, other: &mut Vec<T>) {
        self.tail.append(other)
    }

    /// A structure preserving `map`. This is useful for when
    /// we wish to keep the `NonEmpty` structure guaranteeing
    /// that there is at least one element. Otherwise, we can
    /// use `nonempty.iter().map(f)`.
    pub fn map<U, F>(self, mut f: F) -> NonEmpty<U>
    where
        F: FnMut(T) -> U,
    {
        NonEmpty {
            head: Box::new(f(*self.head)),
            tail: self.tail.into_iter().map(f).collect(),
        }
    }

    /// A structure preserving, fallible mapping function.
    pub fn try_map<E, U, F>(self, mut f: F) -> Result<NonEmpty<U>, E>
    where
        F: FnMut(T) -> Result<U, E>,
    {
        Ok(NonEmpty {
            head: Box::new(f(*self.head)?),
            tail: self.tail.into_iter().map(f).collect::<Result<_, _>>()?,
        })
    }

    /// When we have a function that goes from some `T` to a `NonEmpty<U>`,
    /// we may want to apply it to a `NonEmpty<T>` but keep the structure flat.
    /// This is where `flat_map` shines.
    pub fn flat_map<U, F>(self, mut f: F) -> NonEmpty<U>
    where
        F: FnMut(T) -> NonEmpty<U>,
    {
        let mut heads = f(*self.head);
        let mut tails = self
            .tail
            .into_iter()
            .flat_map(|t| f(t).into_iter())
            .collect();
        heads.append(&mut tails);
        heads
    }

    /// Flatten nested `NonEmpty`s into a single one.
    pub fn flatten(full: NonEmpty<NonEmpty<T>>) -> Self {
        full.flat_map(|n| n)
    }

    /// Binary searches this sorted non-empty vector for a given element.
    ///
    /// If the value is found then Result::Ok is returned, containing the index of the matching element.
    /// If there are multiple matches, then any one of the matches could be returned.
    ///
    /// If the value is not found then Result::Err is returned, containing the index where a
    /// matching element could be inserted while maintaining sorted order.
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|p| p.cmp(x))
    }

    /// Binary searches this sorted non-empty with a comparator function.
    ///
    /// The comparator function should implement an order consistent with the sort order of the underlying slice,
    /// returning an order code that indicates whether its argument is Less, Equal or Greater the desired target.
    ///
    /// If the value is found then Result::Ok is returned, containing the index of the matching element.
    /// If there are multiple matches, then any one of the matches could be returned.
    /// If the value is not found then Result::Err is returned, containing the index where a matching element could be
    /// inserted while maintaining sorted order.
    pub fn binary_search_by<'a, F>(&'a self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a T) -> Ordering,
    {
        match f(&self.head) {
            Ordering::Equal => Ok(0),
            Ordering::Greater => Err(0),
            Ordering::Less => self
                .tail
                .binary_search_by(f)
                .map(|index| index + 1)
                .map_err(|index| index + 1),
        }
    }

    /// Binary searches this sorted non-empty vector with a key extraction function.
    ///
    /// Assumes that the vector is sorted by the key.
    ///
    /// If the value is found then Result::Ok is returned, containing the index of the matching element. If there are multiple matches,
    /// then any one of the matches could be returned. If the value is not found then Result::Err is returned,
    /// containing the index where a matching element could be inserted while maintaining sorted order.
    pub fn binary_search_by_key<'a, B, F>(&'a self, b: &B, mut f: F) -> Result<usize, usize>
    where
        B: Ord,
        F: FnMut(&'a T) -> B,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Returns the maximum element in the non-empty vector.
    ///
    /// This will return the first item in the vector if the tail is empty.
    pub fn maximum(&self) -> &T
    where
        T: Ord,
    {
        self.maximum_by(|i, j| i.cmp(j))
    }

    /// Returns the minimum element in the non-empty vector.
    ///
    /// This will return the first item in the vector if the tail is empty.
    pub fn minimum(&self) -> &T
    where
        T: Ord,
    {
        self.minimum_by(|i, j| i.cmp(j))
    }

    /// Returns the element that gives the maximum value with respect to the specified comparison function.
    ///
    /// This will return the first item in the vector if the tail is empty.
    pub fn maximum_by<F>(&self, compare: F) -> &T
    where
        F: Fn(&T, &T) -> Ordering,
    {
        let mut max = &*self.head;
        for i in self.tail.iter() {
            max = match compare(max, i) {
                Ordering::Equal => max,
                Ordering::Less => i,
                Ordering::Greater => max,
            };
        }
        max
    }

    /// Returns the element that gives the minimum value with respect to the specified comparison function.
    ///
    /// This will return the first item in the vector if the tail is empty.
    pub fn minimum_by<F>(&self, compare: F) -> &T
    where
        F: Fn(&T, &T) -> Ordering,
    {
        self.maximum_by(|a, b| compare(a, b).reverse())
    }

    /// Returns the element that gives the maximum value with respect to the specified function.
    ///
    /// This will return the first item in the vector if the tail is empty.
    pub fn maximum_by_key<U, F>(&self, f: F) -> &T
    where
        U: Ord,
        F: Fn(&T) -> &U,
    {
        self.maximum_by(|i, j| f(i).cmp(f(j)))
    }

    /// Returns the element that gives the minimum value with respect to the specified function.
    ///
    /// This will return the first item in the vector if the tail is empty.
    pub fn minimum_by_key<U, F>(&self, f: F) -> &T
    where
        U: Ord,
        F: Fn(&T) -> &U,
    {
        self.minimum_by(|i, j| f(i).cmp(f(j)))
    }
}

impl<T: Default> Default for NonEmpty<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> From<NonEmpty<T>> for Vec<T> {
    /// Turns a non-empty list into a Vec.
    fn from(nonempty: NonEmpty<T>) -> Vec<T> {
        iter::once(*nonempty.head).chain(nonempty.tail).collect()
    }
}

impl<T> From<NonEmpty<T>> for (T, Vec<T>) {
    /// Turns a non-empty list into a Vec.
    fn from(nonempty: NonEmpty<T>) -> (T, Vec<T>) {
        (*nonempty.head, nonempty.tail)
    }
}

impl<T> From<(T, Vec<T>)> for NonEmpty<T> {
    /// Turns a pair of an element and a Vec into
    /// a NonEmpty.
    fn from((head, tail): (T, Vec<T>)) -> Self {
        NonEmpty {
            head: Box::new(head),
            tail,
        }
    }
}

impl<T> IntoIterator for NonEmpty<T> {
    type Item = T;
    type IntoIter = iter::Chain<iter::Once<T>, vec::IntoIter<Self::Item>>;

    fn into_iter(self) -> Self::IntoIter {
        iter::once(*self.head).chain(self.tail)
    }
}

impl<'a, T> IntoIterator for &'a NonEmpty<T> {
    type Item = &'a T;
    type IntoIter = iter::Chain<iter::Once<&'a T>, std::slice::Iter<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        iter::once(&*self.head).chain(self.tail.iter())
    }
}

impl<T> std::ops::Index<usize> for NonEmpty<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        if index > 0 {
            &self.tail[index - 1]
        } else {
            &self.head
        }
    }
}

impl<T> std::ops::IndexMut<usize> for NonEmpty<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        if index > 0 {
            &mut self.tail[index - 1]
        } else {
            &mut self.head
        }
    }
}

impl<A> Extend<A> for NonEmpty<A> {
    fn extend<T: IntoIterator<Item = A>>(&mut self, iter: T) {
        self.tail.extend(iter)
    }
}

pub mod serialize {
    use std::{convert::TryFrom, fmt};

    use super::NonEmpty;

    #[derive(Debug)]
    pub enum Error {
        Empty,
    }

    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Empty => f.write_str(
                    "the vector provided was empty, NonEmpty needs at least one element",
                ),
            }
        }
    }

    impl<T> TryFrom<Vec<T>> for NonEmpty<T> {
        type Error = Error;

        fn try_from(vec: Vec<T>) -> Result<Self, Self::Error> {
            NonEmpty::from_vec(vec).ok_or(Error::Empty)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NonEmpty;

    #[test]
    fn test_from_conversion() {
        let result = NonEmpty::from((1, vec![2, 3, 4, 5]));
        let expected = NonEmpty {
            head: Box::new(1),
            tail: vec![2, 3, 4, 5],
        };
        assert_eq!(result, expected);
    }

    #[test]
    fn test_into_iter() {
        let nonempty = NonEmpty::from((0, vec![1, 2, 3]));
        for (i, n) in nonempty.into_iter().enumerate() {
            assert_eq!(i as i32, n);
        }
    }

    #[test]
    fn test_iter_syntax() {
        let nonempty = NonEmpty::from((0, vec![1, 2, 3]));
        for n in &nonempty {
            let _ = *n; // Prove that we're dealing with references.
        }
        for _ in nonempty {}
    }

    #[test]
    fn test_iter_both_directions() {
        let nonempty = NonEmpty::from((0, vec![1, 2, 3]));
        assert_eq!(
            nonempty.iter().cloned().collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            nonempty.iter().rev().cloned().collect::<Vec<_>>(),
            vec![3, 2, 1, 0]
        );
    }

    #[test]
    fn test_iter_both_directions_at_once() {
        let nonempty = NonEmpty::from((0, vec![1, 2, 3]));
        let mut i = nonempty.iter();
        assert_eq!(i.next(), Some(&0));
        assert_eq!(i.next_back(), Some(&3));
        assert_eq!(i.next(), Some(&1));
        assert_eq!(i.next_back(), Some(&2));
        assert_eq!(i.next(), None);
        assert_eq!(i.next_back(), None);
    }

    #[test]
    fn test_mutate_head() {
        let mut non_empty = NonEmpty::new(42);
        *non_empty.head += 1;
        assert_eq!(*non_empty.head, 43);

        let mut non_empty = NonEmpty::from((1, vec![4, 2, 3]));
        *non_empty.head *= 42;
        assert_eq!(*non_empty.head, 42);
    }

    #[test]
    fn test_to_nonempty() {
        use std::iter::{empty, once};

        assert_eq!(NonEmpty::<()>::collect(empty()), None);
        assert_eq!(NonEmpty::<()>::collect(once(())), Some(NonEmpty::new(())));
        assert_eq!(
            NonEmpty::<u8>::collect(once(1).chain(once(2))),
            Some(nonempty_boxed!(1, 2))
        );
    }

    #[test]
    fn test_try_map() {
        assert_eq!(
            nonempty_boxed!(1, 2, 3, 4).try_map(Ok::<_, String>),
            Ok(nonempty_boxed!(1, 2, 3, 4))
        );
        assert_eq!(
            nonempty_boxed!(1, 2, 3, 4).try_map(|i| if i % 2 == 0 {
                Ok(i)
            } else {
                Err("not even")
            }),
            Err("not even")
        );
    }

    #[cfg(feature = "serialize")]
    mod serialize {
        use super::NonEmpty;
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
        pub struct SimpleSerializable(pub i32);

        #[test]
        fn test_simple_round_trip() -> Result<(), Box<dyn std::error::Error>> {
            // Given
            let mut non_empty = NonEmpty::new(SimpleSerializable(42));
            non_empty.push(SimpleSerializable(777));

            // When
            let res = serde_json::from_str::<'_, NonEmpty<SimpleSerializable>>(
                &serde_json::to_string(&non_empty)?,
            )?;

            // Then
            assert_eq!(res, non_empty);

            Ok(())
        }

        #[test]
        fn test_serialization() -> Result<(), Box<dyn std::error::Error>> {
            let ne = nonempty_boxed![1, 2, 3, 4, 5];
            let ve = vec![1, 2, 3, 4, 5];

            assert_eq!(serde_json::to_string(&ne)?, serde_json::to_string(&ve)?);

            Ok(())
        }
    }
}
