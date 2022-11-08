//
// /// A helper data structure that behaves like a HashMap, but is based on a vector.
// /// It can be used if the key is essentially a usize (i.e. implements Into<usize>).
// /// It is wasteful if we try to insert spare data (when we expect a lot of unpopulated elements in the middle).
// /// It is a good candidate if we expect to have
// pub mod vector_map {
//     use std::marker::PhantomData;
//     use std::ops::{Index, IndexMut};
//
//     struct VectorMap<Key: Into<usize>, Value> {
//         pub vector: Vec<Option<Value>>,
//         phantom_data: PhantomData<Key>
//     }
//
//     impl<Key: Into<usize>, Value> VectorMap<Key, Value> {
//         pub fn get(&self, key: Key) -> Option<&Value> {
//             let index: usize = <Key as Into<usize>>::into(key);
//             self.vector.get(index).and_then(|x| x.as_ref())
//         }
//         pub fn get_mut(&mut self, key: Key) -> Option<&mut Value> {
//             let index: usize = <Key as Into<usize>>::into(key);
//             self.vector.get_mut(index).and_then(|x| x.as_mut())
//         }
//     }
//
//     impl<Key: Into<usize>, Value> Index<Key> for VectorMap<Key, Value> {
//         type Output = Value;
//         fn index(&self, key: Key) -> &Value {
//             let index: usize = <Key as Into<usize>>::into(key);
//             self.vector[index].as_ref().unwrap()
//         }
//     }
//
//     impl<Key: Into<usize>, Value> IndexMut<Key> for VectorMap<Key, Value> {
//         fn index_mut(&mut self, key: Key) -> &mut Value {
//             let index: usize = <Key as Into<usize>>::into(key);
//             self.vector.index_mut(index).as_mut().unwrap()
//         }
//     }
// }