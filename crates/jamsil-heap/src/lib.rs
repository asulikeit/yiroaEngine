//! Table storage (spec 10): heap pages, the records of the `HEAP` resource manager and their redo, the
//! free-space and visibility maps, and (to come) overflow values and the access interface of 10.8.
//!
//! The page operations are plain functions over page bytes. The normal path and redo call the same ones
//! (spec 10.3: "the normal path and redo call the same function"), so the page that a record produces on
//! redo cannot differ from the page it was written from; the tests run both and compare.
#![forbid(unsafe_code)]

pub mod maps;
pub mod pageops;
pub mod redo;
pub mod rel;

pub use maps::*;
pub use pageops::*;
pub use redo::*;
pub use rel::*;
