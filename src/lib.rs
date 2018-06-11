// Copyright 2018 0-0-1 and Contributors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This small library provides the [`Drc`] and [`Weak`] types. 'Drc'
//! stands for 'Dynamically Reference Counted', and is the primary type that
//! is used to interact with data stored within. It behaves identically to an
//! [`Rc`], except that it can be converted into an [`Arc`], which is safe to
//! pass across threads and then convert back into a `Drc`. This prevents
//! unnecessary atomic operations when working on a single thread, without
//! limiting the data to the thread.
//!
//! # Technical Details
//!
//! Essentially, when an [`Arc`] is converted into a [`Drc`], it is actually
//! preserved in memory within a data structure located on the heap. Kept here
//! are also a "local" strong and weak reference count. If the local strong
//! reference count is zero, the stored `Arc` will be dropped, but at any
//! other positive value, the `Arc` will be preserved. This effectively means
//! that, within a thread, a `Drc` may be cloned and passed around indefinitely
//! without any atomic operations occuring until the final `Drc` is dropped.
//!
//! `Drc`'s [`Weak`] functions similarly, but is a bit more complicated than
//! `Arc`'s [`Weak`][`ArcWeak`] or [`Rc`]'s [`Weak`][`RcWeak`]. Essentially,
//! even when `Drc`'s local strong reference count reaches zero, though the
//! `Arc` will be dropped, a `Drc` can still be created if another `Arc` or
//! set of `Drc`s exists by upgrading a `Weak` `Drc`.
//!
//! [`Drc::new`] is simply a convenience method to create the `Arc` and place
//! it within. It works exactly the same way as using [`from`] with an `Arc`.
//!
//! [`Drc`]: ./struct.Drc.html
//! [`Weak`]: ./struct.Weak.html
//!
//! [`Drc::new`]: ./struct.Drc.html#method.new
//!
//! [`Rc`]: https://doc.rust-lang.org/std/rc/struct.Rc.html
//! [`RcWeak`]: https://doc.rust-lang.org/std/rc/struct.Weak.html
//! [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
//! [`ArcWeak`]: https://doc.rust-lang.org/std/sync/struct.Weak.html
//!
//! [`from`]: https://doc.rust-lang.org/std/convert/trait.From.html#tymethod.from

mod drc;
mod weak;

pub use drc::Drc;
pub use weak::Weak;
