// Copyright 2018 0-0-1 and Contributors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// This file contains a substantial portion of code derived from
// https://github.com/rust-lang/rust/blob/master/src/liballoc/rc.rs
// which has the following license header:
//
//      Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
//      file at the top-level directory of this distribution and at
//      http://rust-lang.org/COPYRIGHT.
//
//      Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
//      http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
//      <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
//      option. This file may not be copied, modified, or distributed
//      except according to those terms.

use std::borrow::Borrow;
use std::cell::Cell;
use std::cmp::Ordering;
use std::ffi::{CStr, CString, OsStr, OsString};
use std::fmt::{Debug, Display, Formatter, Pointer, Result as FmtResult};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::{Arc, Weak as WeakArc};

use weak::Weak;

/// A single-threaded reference-counting pointer with the special ability to be
/// converted into an [`Arc`]. 'Drc' stands for 'Dynamically Reference
/// Counted'.
///
/// See the [crate-level documentation][crate] for more details.
///
/// The inherent methods of `Drc` are all associated functions, which means you
/// have to call them as e.g. [`Drc::get_mut(&mut value)`][`get_mut`] instead
/// `value.get_mut()`. This avoids conflict with methods of the inner type `T`.
///
/// [crate]: ./index.html
///
/// [`get_mut`]: ./struct.Drc.html#method.get_mut
///
/// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
pub struct Drc<T>
where
    T: ?Sized,
{
    pub(crate) ptr: NonNull<DrcInner<T>>,
    pub(crate) phantom: PhantomData<T>,
}

impl<T> Drc<T> {
    /// Constructs a new `Drc`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    /// ```
    pub fn new(value: T) -> Drc<T> {
        Drc::from_arc(Arc::new(value))
    }
}

impl<T> Drc<T>
where
    T: ?Sized,
{
    /// Clones the internal [`Arc`] (incrementing the atomic strong reference
    /// count) so that the shared state can be referenced on another thread.
    /// Then, returns this newly cloned `Arc`. To convert the `Arc` back
    /// into a `Drc` (with a [`separate`] local state), use [`from`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    /// let arc_five = Drc::detach(&five);
    ///
    /// assert_eq!(*five, *arc_five);
    /// ```
    ///
    /// [`separate`]: ./struct.Drc.html#method.separate
    ///
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    ///
    /// [`from`]: https://doc.rust-lang.org/std/convert/trait.From.html#tymethod.from
    pub fn detach(this: &Drc<T>) -> Arc<T> {
        Arc::clone(this.arc())
    }

    /// Clone the internal [`Arc`] (incrementing the atomic strong reference
    /// count), create new local reference counts, and associate a new `Drc` to
    /// these new reference counts.
    ///
    /// This is not too useful outside of testing, but it is provided
    /// as a way to simulate the intended process of sending an `Arc` across a
    /// thread and converting it back into a `Drc` without the need for
    /// multiple method calls.
    ///
    /// [`Drc::linked`][`linked`] will not evaluate true for separated values,
    /// though [`Drc::ptr_eq`][`ptr_eq`] will.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    /// let separate_five = Drc::separate(&five);
    ///
    /// assert!(Drc::ptr_eq(&five, &separate_five));
    /// assert!(!Drc::linked(&five, &separate_five));
    /// ```
    ///
    /// [`linked`]: ./struct.Drc.html#method.linked
    /// [`ptr_eq`]: ./struct.Drc.html#method.ptr_eq
    ///
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    pub fn separate(this: &Drc<T>) -> Drc<T> {
        Drc::from_arc(Drc::detach(this))
    }
}

impl<T> Drc<T> {
    /// Returns the contained value, if the [`Arc`] associated with the `Drc`
    /// has exactly one strong reference.
    ///
    /// Otherwise, an [`Err`][`Result`] will be returned with the same `Drc`
    /// that was passed in.
    ///
    /// This will succeed even if there are outstanding weak references.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// // The `Drc` here is the only strong reference to 3, so it is successfully
    /// // unwrapped.
    /// let x = Drc::new(3);
    /// assert_eq!(Drc::try_unwrap(x), Ok(3));
    ///
    /// // There are two `Drc` strong references to 4, so it is not successfully
    /// // unwrapped.
    /// let x = Drc::new(4);
    /// let _y = Drc::clone(&x);
    /// assert_eq!(*Drc::try_unwrap(x).unwrap_err(), 4);
    ///
    /// // There is a `Drc` and an `Arc` strong reference to 5, so it is not
    /// // sucessfully unwrapped.
    /// let x = Drc::new(5);
    /// let _y = Drc::detach(&x);
    /// assert_eq!(*Drc::try_unwrap(x).unwrap_err(), 5);
    /// ```
    ///
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    /// [`Result`]: https://doc.rust-lang.org/std/result/enum.Result.html
    pub fn try_unwrap(this: Drc<T>) -> Result<T, Drc<T>> {
        if Drc::strong_count(&this, true) == 1 {
            match Arc::try_unwrap(unsafe { this.take_arc() }) {
                Ok(value) => {
                    this.strong().set(0);
                    let weak = this.weak().get();
                    if weak == 1 {
                        unsafe {
                            // This strong pointer was the last pointer, weak or
                            // strong, associated with the `Arc`. Simply deallocate
                            // the storage (only integers and `None`s are stored).
                            Box::from_raw(this.ptr.as_ptr());
                        }
                    } else {
                        // Remove implict weak pointer. Explicit ones remain
                        // to clean up the structure.
                        this.weak().set(weak - 1);
                    }

                    // Prevent a double free.
                    mem::forget(this);

                    // Return the successfully unwrapped value.
                    Ok(value)
                },
                Err(arc) => {
                    // Return the `Arc` to its rightful place.
                    unsafe { this.set_arc(arc) };

                    Err(this)
                },
            }
        } else {
            // We know it can't be zero, as *this* `Drc` exists. Therefore, there
            // is more than one strong pointer, making this a failure.

            Err(this)
        }
    }
}

impl<T> Drc<T>
where
    T: ?Sized,
{
    /// Creates a new [`Weak`] pointer to this value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// let weak_five = Drc::downgrade(&five);
    /// ```
    ///
    /// [`Weak`]: ./struct.Weak.html
    pub fn downgrade(this: &Drc<T>) -> Weak<T> {
        unsafe {
            let storage = this.ptr.as_ptr();
            (*storage).weak.set((*storage).weak.get() + 1);

            if (*storage).weak_ref.is_none() {
                (*storage).weak_ref = Some(Arc::downgrade(this.arc()));
            }
        }

        Weak { ptr: this.ptr }
    }

    /// If `local`, gets the number of [`Weak` (`Drc`)][`Weak`] pointers
    /// associated with the same internal [`Arc`]. Otherwise, gets the number of
    /// [`Weak` (`Arc`)][`WeakArc`] pointers associated associated with the
    /// value.
    ///
    /// It's worth noting that neither of these values are the total counts of
    /// weak pointers associated with a given value. Using a `local` value
    /// of `false` will return the number of `Drc` sets containing at least
    /// one weak pointer plus the number of `Arc` weak pointers. This is
    /// not equivalent to the sum of the total counts for each type of weak
    /// pointer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    /// use std::sync::Arc;
    ///
    /// let five = Drc::new(5);
    /// let _weak_five_a = Drc::downgrade(&five);
    /// let _weak_five_b = Drc::downgrade(&five);
    /// let _weak_five_c = Drc::downgrade(&five);
    ///
    /// // No contribution because no weak pointers.
    /// let _separate_five = Drc::separate(&five);
    ///
    /// // detached_five is an Arc that points to the same value.
    /// let detached_five = Drc::detach(&five);
    /// let _weak_detached_five = Arc::downgrade(&detached_five);
    ///
    /// // 3 values:
    /// // _weak_five_a, _weak_five_b, _weak_five_c
    /// assert_eq!(3, Drc::weak_count(&five, true));
    ///
    /// // 2 values:
    /// // (_weak_five_a, _weak_five_b, _weak_five_c), _weak_detached_five
    /// assert_eq!(2, Drc::weak_count(&five, false));
    /// ```
    ///
    /// [`Weak`]: ./struct.Weak.html
    ///
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    /// [`WeakArc`]: https://doc.rust-lang.org/std/sync/struct.Weak.html
    pub fn weak_count(this: &Drc<T>, local: bool) -> usize {
        if local {
            this.weak().get() - 1
        } else {
            Arc::weak_count(this.arc())
        }
    }

    /// If `local`, gets the number of `Drc` pointers associated with the same
    /// internal [`Arc`]. Otherwise, gets the number of `Arc`s associated with
    /// the value.
    ///
    /// It's worth noting that neither of these values are the total counts of
    /// strong pointers associated with a given value. Using a `local` value of
    /// `false` will return the number of `Drc` sets containing at least one
    /// strong pointer plus the number of `Arc` pointers. This is not
    /// equivalent to the sum of the total counts for each type of strong
    /// pointer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    /// use std::sync::Arc;
    ///
    /// let five = Drc::new(5);
    /// let _also_five = Drc::clone(&five);
    /// let _still_five = Drc::clone(&five);
    ///
    /// // No contribution because no strong pointer.
    /// let _weak_separate_five = {
    ///     let separate_five = Drc::separate(&five);
    ///     Drc::downgrade(&separate_five)
    /// };
    ///
    /// // This is basically a glorified Arc, basically (Arc,)
    /// let _strong_separate_five = Drc::separate(&five);
    ///
    /// // detached_five is an Arc that points to the same value.
    /// let detached_five = Drc::detach(&five);
    /// let _also_detached_five = Arc::clone(&detached_five);
    ///
    /// // 3 values:
    /// // five, _also_five, _still_five
    /// assert_eq!(3, Drc::strong_count(&five, true));
    ///
    /// // 4 values:
    /// // (five, _also_five, _still_five), (_strong_separate_five,), detached_five,
    /// //     _also_detached_five
    /// assert_eq!(4, Drc::strong_count(&five, false));
    /// ```
    ///
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    pub fn strong_count(this: &Drc<T>, local: bool) -> usize {
        if local {
            this.strong().get()
        } else {
            Arc::strong_count(this.arc())
        }
    }

    /// Returns a mutable reference to the inner value, if there are no other
    /// `Drc`s, [`Arc`]s, [weak `Drc`][`Weak`]s, or [weak `Arc`][`WeakArc`]s to
    /// the same value.
    ///
    /// Returns [`None`][`Option`] otherwise, because it is not safe to mutate a
    /// shared value.
    ///
    /// See also [`make_mut`], which will [`clone`] the inner value when it's
    /// shared.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let mut x = Drc::new(3);
    /// *Drc::get_mut(&mut x).unwrap() = 4;
    /// assert_eq!(*x, 4);
    ///
    /// let _y = Drc::clone(&x);
    /// assert!(Drc::get_mut(&mut x).is_none());
    /// ```
    ///
    /// [`Weak`]: ./struct.Weak.html
    ///
    /// [`make_mut`]: ./struct.Drc.html#method.make_mut
    ///
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    /// [`WeakArc`]: https://doc.rust-lang.org/std/sync/struct.Weak.html
    /// [`Option`]: https://doc.rust-lang.org/std/option/enum.Option.html
    ///
    /// [`clone`]: https://doc.rust-lang.org/std/clone/trait.Clone.html#tymethod.clone
    pub fn get_mut(this: &mut Drc<T>) -> Option<&mut T> {
        if Drc::is_unique(this) {
            // Since no mutation is actually happening, we don't need to do anything
            // special in this case, unlike in `make_mut`.
            Arc::get_mut(unsafe { this.arc_mut() })
        } else {
            None
        }
    }

    /// Returns true if two `Drc`s point to the same value (not just values that
    /// compare as equal). Note that as long as the **value** is the same,
    /// association to the same `Arc` is not necessary.
    ///
    /// Contrast with [`linked`], which checks if two `Drc`s are associated with
    /// the same `Arc` (i.e. they were cloned from the same `Drc` without
    /// [`detach`ment][`detach`] or becoming [`separate`]).
    ///
    /// Compare with [`arc_ptr_eq`], which applies the same check for the
    /// **value** on a `Drc` and [`Arc`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// // Associated to same `Arc`.
    /// let same_five = Drc::clone(&five);
    ///
    /// // Associated to different `Arc` but same value.
    /// let separate_five = Drc::separate(&five);
    ///
    /// // A detached and converted `Drc` is the same as a separated `Drc`, so
    /// // this is also associated to a different `Arc` but same value.
    /// let detached_five = Drc::from(Drc::detach(&five));
    ///
    /// // An equal value located at a different memory address.
    /// let other_five = Drc::new(5);
    ///
    /// assert!(Drc::ptr_eq(&five, &same_five));
    /// assert!(Drc::ptr_eq(&five, &separate_five));
    /// assert!(Drc::ptr_eq(&five, &detached_five));
    /// assert!(!Drc::ptr_eq(&five, &other_five));
    /// ```
    ///
    /// [`linked`]: ./struct.Drc.html#method.linked
    /// [`detach`]: ./struct.Drc.html#method.detach
    /// [`separate`]: ./struct.Drc.html#method.separate
    /// [`arc_ptr_eq`]: ./struct.Drc.html#method.arc_ptr_eq
    ///
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    pub fn ptr_eq(this: &Drc<T>, other: &Drc<T>) -> bool {
        Arc::ptr_eq(this.arc(), other.arc())
    }

    /// Returns true if a `Drc` and an [`Arc`] points to the same value (not
    /// just values that compare as equal). Note that as long as the
    /// **value** is the same, association of the passed `Drc` to the `Arc`
    /// is not necessary (and is in fact impossible, as a reference to the
    /// internal `Arc` cannot be retrieved externally).
    ///
    /// Contrast with [`linked`], which checks if two `Drc`s are associated with
    /// the same `Arc` (i.e. they were cloned from the same `Drc` without
    /// [`detach`ment][`detach`] or becoming [`separate`]).
    ///
    /// Compare with [`ptr_eq`], which applies the same check for the **value**
    /// on two `Drc`s.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    /// use std::sync::Arc;
    ///
    /// let five = Drc::new(5);
    ///
    /// // Points to the same value. This is an `Arc`.
    /// let detached_five = Drc::detach(&five);
    ///
    /// // Points to an equal value located at a different memory address.
    /// let other_five = Arc::new(5);
    ///
    /// assert!(Drc::arc_ptr_eq(&five, &detached_five));
    /// assert!(!Drc::arc_ptr_eq(&five, &other_five));
    /// ```
    ///
    /// [`linked`]: ./struct.Drc.html#method.linked
    /// [`detach`]: ./struct.Drc.html#method.detach
    /// [`separate`]: ./struct.Drc.html#method.separate
    /// [`ptr_eq`]: ./struct.Drc.html#method.ptr_eq
    ///
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    pub fn arc_ptr_eq(this: &Drc<T>, other: &Arc<T>) -> bool {
        Arc::ptr_eq(this.arc(), other)
    }

    /// Returns true if two `Drc`s are associated with the same [`Arc`] (i.e.
    /// they were cloned from the same `Drc` without
    /// [`detach`ment][`detach`] or becoming [`separate`]).
    ///
    /// Contrast with [`ptr_eq`] or [`arc_ptr_eq`], which check if two `Drc`s or
    /// a `Drc` and an `Arc` (respectively) point to the same **value**,
    /// regardless of whether the internal `Arc` is the same (which is of
    /// course impossible in `arc_ptr_eq`'s case, as a reference to the
    /// internal `Arc` cannot be retrieved externally).
    ///
    /// Although it was just stated that an internal `Arc` cannot be retrieved
    /// externally, it's worth explicitly noting that no analogue of
    /// `arc_ptr_eq` exists for `linked` because a reference to the
    /// "linked" `Arc` simply cannot be retrieved (`Drc::from(arc)` takes
    /// ownership of the `Arc`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// // Associated to same `Arc`.
    /// let same_five = Drc::clone(&five);
    ///
    /// // Associated to different `Arc` but same value.
    /// let separate_five = Drc::separate(&five);
    ///
    /// // A detached and converted `Drc` is the same as a separated `Drc`, so
    /// // this is also associated to a different `Arc` but same value.
    /// let detached_five = Drc::from(Drc::detach(&five));
    ///
    /// // An equal value located at a different memory address.
    /// let other_five = Drc::new(5);
    ///
    /// assert!(Drc::linked(&five, &same_five));
    /// assert!(!Drc::linked(&five, &separate_five));
    /// assert!(!Drc::linked(&five, &detached_five));
    /// assert!(!Drc::linked(&five, &other_five));
    /// ```
    ///
    /// [`detach`]: ./struct.Drc.html#method.detach
    /// [`separate`]: ./struct.Drc.html#method.separate
    /// [`ptr_eq`]: ./struct.Drc.html#method.ptr_eq
    /// [`arc_ptr_eq`]: ./struct.Drc.html#method.arc_ptr_eq
    ///
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    pub fn linked(this: &Drc<T>, other: &Drc<T>) -> bool {
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T> Drc<T>
where
    T: Clone,
{
    /// Makes a mutable reference into the given `Drc`.
    ///
    /// If there are other `Drc`s, [`Arc`]s, [weak `Drc`][`Weak`]s, or
    /// [weak `Arc`][`WeakArc`]s to the same value, then `make_mut` will invoke
    /// [`clone`] on the inner value to ensure unique ownership. This is also
    /// referred to as clone-on-write.
    ///
    /// See also [`get_mut`], which will fail rather than cloning.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let mut data = Drc::new(5);
    ///
    /// *Drc::make_mut(&mut data) += 1; // Won't clone anything
    /// let mut other_data = Drc::clone(&data); // Won't clone inner data
    /// *Drc::make_mut(&mut data) += 1; // Clones inner data
    /// *Drc::make_mut(&mut data) += 1; // Won't clone anything
    /// *Drc::make_mut(&mut other_data) *= 2; // Won't clone anything
    /// let mut detached_data = Drc::detach(&other_data); // Won't clone inner data
    /// *Drc::make_mut(&mut other_data) += 1; // Clones inner data
    /// *Drc::make_mut(&mut other_data) += 2; // Won't clone anything
    ///
    /// // Now `data`, `other_data`, and `detached_data` all point to different values.
    /// assert_eq!(*data, 8);
    /// assert_eq!(*other_data, 15);
    /// assert_eq!(*detached_data, 12);
    /// ```
    ///
    /// [`Weak`]: ./struct.Weak.html
    ///
    /// [`get_mut`]: ./struct.Drc.html#method.get_mut
    ///
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    /// [`WeakArc`]: https://doc.rust-lang.org/std/sync/struct.Weak.html
    ///
    /// [`clone`]: https://doc.rust-lang.org/std/clone/trait.Clone.html#tymethod.clone
    pub fn make_mut(this: &mut Drc<T>) -> &mut T {
        if Drc::strong_count(this, true) != 1 {
            // There are other `Drc`s associated with the same `Arc`, clone data.
            *this = Drc::new((**this).clone());

            // This is a brand new `Arc`, this should not perform further cloning, but it
            // will sadly do a few atomic operations that it does not need to do.
            Arc::make_mut(unsafe { this.arc_mut() })
        } else {
            if Drc::weak_count(this, true) != 0 {
                // There are some weak pointers, so we need to make a new `Drc` as the
                // `Arc` is guaranteed to change (if our weak pointer count is at least one, so
                // too must `Arc`'s count) but our local weak pointers would
                // not care, thus creating the possibility of multiple
                // aliasing. We need to make a new `Drc` but leave the
                // weak pointers behind.

                // Steal the `Arc` because we're sticking it into a new pointer without updating
                // the atomic reference counts.
                let arc = unsafe { this.take_arc() };

                // Set the strong count to zero; we are moving to a new pointer.
                this.strong().set(0);

                // Remove the implicit weak pointer, but don't clean up because there are still
                // more weak pointers that will do that for us.
                this.weak().set(this.weak().get() - 1);

                // `Drc::from_arc` does not modify the `Arc` at all, it literally sets its bytes
                // in the newly allocated `DrcInner` and leaves it be.
                let mut swap = Drc::from_arc(arc);

                // Place the newly crafted `Drc` pointer into the current `Drc`. This is
                // associated with the same `Arc`, but this time
                // `Arc::make_mut` is safe to call.
                mem::swap(this, &mut swap);

                // Forget about the old `Drc` because we already cleaned it up (by setting
                // strong count to zero, stealing the `Arc`, and removing the
                // implicit weak pointer).
                mem::forget(swap);
            }

            // At this point we have guaranteed that there are 0 weak pointers associated
            // with the current `Drc`'s `DrcInner`, and that the current `Drc`
            // is the only strong pointer associated with the same `DrcInner`.
            // Therefore, any mutation that `Arc::make_mut` may or may not do
            // to retrieve the value will not be done in a way that allows
            // multiple aliasing of a mutable reference (remember that the `Arc` does not
            // know we have our own strong/weak reference system).
            Arc::make_mut(unsafe { this.arc_mut() })
        }
    }
}

impl<T> Drc<T>
where
    T: ?Sized,
{
    /// Ensure that **local** weak count == 0 and strong count == 1.
    fn is_unique(this: &Drc<T>) -> bool {
        // This only checks the local reference counts, as `Drc::get_mut` has to defer
        // to `Arc::get_mut` anyway.
        Drc::weak_count(this, true) == 0 && Drc::strong_count(this, true) == 1
    }

    /// Create a `Drc` from an `Arc`, without incrementing the atomic reference
    /// counts (as the `Arc` is hijacked by the `Drc` rather than dissected in
    /// some hacky way).
    fn from_arc(arc: Arc<T>) -> Drc<T> {
        unsafe {
            Drc {
                ptr: NonNull::new_unchecked(Box::into_raw(Box::new(DrcInner {
                    strong: Cell::new(1),
                    weak: Cell::new(1),
                    strong_ref: Some(arc),
                    weak_ref: None,
                }))),
                phantom: PhantomData,
            }
        }
    }

    /// Retrieve the raw `Cell` that holds the number of weak pointers
    /// associated with the local `Arc`.
    fn weak(&self) -> &Cell<usize> {
        unsafe { &self.ptr.as_ref().weak }
    }

    /// Retrieve the raw `Cell` that holds the number of strong pointers
    /// associated with the local `Arc`.
    fn strong(&self) -> &Cell<usize> {
        unsafe { &self.ptr.as_ref().strong }
    }

    /// Retrieves the contained `Arc` without having to unwrap the option due to
    /// the fact that we guarantee the `Arc`'s existence by this `Drc`'s
    /// existence.
    fn arc(&self) -> &Arc<T> {
        // Compiler should hopefully easily see this as a no-op, but this is a
        // precaution against an Arc<T> somehow not being subject to null pointer
        // optimization.
        assert_eq!(
            mem::size_of::<Arc<T>>(),
            mem::size_of::<Option<Arc<T>>>(),
            "Error within drc::Drc<T>: Null pointer optimization does not apply to Arc<T>! If you \
             see this panic, please report it to the maintainer(s) of the \"drc\" crate."
        );

        // This is safe because the value will *always* be Some when a strong
        // pointer (i.e. self) exists.
        unsafe { &*(&self.ptr.as_ref().strong_ref as *const _ as *const Arc<T>) }
    }

    /// Ensure that there is 1 local strong pointer and 0 real local weak
    /// pointers before calling this method. Otherwise, this simply takes a
    /// mutable reference to the `Arc` from the `Option` with the same
    /// justification as `arc`.
    unsafe fn arc_mut(&mut self) -> &mut Arc<T> {
        // Compiler should hopefully easily see this as a no-op, but this is a
        // precaution against an Arc<T> somehow not being subject to null pointer
        // optimization.
        assert_eq!(
            mem::size_of::<Arc<T>>(),
            mem::size_of::<Option<Arc<T>>>(),
            "Error within drc::Drc<T>: Null pointer optimization does not apply to Arc<T>! If you \
             see this panic, please report it to the maintainer(s) of the \"drc\" crate."
        );

        &mut *(&mut self.ptr.as_mut().strong_ref as *mut _ as *mut Arc<T>)
    }

    /// Ensure that the strong count is 1 before calling this method OR set it
    /// to 0. Otherwise, this simply removes the `Arc` from the `Option`
    /// with the same justification as `arc`.
    unsafe fn take_arc(&self) -> Arc<T> {
        // Compiler should hopefully easily see this as a no-op, but this is a
        // precaution against an Arc<T> somehow not being subject to null pointer
        // optimization.
        assert_eq!(
            mem::size_of::<Arc<T>>(),
            mem::size_of::<Option<Arc<T>>>(),
            "Error within drc::Drc<T>: Null pointer optimization does not apply to Arc<T>! If you \
             see this panic, please report it to the maintainer(s) of the \"drc\" crate."
        );

        let storage = self.ptr.as_ptr();

        let arc = ptr::read(&(*storage).strong_ref as *const _ as *const Arc<T>);
        ptr::write(&mut (*storage).strong_ref, None);
        arc
    }

    /// This should only be used right after `take_arc`, and only then if the
    /// `Arc` is untouched.
    unsafe fn set_arc(&self, arc: Arc<T>) {
        // Compiler should hopefully easily see this as a no-op, but this is a
        // precaution against an Arc<T> somehow not being subject to null pointer
        // optimization.
        assert_eq!(
            mem::size_of::<Arc<T>>(),
            mem::size_of::<Option<Arc<T>>>(),
            "Error within drc::Drc<T>: Null pointer optimization does not apply to Arc<T>! If you \
             see this panic, please report it to the maintainer(s) of the \"drc\" crate."
        );

        ptr::write(
            &mut (*self.ptr.as_ptr()).strong_ref as *mut _ as *mut Arc<T>,
            arc,
        );
    }
}

impl<T> Clone for Drc<T>
where
    T: ?Sized,
{
    /// Makes a clone of the `Drc` pointer.
    ///
    /// This creates another pointer to the same inner value and associated with
    /// the same contained `Arc`. This increases the local strong reference
    /// count, but does not touch the `Arc` at all.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// let same_five = Drc::clone(&five);
    ///
    /// // `five` and `same_five` share the same `Arc`.
    /// assert!(Drc::linked(&five, &same_five));
    ///
    /// // Local strong reference count of 2:
    /// // `five`, `same_five`
    /// assert_eq!(2, Drc::strong_count(&five, true));
    ///
    /// // `Arc` strong reference count of 1:
    /// // (`five`, `same_five`)
    /// assert_eq!(1, Drc::strong_count(&five, false));
    /// ```
    fn clone(&self) -> Drc<T> {
        self.strong().set(self.strong().get() + 1);
        Drc {
            ptr: self.ptr,
            phantom: PhantomData,
        }
    }
}

impl<T> Drop for Drc<T>
where
    T: ?Sized,
{
    /// Drops the `Drc`.
    ///
    /// This will decrement the local strong reference count. In the case that
    /// this is the last `Drc` associated with the inner `Arc` (i.e. the
    /// local strong reference count reaches zero), the inner `Arc` is
    /// dropped too.
    ///
    /// A local [`Weak`] pointer may still exist, and assuming the value still
    /// persists within the `Arc`'s innards, said local `Weak` pointer might
    /// still be upgradeable even in the case that this is the last local
    /// `Drc`. In that case, the stored weak `Arc` will be upgraded to
    /// repopulate the inner `Arc`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo = Drc::new(Foo);
    /// let foo2 = Drc::clone(&foo);
    ///
    /// drop(foo); // Doesn't print anything
    /// drop(foo2); // Prints "dropped!"
    /// ```
    ///
    /// [`Weak`]: ./struct.Weak.html
    fn drop(&mut self) {
        unsafe {
            let strong = self.strong().get();
            self.strong().set(strong - 1);
            // In the case that this is the last strong pointer, it falls to us to
            // clean up the internal `Arc` as well as the implicit weak pointer.
            if strong == 1
            /* now it is 0 */
            {
                // Drop the contained `Arc`.
                drop(self.take_arc());

                // Clean up our implicit weak pointer.
                let weak = self.weak().get();
                self.weak().set(weak - 1);
                // In the case that that was the last weak pointer, it falls to us to
                // deallocate the `DrcInner`. If there are remaining weak pointers, they
                // can handle it themselves.
                //
                // We can not simply materialize the implicit weak pointer because it
                // will think that the implicit weak pointer is already gone due to
                // strong count being zero, and if we did not decrement strong count,
                // it wouldn't think there was any work to be done when it was dropped.
                if weak == 1
                /* now it is 0 */
                {
                    // Recreate the `Box` that the `DrcInner` was originally allocated as,
                    // so that its `Drop` implementation can run.
                    // Since we use only safe types within the `DrcInner`, this will
                    // work as expected.
                    Box::from_raw(self.ptr.as_ptr());
                }
            }
        }
    }
}

impl<T> AsRef<T> for Drc<T>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T> Borrow<T> for Drc<T>
where
    T: ?Sized,
{
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T> Default for Drc<T>
where
    T: Default,
{
    /// Creates a new `Drc<T>`, with the `Default` value for `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let x: Drc<i32> = Default::default();
    /// assert_eq!(*x, 0);
    /// ```
    fn default() -> Drc<T> {
        Drc::new(Default::default())
    }
}

impl<T> Deref for Drc<T>
where
    T: ?Sized,
{
    type Target = T;

    fn deref(&self) -> &T {
        &**(self.arc())
    }
}

impl<T> Eq for Drc<T>
where
    T: Eq + ?Sized,
{
}

impl From<CString> for Drc<CStr> {
    fn from(c_string: CString) -> Drc<CStr> {
        Drc::from(Arc::from(c_string))
    }
}

impl From<String> for Drc<str> {
    fn from(string: String) -> Drc<str> {
        Drc::from(Arc::from(string))
    }
}

impl From<OsString> for Drc<OsStr> {
    fn from(os_string: OsString) -> Drc<OsStr> {
        Drc::from(Arc::from(os_string))
    }
}

impl From<PathBuf> for Drc<Path> {
    fn from(path: PathBuf) -> Drc<Path> {
        Drc::from(Arc::from(path))
    }
}

impl<'a> From<&'a CStr> for Drc<CStr> {
    fn from(c_string: &'a CStr) -> Drc<CStr> {
        Drc::from(Arc::from(c_string))
    }
}

impl<'a> From<&'a str> for Drc<str> {
    fn from(string: &'a str) -> Drc<str> {
        Drc::from(Arc::from(string))
    }
}

impl<'a> From<&'a OsStr> for Drc<OsStr> {
    fn from(os_string: &'a OsStr) -> Drc<OsStr> {
        Drc::from(Arc::from(os_string))
    }
}

impl<'a> From<&'a Path> for Drc<Path> {
    fn from(path: &'a Path) -> Drc<Path> {
        Drc::from(Arc::from(path))
    }
}

impl<'a, T> From<&'a [T]> for Drc<[T]>
where
    T: Clone,
{
    fn from(slice: &'a [T]) -> Drc<[T]> {
        Drc::from(Arc::from(slice))
    }
}

impl<T> From<Arc<T>> for Drc<T>
where
    T: ?Sized,
{
    fn from(arc: Arc<T>) -> Drc<T> {
        Drc::from_arc(arc)
    }
}

impl<T> From<Box<T>> for Drc<T>
where
    T: ?Sized,
{
    fn from(box_: Box<T>) -> Drc<T> {
        Drc::from(Arc::from(box_))
    }
}

impl<T> From<T> for Drc<T> {
    fn from(value: T) -> Drc<T> {
        Drc::from(Arc::from(value))
    }
}

impl<T> From<Vec<T>> for Drc<[T]> {
    fn from(vec: Vec<T>) -> Drc<[T]> {
        Drc::from(Arc::from(vec))
    }
}

impl<T> Hash for Drc<T>
where
    T: Hash + ?Sized,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        (**self).hash(state);
    }
}

impl<T> Ord for Drc<T>
where
    T: Ord + ?Sized,
{
    /// Comparison for two `Drc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::cmp::Ordering;
    ///
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// assert_eq!(Ordering::Less, five.cmp(&Drc::new(6)));
    /// ```
    #[inline]
    fn cmp(&self, other: &Drc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T> PartialEq<Drc<T>> for Drc<T>
where
    T: PartialEq<T> + ?Sized,
{
    /// Equality for two `Drc`s.
    ///
    /// Two `Drc`s are equal if their inner values are equal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// assert!(five == Drc::new(5));
    /// ```
    fn eq(&self, other: &Drc<T>) -> bool {
        **self == **other
    }

    /// Equality for two `Drc`s.
    ///
    /// Two `Drc`s are unequal if their inner values are unequal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// assert!(five != Drc::new(6));
    /// ```
    fn ne(&self, other: &Drc<T>) -> bool {
        **self != **other
    }
}

impl<T> PartialOrd<Drc<T>> for Drc<T>
where
    T: PartialOrd<T> + ?Sized,
{
    /// Partial comparison for two `Drc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::cmp::Ordering;
    ///
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// assert_eq!(Some(Ordering::Less), five.partial_cmp(&Drc::new(6)));
    /// ```
    #[inline(always)]
    fn partial_cmp(&self, other: &Drc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Drc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// assert!(five < Drc::new(6));
    /// ```
    #[inline(always)]
    fn lt(&self, other: &Drc<T>) -> bool {
        **self < **other
    }

    /// 'Less than or equal to' comparison for two `Drc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// assert!(five <= Drc::new(6));
    /// ```
    #[inline(always)]
    fn le(&self, other: &Drc<T>) -> bool {
        **self <= **other
    }

    /// Greater-than comparison for two `Drc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// assert!(five > Drc::new(4));
    /// ```
    #[inline(always)]
    fn gt(&self, other: &Drc<T>) -> bool {
        **self > **other
    }

    /// 'Greater-than or equal to' comparison for two `Drc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// assert!(five >= Drc::new(4));
    /// ```
    #[inline(always)]
    fn ge(&self, other: &Drc<T>) -> bool {
        **self >= **other
    }
}

impl<T> Debug for Drc<T>
where
    T: Debug + ?Sized,
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{:?}", &**self)
    }
}

impl<T> Display for Drc<T>
where
    T: Display + ?Sized,
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", &**self)
    }
}

impl<T> Pointer for Drc<T>
where
    T: ?Sized,
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{:p}", &**self)
    }
}

impl<T> UnwindSafe for Drc<T>
where
    T: RefUnwindSafe + ?Sized,
{
}

pub(crate) struct DrcInner<T>
where
    T: ?Sized,
{
    pub(crate) strong: Cell<usize>,
    pub(crate) weak: Cell<usize>,
    pub(crate) strong_ref: Option<Arc<T>>,
    pub(crate) weak_ref: Option<WeakArc<T>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let drc = Drc::new(17usize);

        let storage = unsafe { drc.ptr.as_ref() };
        assert_eq!(
            storage.strong.get(),
            1,
            "new Drc did not have strong count of 1."
        );
        assert_eq!(
            storage.weak.get(),
            1,
            "new Drc did not have weak count of 1 (implicit)."
        );

        assert!(
            storage.strong_ref.is_some(),
            "new Drc did not have a stored Arc."
        );
        assert!(storage.weak_ref.is_none(), "new Drc had a stored WeakArc.");

        let strong_ref = storage.strong_ref.as_ref().unwrap();
        assert_eq!(**strong_ref, 17usize, "new Drc stored value incorrectly.");
        assert_eq!(
            Arc::strong_count(strong_ref),
            1,
            "internal Arc had an unexpected strong reference count."
        );
        assert_eq!(
            Arc::weak_count(strong_ref),
            0,
            "internal Arc had an unexpected weak reference count."
        );
    }

    #[test]
    fn detach() {
        let drc = Drc::new(17usize);

        let storage = unsafe { drc.ptr.as_ref() };
        let arc_ptr = &**storage.strong_ref.as_ref().unwrap() as *const _;

        let new_arc = Drc::detach(&drc);

        assert_eq!(
            unsafe { drc.ptr.as_ref() } as *const _,
            storage as *const _,
            "original Drc changed pointer location when detached."
        );
        assert_eq!(
            &*new_arc as *const _, arc_ptr,
            "detached Arc did not have the same pointer value as original Drc."
        );
    }

    #[test]
    fn separate() {
        let drc_1 = Drc::new(17usize);
        let storage_1 = unsafe { drc_1.ptr.as_ref() };
        let data_pointer_1 = &**storage_1.strong_ref.as_ref().unwrap() as *const _;

        let drc_2 = Drc::separate(&drc_1);
        let storage_2 = unsafe { drc_2.ptr.as_ref() };
        let data_pointer_2 = &**storage_2.strong_ref.as_ref().unwrap() as *const _;

        assert_eq!(
            unsafe { drc_1.ptr.as_ref() } as *const _,
            storage_1 as *const _,
            "original Drc changed local storage location when separated."
        );
        assert_eq!(
            &**storage_1.strong_ref.as_ref().unwrap() as *const _,
            data_pointer_1,
            "original Drc changed data pointer location when separated."
        );

        assert_ne!(
            storage_1 as *const _, storage_2 as *const _,
            "new Drc still shared local storage location with original Drc when separated."
        );
        assert_eq!(
            data_pointer_1, data_pointer_2,
            "new Drc did not share data pointer location with original Drc when separated."
        );

        assert_eq!(
            storage_1.strong.get(),
            1,
            "original Drc had a different strong reference count than expected when separated."
        );
        assert_eq!(
            storage_1.weak.get(),
            1,
            "original Drc had a different weak reference count than expected when separated."
        );
        assert!(
            storage_1.weak_ref.is_none(),
            "original Drc had a WeakArc stored when separated."
        );

        assert_eq!(
            storage_2.strong.get(),
            1,
            "new Drc had a different strong reference count than expected."
        );
        assert_eq!(
            storage_2.weak.get(),
            1,
            "new Drc had a different weak reference count than expected."
        );
        assert!(
            storage_2.weak_ref.is_none(),
            "new Drc had a WeakArc stored."
        );
    }

    // Duplicate of the documentation example because it sufficiently tests the
    // method, but it's still repeated here in case the docs change.
    #[test]
    fn try_unwrap() {
        // The `Drc` here is the only strong reference to 3, so it is successfully
        // unwrapped.
        let x = Drc::new(3);
        assert_eq!(Drc::try_unwrap(x), Ok(3));

        // There are two `Drc` strong references to 4, so it is not successfully
        // unwrapped.
        let x = Drc::new(4);
        let _y = Drc::clone(&x);
        assert_eq!(*Drc::try_unwrap(x).unwrap_err(), 4);

        // There is a `Drc` and an `Arc` strong reference to 5, so it is not
        // sucessfully unwrapped.
        let x = Drc::new(5);
        let _y = Drc::detach(&x);
        assert_eq!(*Drc::try_unwrap(x).unwrap_err(), 5);
    }

    #[test]
    fn downgrade() {
        let drc = Drc::new(17usize);
        let drc_storage = unsafe { drc.ptr.as_ref() };

        let weak = Drc::downgrade(&drc);
        let weak_storage = unsafe { weak.ptr.as_ref() };

        assert_eq!(
            unsafe { drc.ptr.as_ref() } as *const _,
            drc_storage as *const _,
            "Drc changed local storage location when downgraded."
        );
        assert_eq!(
            drc_storage as *const _, weak_storage as *const _,
            "Drc did not share a storage location with new, weak Drc when downgraded."
        );

        assert!(
            drc_storage.strong_ref.is_some(),
            "Drc storage lost its strong reference when downgraded."
        );
        assert!(
            drc_storage.weak_ref.is_some(),
            "Drc storage did not gain a weak reference when downgraded."
        );

        assert_eq!(
            drc_storage.strong.get(),
            1,
            "Drc had a different strong reference count than expected when downgraded."
        );
        assert_eq!(
            drc_storage.weak.get(),
            2,
            "Drc had a different weak reference count than expected when downgraded."
        );

        let strong_ref = drc_storage.strong_ref.as_ref().unwrap();

        assert_eq!(
            Arc::strong_count(strong_ref),
            1,
            "internal Arc had an unexpected strong reference count when Drc downgraded."
        );
        assert_eq!(
            Arc::weak_count(strong_ref),
            1,
            "internal Arc had an unexpected weak reference count when Drc downgraded."
        );

        let _weak_2 = Drc::downgrade(&drc);

        assert_eq!(
            unsafe { drc.ptr.as_ref() } as *const _,
            drc_storage as *const _,
            "original Drc changed local storage location when downgraded a second time."
        );

        assert!(
            drc_storage.strong_ref.is_some(),
            "Drc storage lost its strong reference when downgraded a second time."
        );
        assert!(
            drc_storage.weak_ref.is_some(),
            "Drc storage did not gain a weak reference when downgraded a second time."
        );

        assert_eq!(
            drc_storage.strong.get(),
            1,
            "Drc had a different strong reference count than expected when downgraded a second \
             time."
        );
        assert_eq!(
            drc_storage.weak.get(),
            3,
            "Drc had a different weak reference count than expected when downgraded a second time."
        );

        assert_eq!(
            drc_storage.strong_ref.as_ref().unwrap() as *const _,
            strong_ref as *const _,
            "Drc had a different internal strong Arc when downgraded a second time."
        );

        assert_eq!(
            Arc::strong_count(strong_ref),
            1,
            "internal Arc had an unexpected strong reference count when Drc downgraded a second \
             time."
        );
        assert_eq!(
            Arc::weak_count(strong_ref),
            1,
            "internal Arc had an unexpected weak reference count when Drc downgraded a second \
             time."
        );
    }

    // Duplicate of the documentation example because it sufficiently tests the
    // method, but it's still repeated here in case the docs change.
    #[test]
    fn weak_count() {
        let five = Drc::new(5);
        let _weak_five_a = Drc::downgrade(&five);
        let _weak_five_b = Drc::downgrade(&five);
        let _weak_five_c = Drc::downgrade(&five);

        // No contribution because no weak pointers.
        let _separate_five = Drc::separate(&five);

        // detached_five is an Arc that points to the same value.
        let detached_five = Drc::detach(&five);
        let _weak_detached_five = Arc::downgrade(&detached_five);

        // 3 values:
        // _weak_five_a, _weak_five_b, _weak_five_c
        assert_eq!(3, Drc::weak_count(&five, true));

        // 2 values:
        // (_weak_five_a, _weak_five_b, _weak_five_c), _weak_detached_five
        assert_eq!(2, Drc::weak_count(&five, false));
    }

    // Duplicate of the documentation example because it sufficiently tests the
    // method, but it's still repeated here in case the docs change.
    #[test]
    fn strong_count() {
        let five = Drc::new(5);
        let _also_five = Drc::clone(&five);
        let _still_five = Drc::clone(&five);

        // No contribution because no strong pointer.
        let _weak_separate_five = {
            let separate_five = Drc::separate(&five);
            Drc::downgrade(&separate_five)
        };

        // This is basically a glorified Arc, basically (Arc,)
        let _strong_separate_five = Drc::separate(&five);

        // detached_five is an Arc that points to the same value.
        let detached_five = Drc::detach(&five);
        let _also_detached_five = Arc::clone(&detached_five);

        // 3 values:
        // five, _also_five, _still_five
        assert_eq!(3, Drc::strong_count(&five, true));

        // 4 values:
        // (five, _also_five, _still_five), (_strong_separate_five,), detached_five,
        //     _also_detached_five
        assert_eq!(4, Drc::strong_count(&five, false));
    }

    #[test]
    fn get_mut() {
        // Test case for intended way of working.
        {
            let mut drc = Drc::new(17usize);
            *Drc::get_mut(&mut drc).unwrap() = 117usize;

            assert_eq!(*drc, 117usize, "Intended Drc get_mut mutation failed.");
        }

        // Test case for restriction "No other `Drc`s"
        {
            // LOCAL
            {
                let mut drc = Drc::new(17usize);
                let mut drc_clone = Drc::clone(&drc);

                assert!(
                    Drc::get_mut(&mut drc).is_none(),
                    "Drc::get_mut returned Some(_) when another local Drc existed."
                );
                assert!(
                    Drc::get_mut(&mut drc_clone).is_none(),
                    "Drc::get_mut returned Some(_) on Drc clone when original Drc existed."
                );
            }

            // NONLOCAL
            {
                let mut drc = Drc::new(17usize);
                let mut drc_separate = Drc::separate(&drc);

                assert!(
                    Drc::get_mut(&mut drc).is_none(),
                    "Drc::get_mut returned Some(_) when another nonlocal Drc existed."
                );
                assert!(
                    Drc::get_mut(&mut drc_separate).is_none(),
                    "Drc::get_mut returned Some(_) on separate Drc when original Drc existed."
                );
            }
        }

        // Test case for restriction "No other `Arc`s"
        {
            let mut drc = Drc::new(17usize);
            let _arc = Drc::detach(&drc);

            assert!(
                Drc::get_mut(&mut drc).is_none(),
                "Drc::get_mut returned Some(_) when another Arc existed."
            );
        }

        // Test case for restriction "No other weak `Drc`s"
        {
            // LOCAL
            {
                let mut drc = Drc::new(17usize);
                let weak_drc = Drc::downgrade(&drc);

                assert!(
                    Drc::get_mut(&mut drc).is_none(),
                    "Drc::get_mut returned Some(_) when a local weak Drc existed."
                );
                assert!(
                    Drc::get_mut(&mut Weak::upgrade(&weak_drc).unwrap()).is_none(),
                    "Drc::get_mut returned Some(_) on upgraded local weak Drc when original Drc \
                     existed."
                );
            }

            // NONLOCAL
            {
                let mut drc = Drc::new(17usize);
                let weak_drc = Drc::downgrade(&Drc::separate(&drc));

                assert!(
                    Drc::get_mut(&mut drc).is_none(),
                    "Drc::get_mut returned Some(_) when a nonlocal Drc existed."
                );
                assert!(
                    Drc::get_mut(&mut Weak::upgrade(&weak_drc).unwrap()).is_none(),
                    "Drc::get_mut returned Some(_) on upgraded nonlocal weak Drc when original \
                     Drc existed."
                );
            }
        }

        // Test case for restriction "No other weak `Arc`s"
        {
            // DETACH THEN DOWNGRADE
            {
                let mut drc = Drc::new(17usize);
                let weak_arc = Arc::downgrade(&Drc::detach(&drc));

                assert!(
                    Drc::get_mut(&mut drc).is_none(),
                    "Drc::get_mut returned Some(_) when a weak Arc existed."
                );
                assert!(
                    Arc::get_mut(&mut WeakArc::upgrade(&weak_arc).unwrap()).is_none(),
                    "Arc::get_mut returned Some(_) on upgraded weak Arc when original Drc existed."
                );
            }

            // DOWNGRADE THEN DETACH
            {
                let mut drc = Drc::new(17usize);
                let weak_arc = Weak::detach(&Drc::downgrade(&drc));

                assert!(
                    Drc::get_mut(&mut drc).is_none(),
                    "Drc::get_mut returned Some(_) when a weak Arc existed."
                );
                assert!(
                    Arc::get_mut(&mut WeakArc::upgrade(&weak_arc).unwrap()).is_none(),
                    "Arc::get_mut returned Some(_) on upgraded weak Arc when original Drc existed."
                );
            }
        }
    }

    #[test]
    fn ptr_eq() {
        // Test case for separate Drcs created from Arcs.
        {
            // EQUAL
            {
                let (drc_1, drc_2): (Drc<usize>, Drc<usize>) = {
                    let arc = Arc::new(17usize);
                    let drc_1 = Drc::from(Arc::clone(&arc));
                    (drc_1, Drc::from(arc))
                };

                assert!(
                    Drc::ptr_eq(&drc_1, &drc_2),
                    "Drcs created from two associated Arcs were not pointer equal."
                );
            }

            // NOT EQUAL
            {
                let drc_1: Drc<usize> = Drc::from(Arc::new(17usize));
                let drc_2: Drc<usize> = Drc::from(Arc::new(17usize));

                assert!(
                    !Drc::ptr_eq(&drc_1, &drc_2),
                    "Drcs created from two separate Arcs were pointer equal."
                );
            }
        }

        // Test case for separate Drcs created manually with detach and from.
        {
            // EQUAL
            {
                let drc_1: Drc<usize> = Drc::new(17usize);
                let drc_2: Drc<usize> = Drc::from(Drc::detach(&drc_1));

                assert!(
                    Drc::ptr_eq(&drc_1, &drc_2),
                    "Separate Drcs created manually with 'detach' and 'from' were not pointer \
                     equal."
                );
            }

            // NOT EQUAL
            // NOT APPLICABLE
        }

        // Test case for separate Drcs created with separate.
        {
            // EQUAL
            {
                let drc_1 = Drc::new(17usize);
                let drc_2 = Drc::separate(&drc_1);

                assert!(
                    Drc::ptr_eq(&drc_1, &drc_2),
                    "Separate Drcs created with 'separate' were not pointer equal."
                );
            }

            // NOT EQUAL
            // NOT APPLICABLE
        }

        // Test case for two local Drcs.
        {
            // EQUAL
            {
                let drc_1 = Drc::new(17usize);
                let drc_2 = Drc::clone(&drc_1);

                assert!(
                    Drc::ptr_eq(&drc_1, &drc_2),
                    "Linked Drcs created with 'clone' were not pointer equal."
                );
            }

            // NOT EQUAL
            {
                let drc_1 = Drc::new(17usize);
                let drc_2 = Drc::new(17usize);

                assert!(
                    !Drc::ptr_eq(&drc_1, &drc_2),
                    "Drcs created using two separate 'new' calls were pointer equal."
                );
            }
        }
    }

    #[test]
    fn arc_ptr_eq() {
        // Test case for an Arc and Drc created from an Arc.
        {
            // EQUAL
            {
                let arc: Arc<usize> = Arc::new(17usize);
                let drc: Drc<usize> = Drc::from(Arc::clone(&arc));

                assert!(
                    Drc::arc_ptr_eq(&drc, &arc),
                    "An Arc and a Drc created from the Arc's clone were not pointer equal."
                );
            }

            // NOT EQUAL
            {
                let arc: Arc<usize> = Arc::new(17usize);
                let drc: Drc<usize> = Drc::from(Arc::new(17usize));

                assert!(
                    !Drc::arc_ptr_eq(&drc, &arc),
                    "An Arc and a Drc created from a different Arc were pointer equal."
                );
            }
        }

        // Test case for an Arc detached from a local Drc, and a local Drc.
        {
            // EQUAL
            {
                let drc = Drc::new(17usize);
                let arc = Drc::detach(&drc);

                assert!(
                    Drc::arc_ptr_eq(&drc, &arc),
                    "A Drc and an Arc detached from the Drc were not pointer equal."
                );
            }

            // NOT EQUAL
            {
                let drc = Drc::new(17usize);
                let arc = Drc::detach(&Drc::new(17usize));

                assert!(
                    !Drc::arc_ptr_eq(&drc, &arc),
                    "A Drc and an Arc detached from a different Drc were pointer equal"
                );
            }
        }

        // Test case for an Arc detached from a separated Drc, and a local Drc.
        {
            // EQUAL
            {
                let drc = Drc::new(17usize);
                let arc = Drc::detach(&Drc::separate(&drc));

                assert!(
                    Drc::arc_ptr_eq(&drc, &arc),
                    "A Drc and an Arc detached from a Drc separated from the original Drc were \
                     not pointer equal."
                );
            }

            // NOT EQUAL
            {
                let drc = Drc::new(17usize);
                let arc = Drc::detach(&Drc::separate(&Drc::new(17usize)));

                assert!(
                    !Drc::arc_ptr_eq(&drc, &arc),
                    "A Drc and an Arc detached from a Drc separated from a different Drc were \
                     pointer equal."
                );
            }
        }

        // Test case for an Arc detached from a cloned local Drc, and a local Drc.
        {
            // EQUAL
            {
                let drc = Drc::new(17usize);
                let arc = Drc::detach(&Drc::clone(&drc));

                assert!(
                    Drc::arc_ptr_eq(&drc, &arc),
                    "A Drc and an Arc detached from a clone of the Drc were not pointer equal."
                );
            }

            // NOT EQUAL
            {
                let drc = Drc::new(17usize);
                let arc = Drc::detach(&Drc::clone(&Drc::new(17usize)));

                assert!(
                    !Drc::arc_ptr_eq(&drc, &arc),
                    "A Drc and an Arc detached from a clone of a different Drc were pointer equal."
                );
            }
        }
    }

    #[test]
    fn linked() {
        // Test case for two local Drcs
        {
            let drc_1 = Drc::new(17usize);
            let drc_2 = Drc::clone(&drc_1);

            assert!(
                Drc::linked(&drc_1, &drc_2),
                "Two Drcs that were the strict definition of 'linked' were not considered linked."
            );
        }

        // Test case for Drcs separated due to being created from Arcs.
        {
            // Associated Arcs
            {
                let (drc_1, drc_2): (Drc<usize>, Drc<usize>) = {
                    let arc = Arc::new(17usize);
                    let drc_1 = Drc::from(Arc::clone(&arc));
                    (drc_1, Drc::from(arc))
                };

                assert!(
                    !Drc::linked(&drc_1, &drc_2),
                    "Two Drcs separated due to being created from two different but associated \
                     Arcs were considered linked."
                );
            }

            // Not Associated Arcs
            {
                let drc_1: Drc<usize> = Drc::from(Arc::new(17usize));
                let drc_2: Drc<usize> = Drc::from(Arc::new(17usize));

                assert!(
                    !Drc::linked(&drc_1, &drc_2),
                    "Two Drcs separated due to being created from two different and unassociated \
                     Arcs were considered linked."
                );
            }
        }

        // Test case for Drcs separated due to one being detached and then converted
        // manually.
        {
            let drc_1: Drc<usize> = Drc::new(17usize);
            let drc_2: Drc<usize> = Drc::from(Drc::detach(&drc_1));

            assert!(
                !Drc::linked(&drc_1, &drc_2),
                "Two Drcs separated due to one being detached from the other and then converted \
                 back into a Drc were considered linked."
            );
        }

        // Test case for Drcs separated due to the 'separate' method.
        {
            let drc_1 = Drc::new(17usize);
            let drc_2 = Drc::separate(&drc_1);

            assert!(
                !Drc::linked(&drc_1, &drc_2),
                "Two Drcs separated with the 'separate' method were considered linked."
            );
        }

        // Test case for two separately created Drcs.
        {
            // Using 'new'
            {
                let drc_1 = Drc::new(17usize);
                let drc_2 = Drc::new(17usize);

                assert!(
                    !Drc::linked(&drc_1, &drc_2),
                    "Two Drcs separated due to being made with separate 'new' calls were \
                     considered linked."
                );
            }

            // One using 'new', one using 'Drc::from(arc)'
            {
                let drc_1: Drc<usize> = Drc::new(17usize);
                let drc_2: Drc<usize> = Drc::from(Arc::new(17usize));

                assert!(
                    !Drc::linked(&drc_1, &drc_2),
                    "Two Drcs separated due to one being made with 'new' and the other made with \
                     'Drc::from(arc)' were considered linked."
                );
            }
        }
    }

    // This is pretty much a slightly altered test for get_mut.
    #[test]
    fn make_mut() {
        // Test case for way of working that does not alter the Drc.
        {
            let mut drc = Drc::new(17usize);
            let storage = drc.ptr.as_ptr();

            *Drc::get_mut(&mut drc).unwrap() = 117usize;

            assert_eq!(*drc, 117usize, "Intended Drc make_mut mutation failed.");

            assert_eq!(
                storage,
                drc.ptr.as_ptr(),
                "Drc storage changed with make_mut when it should not have."
            );
        }

        // Test case for behavior with another Drc. This will alter the original Drc or
        // its storage in some way.
        {
            // LOCAL
            {
                let mut drc = Drc::new(17usize);
                let mut drc_clone = Drc::clone(&drc);

                let storage = drc.ptr.as_ptr();

                let val_ptr = &*drc as *const _;

                *Drc::make_mut(&mut drc) = 117usize;

                assert_eq!(*drc, 117usize, "Intended Drc make_mut mutation failed.");
                assert_eq!(*drc_clone, 17usize, "Drc make_mut mutation affected clone.");

                assert_ne!(
                    storage,
                    drc.ptr.as_ptr(),
                    "Drc make_mut did not change storage when it should have."
                );
                assert_eq!(
                    storage,
                    drc_clone.ptr.as_ptr(),
                    "Drc make_mut changed storage for clone when it should not have."
                );

                assert_eq!(
                    unsafe { &**(*storage).strong_ref.as_ref().unwrap() } as *const _,
                    val_ptr,
                    "Drc make_mut changed original storage's Arc pointer when it should not have."
                );

                *Drc::make_mut(&mut drc_clone) = 1117usize;

                assert_eq!(
                    *drc_clone, 1117usize,
                    "Intended Drc clone make_mut mutation failed."
                );

                assert_eq!(
                    storage,
                    drc_clone.ptr.as_ptr(),
                    "Drc clone make_mut changed storage when it should not have."
                );
            }

            // NONLOCAL
            {
                let mut drc = Drc::new(17usize);
                let mut drc_separate = Drc::separate(&drc);

                let original_storage = drc.ptr.as_ptr();
                let separate_storage = drc_separate.ptr.as_ptr();

                let original_storage_ref = unsafe { (&*original_storage) };
                let separate_storage_ref = unsafe { (&*separate_storage) };

                let val_ptr = &*drc as *const _;

                assert!(
                    Arc::ptr_eq(
                        original_storage_ref.strong_ref.as_ref().unwrap(),
                        separate_storage_ref.strong_ref.as_ref().unwrap()
                    ),
                    "Values should be linked."
                );

                *Drc::make_mut(&mut drc) = 117usize;

                assert_eq!(*drc, 117usize, "Intended Drc make_mut mutation failed.");
                assert_eq!(
                    *drc_separate, 17usize,
                    "Drc make_mut mutation affected separated Drc."
                );

                assert_eq!(
                    original_storage,
                    drc.ptr.as_ptr(),
                    "Drc make_mut changed storage when it should not have."
                );
                assert_eq!(
                    separate_storage,
                    drc_separate.ptr.as_ptr(),
                    "Drc make_mut changed storage for separate Drc when it should not have."
                );

                {
                    let original_storage_arc = original_storage_ref.strong_ref.as_ref().unwrap();
                    let separate_storage_arc = separate_storage_ref.strong_ref.as_ref().unwrap();
                    assert!(
                        !Arc::ptr_eq(original_storage_arc, separate_storage_arc),
                        "Drc make_mut did not change Arcs."
                    );

                    assert_eq!(
                        &*drc_separate as *const _, val_ptr,
                        "Drc make_mut changed value pointer for separate Drc when it should not \
                         have."
                    );

                    assert_eq!(
                        Arc::strong_count(original_storage_arc),
                        1,
                        "Drc make_mut did not update strong count properly."
                    );
                    assert_eq!(
                        Arc::strong_count(separate_storage_arc),
                        1,
                        "Drc make_mut did not update strong count properly."
                    );
                }

                *Drc::make_mut(&mut drc_separate) = 1117usize;

                assert_eq!(
                    *drc_separate, 1117usize,
                    "Intended separate Drc make_mut mutation failed."
                );

                assert_eq!(
                    separate_storage,
                    drc_separate.ptr.as_ptr(),
                    "Separate Drc make_mut changed storage when it should not have."
                );

                assert_eq!(
                    &*drc_separate as *const _, val_ptr,
                    "Separate Drc make_mut changed value pointer when it should not have."
                );
            }
        }

        // Test case for behavior with another Arc. This will alter the original Drc or
        // its storage in some way.
        {
            // WITHOUT LOCAL WEAK DRC (does not alter storage)
            {
                let mut drc = Drc::new(17usize);
                let mut arc = Drc::detach(&drc);

                let original_storage = drc.ptr.as_ptr();

                let val_ptr = &*drc as *const _;

                assert!(
                    Drc::arc_ptr_eq(&drc, &arc),
                    "Values should be pointer equal."
                );

                *Drc::make_mut(&mut drc) = 117usize;

                assert_eq!(*drc, 117usize, "Intended Drc make_mut mutation failed.");
                assert_eq!(
                    *arc, 17usize,
                    "Drc make_mut mutation affected detached Arc."
                );

                assert_eq!(
                    original_storage,
                    drc.ptr.as_ptr(),
                    "Drc make_mut mutation changed storage when it should not have."
                );

                assert_eq!(
                    &*arc as *const _, val_ptr,
                    "Drc make_mut mutation changed Arc value pointer when it should not have."
                );

                *Arc::make_mut(&mut arc) = 1117usize;

                assert_eq!(
                    *arc, 1117usize,
                    "Intended detached Arc make_mut mutation failed."
                );

                assert_eq!(
                    &*arc as *const _, val_ptr,
                    "Detached Arc make_mut mutation changed value pointer when it should not have."
                );
            }

            // WITH LOCAL WEAK DRC (alters storage)
            {
                let mut drc = Drc::new(17usize);
                let arc = Drc::detach(&drc);

                let original_storage = drc.ptr.as_ptr();

                let val_ptr = &*drc as *const _;

                assert!(
                    Drc::arc_ptr_eq(&drc, &arc),
                    "Values should be pointer equal."
                );

                let drc = {
                    let weak = Drc::downgrade(&drc);

                    *Drc::make_mut(&mut drc) = 117usize;

                    assert_eq!(*drc, 117usize, "Intended Drc make_mut mutation failed.");
                    assert_eq!(
                        *arc, 17usize,
                        "Drc make_mut mutation affected detached Arc."
                    );

                    assert_ne!(
                        original_storage,
                        drc.ptr.as_ptr(),
                        "Drc make_mut mutation did not change storage when it should have."
                    );
                    assert_eq!(
                        original_storage,
                        weak.ptr.as_ptr(),
                        "Drc make_mut mutation changed weak pointer storage when it should not \
                         have."
                    );

                    assert_eq!(
                        &*arc as *const _, val_ptr,
                        "Drc make_mut mutation changed Arc value pointer when it should not have."
                    );

                    weak.upgrade().unwrap()
                };

                assert_eq!(
                    &*drc as *const _, val_ptr,
                    "Drc make_mut mutation altered (now-upgraded) weak pointer pointer location."
                );

                // The remaining functionality should be the same as without weak Drc.
            }
        }

        // Test case for behavior with a weak Drc. This will alter the original Drc or
        // its storage in some way.
        {
            // LOCAL (alters storage and value pointer)
            {
                let mut drc = Drc::new(17usize);

                let original_storage = drc.ptr.as_ptr();

                let val_ptr = &*drc as *const _;

                let weak = Drc::downgrade(&drc);

                *Drc::make_mut(&mut drc) = 117usize;

                assert_eq!(*drc, 117usize, "Intended Drc make_mut mutation failed.");

                assert_ne!(
                    original_storage,
                    drc.ptr.as_ptr(),
                    "Drc make_mut mutation did not change storage when it should have."
                );
                assert_eq!(
                    original_storage,
                    weak.ptr.as_ptr(),
                    "Drc make_mut mutation changed weak pointer storage when it should not have."
                );

                assert_ne!(
                    &*drc as *const _, val_ptr,
                    "Drc make_mut mutation did not change value pointer when it should have."
                );

                assert!(
                    weak.upgrade().is_none(),
                    "Drc make_mut mutation should have removed the only strong pointer to the \
                     original value, thus invalidating the weak pointer."
                );
            }

            // NONLOCAL (does not alter storage but alters value pointer)
            {
                for i in 0usize..3usize {
                    let mut drc = Drc::new(17usize);

                    let original_storage = drc.ptr.as_ptr();

                    let val_ptr = &*drc as *const _;

                    let weak = match i {
                        0 => Drc::downgrade(&Drc::separate(&drc)),
                        1 => Drc::downgrade(&Drc::from(Drc::detach(&drc))),
                        2 => Weak::with_weak_arc(Arc::downgrade(&Drc::detach(&drc))),
                        _ => unreachable!(),
                    };

                    let separate_storage = weak.ptr.as_ptr();

                    *Drc::make_mut(&mut drc) = 117usize;

                    assert_eq!(*drc, 117usize, "Intended Drc make_mut mutation failed.");

                    assert_eq!(
                        original_storage,
                        drc.ptr.as_ptr(),
                        "Drc make_mut mutation changed storage when it should not have."
                    );
                    assert_eq!(
                        separate_storage,
                        weak.ptr.as_ptr(),
                        "Drc make_mut mutation changed weak pointer storage when it should not \
                         have."
                    );

                    assert_ne!(
                        &*drc as *const _, val_ptr,
                        "Drc make_mut mutation did not change value pointer when it should have."
                    );

                    assert!(
                        weak.upgrade().is_none(),
                        "Drc make_mut mutation should have removed the only strong pointer to the \
                         original value, thus invalidating the weak pointer."
                    );
                }
            }
        }

        // Test case for behavior with a weak Arc. This will alter the original Drc's
        // storage, but not it itself.
        {
            // Same code, but different ways of creating a weak Arc each time.
            for i in 0usize..2usize {
                let mut drc = Drc::new(17usize);

                let original_storage = drc.ptr.as_ptr();

                let val_ptr = &*drc as *const _;

                let weak = match i {
                    0 => Arc::downgrade(&Drc::detach(&drc)),
                    1 => Drc::downgrade(&drc).detach(),
                    _ => unreachable!(),
                };

                *Drc::make_mut(&mut drc) = 117usize;

                assert_eq!(*drc, 117usize, "Intended Drc make_mut mutation failed.");

                assert_eq!(
                    original_storage,
                    drc.ptr.as_ptr(),
                    "Drc make_mut mutation changed storage when it should not have."
                );

                assert_ne!(
                    &*drc as *const _, val_ptr,
                    "Drc make_mut mutation did not change value pointer when it should have."
                );

                assert!(
                    weak.upgrade().is_none(),
                    "Drc make_mut mutation should have removed the only strong pointer to the \
                     original value, thus invalidating the weak pointer."
                );
            }
        }
    }

    // Duplicate of the documentation example because it sufficiently tests the
    // method, but it's still repeated here in case the docs change.
    #[test]
    fn clone() {
        let five = Drc::new(5);

        let same_five = Drc::clone(&five);

        // `five` and `same_five` share the same `Arc`.
        assert!(Drc::linked(&five, &same_five));

        // Local strong reference count of 2:
        // `five`, `same_five`
        assert_eq!(2, Drc::strong_count(&five, true));

        // `Arc` strong reference count of 1:
        // (`five`, `same_five`)
        assert_eq!(1, Drc::strong_count(&five, false));
    }

    // Test inspired and not much more complex than the documentation example,
    // because that was considered a sufficient test. The only reason edits were
    // needed was because checking println! output is pointless when we could just
    // set a bool somewhere.
    //
    // Note that this is only considered a sufficient test because the side effects
    // of the dropping are sufficiently tested in the other code.
    #[test]
    fn drop() {
        struct Foo<'a>(&'a Cell<bool>);

        impl<'a> Drop for Foo<'a> {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let cell = Cell::new(false);

        let foo = Drc::new(Foo(&cell));
        let foo2 = Drc::clone(&foo);

        // Cell is currently false.
        assert_eq!(cell.get(), false);

        // Cell is not set to true since Foo::drop is not run.
        mem::drop(foo);
        assert_eq!(cell.get(), false);

        // Cell is set to true here since Foo::drop is run.
        mem::drop(foo2);
        assert_eq!(cell.get(), true);
    }

    #[test]
    fn debug() {
        let drc = Drc::new(17usize);

        assert_eq!(
            format!("{:?}", &*drc),
            format!("{:?}", &drc),
            "Debug outputs of actual value and Drc are not equal!"
        );
    }

    #[test]
    fn display() {
        let drc = Drc::new(17usize);

        assert_eq!(
            format!("{}", &*drc),
            format!("{}", &drc),
            "Display outputs of actual value and Drc are not equal!"
        );
    }

    #[test]
    fn pointer() {
        let drc = Drc::new(17usize);

        assert_eq!(
            format!("{:p}", &*drc),
            format!("{:p}", drc),
            "Pointer outputs of smart value and Drc are not equal!"
        );
    }
}
