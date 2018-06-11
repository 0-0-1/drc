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

use std::cell::Cell;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::Weak as WeakArc;

use drc::{Drc, DrcInner};

/// Weak is a version of [`Drc`] that holds a non-owning reference to the
/// managed value. The value is accessed by calling [`upgrade`] on the `Weak`
/// pointer, which returns an [`Option`][`Option`] `<` `Drc<T>` `>`.
///
/// Since a `Weak` reference does not count towards ownership, it will not
/// prevent the inner value from being dropped, and the `Weak` itself makes no
/// guarantees about the value still being present and may return
/// [`None`][`Option`] when [`upgrade`]d.
///
/// A `Weak` pointer is useful for keeping a temporary reference to the value
/// within `Drc` without extending its lifetime. It is also used to prevent
/// circular references between `Drc` pointers, since mutual owning references
/// would never allow either `Drc` to be dropped. However, this use is
/// generally not recommended for `Drc`s because "interior" `Drc`s would be
/// inaccessible and remove `Send` and `Sync` from the type, meaning an [`Rc`]
/// would be just as good for the task.
///
/// Something that might work better is using [`Arc`]s and
/// [weak `Arc`][`WeakArc`]s within the structure, with the outermost reference
/// obviously having the potential to be converted into a `Drc` or `Weak` `Drc`
/// (i.e. this `Weak`).
///
/// The typical way to obtain a `Weak` pointer is to call [`Drc::downgrade`].
/// Using [`Weak::with_weak_arc`][`with_weak_arc`] is not recommended for most
/// cases, as this performs unnecessary allocation without even knowing if the
/// pointed-to value is still there!
///
/// [`Drc`]: ./struct.Drc.html
///
/// [`Drc::downgrade`]: ./struct.Drc.html#method.downgrade
/// [`upgrade`]: ./struct.Weak.html#method.upgrade
/// [`with_weak_arc`]: ./struct.Weak.html#method.with_weak_arc
///
/// [`Option`]: https://doc.rust-lang.org/std/option/enum.Option.html
/// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
/// [`WeakArc`]: https://doc.rust-lang.org/std/sync/struct.Weak.html
/// [`Rc`]: https://doc.rust-lang.org/std/rc/struct.Rc.html
pub struct Weak<T>
where
    T: ?Sized,
{
    pub(crate) ptr: NonNull<DrcInner<T>>,
}

impl<T> Weak<T> {
    /// Constructs a new `Weak<T>`, allocating memory for not only `T` and its
    /// atomic reference counts but also the local reference counts. Calling
    /// [`upgrade`] on the return value always gives [`None`][`Option`].
    ///
    /// Exercise restraint when using this method; if possible it is recommended
    /// to simply use the [weak `Arc`'s `new` method][`WeakArc::new`] for
    /// whatever purpose this `new` method would fulfill.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Weak;
    ///
    /// let empty: Weak<i64> = Weak::new();
    /// assert!(empty.upgrade().is_none());
    /// ```
    ///
    /// [`upgrade`]: ./struct.Weak.html#method.upgrade
    ///
    /// [`Option`]: https://doc.rust-lang.org/std/option/enum.Option.html
    ///
    /// [`WeakArc::new`]: https://doc.rust-lang.org/std/sync/struct.Weak.html#method.new
    pub fn new() -> Weak<T> {
        Weak::with_weak_arc(WeakArc::new())
    }
}

impl<T> Weak<T>
where
    T: ?Sized,
{
    /// Constructs a new `Weak` pointer, allocating memory for local reference
    /// counts, but using a pre-constructed [weak `Arc`][`WeakArc`]. Calling
    /// [`upgrade`] on the return value may or may not return a [`Drc`]. If the
    /// [`Drc`] can be created, it will make use of this allocated memory.
    ///
    /// Exercise restraint when using this method; if possible it is recommended
    /// to simply use the [weak `Arc`][`WeakArc`] for whatever purpose this
    /// `Weak` would fulfill.
    ///
    /// This method was chosen over an implementation of the [`from`] method
    /// for a reverse [`detach`ment][`detach`] to make it clear that care
    /// should be taken when using this operation instead of alternatives.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use drc::{Drc, Weak};
    ///
    /// let five = Arc::new(5);
    ///
    /// let weak_five = Weak::with_weak_arc(Arc::downgrade(&five));
    ///
    /// let strong_five: Option<Drc<_>> = weak_five.upgrade();
    /// assert!(strong_five.is_some());
    ///
    /// // This is not a recommended use of `with_weak_arc`, as the same could be
    /// // done with `Arc`'s `Weak` type.
    /// ```
    ///
    /// [`Drc`]: ./struct.Drc.html
    ///
    /// [`upgrade`]: ./struct.Weak.html#method.upgrade
    /// [`detach`]: ./struct.Weak.html#method.detach
    ///
    /// [`WeakArc`]: https://doc.rust-lang.org/std/sync/struct.Weak.html
    ///
    /// [`from`]: https://doc.rust-lang.org/std/convert/trait.From.html#tymethod.from
    pub fn with_weak_arc(weak_arc: WeakArc<T>) -> Weak<T> {
        unsafe {
            Weak {
                ptr: NonNull::new_unchecked(Box::into_raw(Box::new(DrcInner {
                    strong: Cell::new(0),
                    weak: Cell::new(1),
                    strong_ref: None,
                    weak_ref: Some(weak_arc),
                }))),
            }
        }
    }

    /// Attempts to upgrade the `Weak` pointer to a [`Drc`], extending the
    /// lifetime of the value if successful.
    ///
    /// Returns [`None`][`Option`] if the value has since been dropped.
    ///
    /// Note that even if there are no `Drc` strong pointers associated with the
    /// same [`Arc`], the value may still persist. Since this `Weak` associates
    /// with a [weak `Arc`][`WeakArc`], meaning that a `Drc` may still be
    /// returned even if the local strong pointer count is 0 (because a new
    /// `Arc` can be generated by upgrading the internal weak `Arc`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// // This is an Arc associated to the same value.
    /// let detached_five = Drc::detach(&five);
    ///
    /// let weak_five = Drc::downgrade(&five);
    ///
    /// let strong_five: Option<Drc<_>> = weak_five.upgrade();
    /// assert!(strong_five.is_some());
    ///
    /// drop(strong_five);
    /// drop(five);
    ///
    /// let strong_five: Option<Drc<_>> = weak_five.upgrade();
    /// assert!(strong_five.is_some());
    ///
    /// drop(strong_five);
    /// drop(detached_five);
    ///
    /// assert!(weak_five.upgrade().is_none());
    /// ```
    ///
    /// [`Drc`]: ./struct.Drc.html
    ///
    /// [`Option`]: https://doc.rust-lang.org/std/option/enum.Option.html
    /// [`Arc`]: https://doc.rust-lang.org/std/sync/struct.Arc.html
    /// [`WeakArc`]: https://doc.rust-lang.org/std/sync/struct.Weak.html
    pub fn upgrade(&self) -> Option<Drc<T>> {
        if unsafe { self.ptr.as_ref() }.strong_ref.is_none() {
            debug_assert_eq!(0, self.strong().get());

            // The lack of a local strong pointer does not necessarily mean that the
            // value is lost. We keep a copy of a weak pointer so we can still upgrade
            // if necessary.
            if let Some(arc) = WeakArc::upgrade(self.weak_arc()) {
                // We have the `Arc`, so increment set the strong reference count to 1
                // (if there was no `strong_ref`, strong reference count is 0).
                self.strong().set(1);

                // Also increment the weak reference count to re-add the implicit weak
                // pointer.
                self.weak().set(self.weak().get() + 1);

                // Add the `Arc` back so the `Drc` will function properly.
                *unsafe { &mut (*self.ptr.as_ptr()).strong_ref } = Some(arc);

                // `DrcInner` is now sufficiently prepared to be used by a `Drc`.
                Some(Drc {
                    ptr: self.ptr,
                    phantom: PhantomData,
                })
            } else {
                // We can't get another `Arc`, so we return `None`. The value is gone.
                None
            }
        } else {
            // The `Arc`'s existence guarantees that the value exists, so we just
            // increment the strong reference counter and return a new strong pointer.
            self.strong().set(self.strong().get() + 1);
            Some(Drc {
                ptr: self.ptr,
                phantom: PhantomData,
            })
        }
    }

    /// Clone the internal [weak `Arc`][`WeakArc`] (incrementing the atomic weak
    /// reference count) so that the shared state can be weakly referenced on
    /// another thread. Then, return this newly cloned weak `Arc`. To convert
    /// the weak `Arc` back into this type of `Weak` (usually not
    /// recommended), use [`Weak::with_weak_arc`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Weak as WeakArc;
    ///
    /// use drc::Drc;
    ///
    /// let five = Drc::new(5);
    ///
    /// let weak_five = Drc::downgrade(&five);
    /// let detached_weak_five: WeakArc<_> = weak_five.detach();
    ///
    /// assert!(Drc::arc_ptr_eq(
    ///     &weak_five.upgrade().unwrap(),
    ///     &detached_weak_five.upgrade().unwrap()
    /// ));
    /// ```
    ///
    /// [`Weak::with_weak_arc`]: ./struct.Weak.html#method.with_weak_arc
    ///
    /// [`WeakArc`]: https://doc.rust-lang.org/std/sync/struct.Weak.html
    ///
    /// [`from`]: https://doc.rust-lang.org/std/convert/trait.From.html#tymethod.from
    pub fn detach(&self) -> WeakArc<T> {
        WeakArc::clone(self.weak_arc())
    }

    /// Similar to how the `Drc` implementation can grab the stored `Arc`
    /// because its existence is guaranteed when at least one `Drc` exists,
    /// we can grab the `WeakArc` for the same reason.
    fn weak_arc(&self) -> &WeakArc<T> {
        // Compiler should hopefully easily see this as a no-op, but this is a
        // precaution against an WeakArc<T> somehow not being subject to null pointer
        // optimization.
        assert_eq!(
            mem::size_of::<WeakArc<T>>(),
            mem::size_of::<Option<WeakArc<T>>>(),
            "Error within drc::Weak<T>: Null pointer optimization does not apply to WeakArc<T>! \
             If you see this panic, please report it to the maintainer(s) of the \"drc\" crate."
        );

        // This is safe because the value will *always* be Some when a weak
        // pointer (i.e. self) exists.
        unsafe { &*(&self.ptr.as_ref().weak_ref as *const _ as *const WeakArc<T>) }
    }

    /// Ensure that the weak count is 1 (i.e. implicit weak only) before
    /// calling this method. Otherwise, this simply removes the `WeakArc`
    /// from the `Option` with the same justification as `weak_arc`.
    unsafe fn take_weak_arc(&self) -> WeakArc<T> {
        // Compiler should hopefully easily see this as a no-op, but this is a
        // precaution against an WeakArc<T> somehow not being subject to null pointer
        // optimization.
        assert_eq!(
            mem::size_of::<WeakArc<T>>(),
            mem::size_of::<Option<WeakArc<T>>>(),
            "Error within drc::Weak<T>: Null pointer optimization does not apply to WeakArc<T>! \
             If you see this panic, please report it to the maintainer(s) of the \"drc\" crate."
        );

        let storage = self.ptr.as_ptr();

        let weak_arc = ptr::read(&(*storage).weak_ref as *const _ as *const WeakArc<T>);
        ptr::write(&mut (*storage).weak_ref, None);
        weak_arc
    }

    /// Retrieve the raw `Cell` that holds the number of weak pointers
    /// associated with the local `Arc` (if it still exists).
    fn weak(&self) -> &Cell<usize> {
        unsafe { &self.ptr.as_ref().weak }
    }

    /// Retrieve the raw `Cell` that holds the number of strong pointers
    /// associated with the local `Arc` (if it still exists).
    fn strong(&self) -> &Cell<usize> {
        unsafe { &self.ptr.as_ref().strong }
    }
}

impl<T> Clone for Weak<T>
where
    T: ?Sized,
{
    /// Makes a clone of the `Weak` pointer that points to the same value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::{Drc, Weak};
    ///
    /// let weak_five = Drc::downgrade(&Drc::new(5));
    ///
    /// Weak::clone(&weak_five);
    /// ```
    fn clone(&self) -> Weak<T> {
        self.weak().set(self.weak().get() + 1);
        Weak { ptr: self.ptr }
    }
}

impl<T> Drop for Weak<T>
where
    T: ?Sized,
{
    /// Drops the `Weak` pointer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::{Drc, Weak};
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
    /// let weak_foo = Drc::downgrade(&foo);
    /// let other_weak_foo = Weak::clone(&weak_foo);
    ///
    /// drop(weak_foo); // Doesn't print anything
    /// drop(foo); // Prints "dropped!"
    ///
    /// assert!(other_weak_foo.upgrade().is_none());
    /// ```
    fn drop(&mut self) {
        unsafe {
            let weak = self.weak().get();
            self.weak().set(weak - 1);
            if self.strong().get() == 0 {
                if weak == 1
                /* now it is 0 */
                {
                    // Recreate the `Box` that the `DrcInner` was originally allocated as,
                    // so that its `Drop` implementation can run.
                    // Since we use only safe types within the `DrcInner`, this will
                    // work as expected.
                    Box::from_raw(self.ptr.as_ptr());
                }
            } else {
                if weak == 2
                /* now it is 1 */
                {
                    // Drop the contained `WeakArc`.
                    drop(self.take_weak_arc());

                    // That's it; the remaining strong pointer(s) or a weak pointer
                    // later downgraded from a strong pointer will clean up the rest.
                }
            }
        }
    }
}

impl<T> Default for Weak<T> {
    /// Constructs a new `Weak<T>`, allocating memory for not only `T` and its
    /// atomic reference counts but also the local reference counts. Calling
    /// [`upgrade`] on the return value always gives [`None`][`Option`].
    ///
    /// Exercise restraint when using this method; if possible it is recommended
    /// to simply use the weak `Arc`'s [`Default`] implementation for
    /// whatever purpose this `new` method would fulfill.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use drc::Weak;
    ///
    /// let empty: Weak<i64> = Default::default();
    /// assert!(empty.upgrade().is_none());
    /// ```
    ///
    /// [`upgrade`]: ./struct.Weak.html#method.upgrade
    ///
    /// [`Option`]: https://doc.rust-lang.org/std/option/enum.Option.html
    /// [`Default`]: https://doc.rust-lang.org/std/default/trait.Default.html
    fn default() -> Weak<T> {
        Weak::new()
    }
}

impl<T> Debug for Weak<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "(Weak)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_default() {
        for i in 0usize..2usize {
            let weak: Weak<usize> = match i {
                0 => Weak::new(),
                1 => Default::default(),
                _ => unreachable!(),
            };

            let storage = unsafe { weak.ptr.as_ref() };

            assert_eq!(
                storage.strong.get(),
                0,
                "Freshly created weak pointer had nonzero strong count."
            );
            assert_eq!(
                storage.weak.get(),
                1,
                "Freshly created weak pointer had a weak count other than one."
            );
            assert!(
                storage.strong_ref.is_none(),
                "Freshly created weak pointer had a strong Arc."
            );
            assert!(
                storage.weak_ref.is_some(),
                "Freshly created weak pointer was missing its weak Arc."
            );

            assert!(
                weak.upgrade().is_none(),
                "Freshly created weak pointer upgraded into a strong Drc."
            );
        }
    }

    #[test]
    fn with_weak_arc() {
        use std::sync::Arc;

        // NO VALUE
        {
            let weak: Weak<usize> = Weak::with_weak_arc(WeakArc::new());

            let storage = unsafe { weak.ptr.as_ref() };

            assert_eq!(
                storage.strong.get(),
                0,
                "Weak pointer from freshly created weak Arc had nonzero strong count."
            );
            assert_eq!(
                storage.weak.get(),
                1,
                "Weak pointer from freshly created weak Arc had a weak count other than one."
            );
            assert!(
                storage.strong_ref.is_none(),
                "Weak pointer from freshly created weak Arc had a strong Arc."
            );
            assert!(
                storage.weak_ref.is_some(),
                "Weak pointer from freshly created weak Arc was missing its weak Arc."
            );

            assert!(
                weak.upgrade().is_none(),
                "Weak pointer from freshly created weak Arc upgraded into a strong Drc."
            );
        }

        // HAS VALUE
        {
            let arc = Arc::new(17usize);
            let drc = {
                let weak = Weak::with_weak_arc(Arc::downgrade(&arc));

                let storage = unsafe { weak.ptr.as_ref() };

                assert_eq!(
                    storage.strong.get(),
                    0,
                    "Weak pointer with freshly made DrcInner had nonzero strong count."
                );
                assert_eq!(
                    storage.weak.get(),
                    1,
                    "Weak pointer with freshly made DrcInner had a weak count other than one."
                );
                assert!(
                    storage.strong_ref.is_none(),
                    "Weak pointer with freshly made DrcInner had a strong Arc."
                );
                assert!(
                    storage.weak_ref.is_some(),
                    "Weak pointer with freshly made DrcInner was missing its weak Arc."
                );

                let maybe_drc = weak.upgrade();
                assert!(
                    maybe_drc.is_some(),
                    "Weak pointer with associated weak Arc that should have been upgradeable did \
                     not upgrade."
                );
                maybe_drc.unwrap()
            };

            assert_eq!(
                *drc, 17usize,
                "Upgraded Drc does not have the proper value."
            );
            assert!(
                Drc::arc_ptr_eq(&drc, &arc),
                "Upgraded Drc did not point to the same value as the Arc."
            );
        }
    }

    // Duplicate of the documentation example because it sufficiently tests the
    // method, but it's still repeated here in case the docs change.
    #[test]
    fn upgrade() {
        let five = Drc::new(5);

        let detached_five = Drc::detach(&five);

        let weak_five = Drc::downgrade(&five);

        let strong_five: Option<Drc<_>> = weak_five.upgrade();
        assert!(strong_five.is_some());

        mem::drop(strong_five);
        mem::drop(five);

        let strong_five: Option<Drc<_>> = weak_five.upgrade();
        assert!(strong_five.is_some());

        mem::drop(strong_five);
        mem::drop(detached_five);

        assert!(weak_five.upgrade().is_none());
    }

    // Duplicate of the documentation example because it sufficiently tests the
    // method, but it's still repeated here in case the docs change.
    #[test]
    fn detach() {
        let five = Drc::new(5);

        let weak_five = Drc::downgrade(&five);
        let detached_weak_five: WeakArc<_> = weak_five.detach();

        assert!(Drc::arc_ptr_eq(
            &weak_five.upgrade().unwrap(),
            &detached_weak_five.upgrade().unwrap()
        ));
    }

    // Test inspired by test for 'Drc::clone', since the documentation example for
    // 'Weak::clone' inherited from 'Rc' isn't sufficient unlike the one for
    // 'Drc::clone'.
    #[test]
    fn clone() {
        let five = Drc::new(5);

        let weak_five = Drc::downgrade(&five);

        let same_weak_five = Weak::clone(&weak_five);

        // `five` and `same_weak_five` (when upgraded) share the same `Arc`.
        assert!(Drc::linked(&five, &same_weak_five.upgrade().unwrap()));

        // Local weak reference count of 2:
        // `weak_five`, `same_weak_five`
        assert_eq!(2, Drc::weak_count(&five, true));

        // `Arc` weak reference count of 1:
        // (`weak_five`, `same_weak_five`)
        assert_eq!(1, Drc::weak_count(&five, false));
    }

    // Test inspired by and not much more complex than the documentation example,
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
        let weak_foo = Drc::downgrade(&foo);
        let other_weak_foo = Weak::clone(&weak_foo);

        // Cell is currently false.
        assert_eq!(cell.get(), false);

        // Cell is not set to true since Foo::drop is not run.
        mem::drop(weak_foo);
        assert_eq!(cell.get(), false);

        // Cell is set to true here since Foo::drop is run.
        mem::drop(foo);
        assert_eq!(cell.get(), true);

        // The remaining weak pointer cannot be upgraded because no strong pointer
        // remains!
        assert!(other_weak_foo.upgrade().is_none());
    }

    #[test]
    fn debug() {
        let weak: Weak<usize> = Weak::new();

        assert_eq!(
            format!("{:?}", weak),
            "(Weak)",
            "Weak pointers should debug-format to the string \"(Weak)\""
        );
    }
}
