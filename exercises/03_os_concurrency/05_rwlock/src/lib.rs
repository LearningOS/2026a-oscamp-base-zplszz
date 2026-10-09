//! # Read-Write Lock (Writer-Priority)
//!
//! In this exercise, you will implement a **writer-priority** read-write lock from scratch using atomics.
//! Multiple readers may hold the lock concurrently; a writer holds it exclusively.
//!
//! **Note:** Rust's standard library already provides [`std::sync::RwLock`]. This exercise implements
//! a minimal version for learning the protocol and policy without using the standard one.
//!
//! ## Common policies for read-write locks
//! Different implementations can give different **priority** when both readers and writers are waiting:
//!
//! - **Reader-priority (读者优先)**: New readers are allowed to enter while a writer is waiting, so writers
//!   may be starved if readers keep arriving.
//! - **Writer-priority (写者优先)**: Once a writer is waiting, no new readers are admitted until that writer
//!   has run; this exercise implements this policy.
//! - **Read-write fair (读写公平)**: Requests are served in a fair order (e.g. FIFO or round-robin), so
//!   neither readers nor writers are systematically starved.
//!
//! ## Key Concepts
//! - **Readers**: share access; many threads can hold a read lock at once.
//! - **Writer**: exclusive access; only one writer, and no readers while the writer holds the lock.
//! - **Writer-priority (this implementation)**: when at least one writer is waiting, new readers block
//!   until the writer runs.
//!
//! ## State (single atomic)
//! We use one `AtomicU32`: bits 0..15 = reader count, bits 15..30 = waiting writer count,
//! and two flags = writer holding / writer waiting. Counting waiting writers
//! prevents one writer from clearing the priority flag while others still wait.
//! All logic is implemented with compare_exchange and load/store; no use of `std::sync::RwLock`.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, Ordering};

/// Maximum number of concurrent readers (fits in state bits).
const READER_MASK: u32 = (1 << 15) - 1;
/// One waiting writer, and the mask for the waiting writer count.
const WRITER_UNIT: u32 = 1 << 15;
const WAITING_WRITER_MASK: u32 = READER_MASK << 15;
/// Bit set when a writer holds the lock.
const WRITER_HOLDING: u32 = 1 << 30;
/// Bit set when at least one writer is waiting (writer-priority: block new readers).
const WRITER_WAITING: u32 = 1 << 31;

/// Writer-priority read-write lock. Implemented from scratch; does not use `std::sync::RwLock`.
pub struct RwLock<T> {
    state: AtomicU32,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for RwLock<T> {}
unsafe impl<T: Send + Sync> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    pub const fn new(data: T) -> Self {
        Self {
            state: AtomicU32::new(0),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquire a read lock. Blocks (spins) until no writer holds and no writer is waiting (writer-priority).
    ///
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        loop {
            let state = self.state.load(Ordering::Acquire);
            if state & (WRITER_HOLDING | WRITER_WAITING) == 0
                && state & READER_MASK != READER_MASK
                && self
                    .state
                    .compare_exchange(state, state + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                return RwLockReadGuard { lock: self };
            }
            core::hint::spin_loop();
        }
    }

    /// Acquire the write lock. Blocks until no readers and no other writer.
    ///
    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        // Register atomically with the priority flag, so readers cannot barge
        // between updating a separate writer counter and publishing the flag.
        loop {
            let state = self.state.load(Ordering::Acquire);
            if state & WAITING_WRITER_MASK != WAITING_WRITER_MASK {
                let next = (state + WRITER_UNIT) | WRITER_WAITING;
                if self
                    .state
                    .compare_exchange(state, next, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
            core::hint::spin_loop();
        }
        loop {
            let state = self.state.load(Ordering::Acquire);
            if state & (READER_MASK | WRITER_HOLDING) == 0 {
                let mut next = (state - WRITER_UNIT) | WRITER_HOLDING;
                if next & WAITING_WRITER_MASK == 0 {
                    next &= !WRITER_WAITING;
                }
                if self
                    .state
                    .compare_exchange(state, next, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return RwLockWriteGuard { lock: self };
                }
            }
            core::hint::spin_loop();
        }
    }
}

/// Guard for a read lock; releases the read lock on drop.
pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> Drop for RwLockReadGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.fetch_sub(1, Ordering::Release);
    }
}

/// Guard for a write lock; releases the write lock on drop.
pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for RwLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for RwLockWriteGuard<'_, T> {
    fn drop(&mut self) {
        // Preserve waiting writers and their priority across writer handoffs.
        self.lock
            .state
            .fetch_and(!WRITER_HOLDING, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_readers_can_hold_lock_simultaneously() {
        let lock = RwLock::new(42);
        let first = lock.read();
        let second = lock.read();
        assert_eq!(*first, 42);
        assert_eq!(*second, 42);
        assert_eq!(lock.state.load(Ordering::Acquire) & READER_MASK, 2);
        drop(first);
        drop(second);
        assert_eq!(lock.state.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_multiple_waiting_writers_keep_priority() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        let lock = Arc::new(RwLock::new(0u32));
        let initial_reader = lock.read();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let mut releases = Vec::new();
        let mut writers = Vec::new();
        for id in 0..2 {
            let lock = Arc::clone(&lock);
            let acquired = acquired_tx.clone();
            let (release_tx, release_rx) = mpsc::channel();
            releases.push(release_tx);
            writers.push(thread::spawn(move || {
                let mut guard = lock.write();
                *guard += 1;
                acquired.send(id).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            }));
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while lock.state.load(Ordering::Acquire) & WAITING_WRITER_MASK != 2 * WRITER_UNIT {
            assert!(Instant::now() < deadline, "writers did not register");
            thread::yield_now();
        }
        let (reader_tx, reader_rx) = mpsc::channel();
        let reader_lock = Arc::clone(&lock);
        let reader = thread::spawn(move || {
            let guard = reader_lock.read();
            reader_tx.send(*guard).unwrap();
        });
        drop(initial_reader);
        let first = acquired_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let state = lock.state.load(Ordering::Acquire);
        assert_ne!(state & WRITER_WAITING, 0);
        assert_eq!(state & WAITING_WRITER_MASK, WRITER_UNIT);
        assert!(matches!(
            reader_rx.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        releases[first].send(()).unwrap();
        let second = acquired_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_ne!(first, second);
        assert!(matches!(
            reader_rx.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        releases[second].send(()).unwrap();
        assert_eq!(reader_rx.recv_timeout(Duration::from_secs(5)).unwrap(), 2);
        for writer in writers {
            writer.join().unwrap();
        }
        reader.join().unwrap();
        assert_eq!(lock.state.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_writer_panic_releases_lock() {
        let lock = Arc::new(RwLock::new(0));
        let writer_lock = Arc::clone(&lock);
        assert!(thread::spawn(move || {
            let mut guard = writer_lock.write();
            *guard = 42;
            panic!("intentional writer panic");
        })
        .join()
        .is_err());
        assert_eq!(lock.state.load(Ordering::Acquire), 0);
        assert_eq!(*lock.read(), 42);
    }

    #[test]
    fn test_multiple_readers() {
        let lock = Arc::new(RwLock::new(0u32));
        let mut handles = vec![];
        for _ in 0..10 {
            let l = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                let g = l.read();
                assert_eq!(*g, 0);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_writer_excludes_readers() {
        let lock = Arc::new(RwLock::new(0u32));
        let lock_w = Arc::clone(&lock);
        let writer = thread::spawn(move || {
            let mut g = lock_w.write();
            *g = 42;
        });
        writer.join().unwrap();
        let g = lock.read();
        assert_eq!(*g, 42);
    }

    #[test]
    fn test_concurrent_reads_after_write() {
        let lock = Arc::new(RwLock::new(Vec::<i32>::new()));
        {
            let mut g = lock.write();
            g.push(1);
            g.push(2);
        }
        let mut handles = vec![];
        for _ in 0..5 {
            let l = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                let g = l.read();
                assert_eq!(g.len(), 2);
                assert_eq!(&*g, &[1, 2]);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_writes_serialized() {
        let lock = Arc::new(RwLock::new(0u64));
        let mut handles = vec![];
        for _ in 0..10 {
            let l = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let mut g = l.write();
                    *g += 1;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*lock.read(), 1000);
    }
}
