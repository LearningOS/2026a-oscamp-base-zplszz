//! # Memory Ordering and Synchronization
//!
//! In this exercise, you will use correct memory ordering to implement thread synchronization primitives.
//!
//! ## Key Concepts
//! - `Ordering::Relaxed`: No synchronization guarantees
//! - `Ordering::Acquire`: Read operation, prevents subsequent reads/writes from being reordered before this operation
//! - `Ordering::Release`: Write operation, prevents preceding reads/writes from being reordered after this operation
//! - `Ordering::AcqRel`: Both Acquire and Release semantics
//! - `Ordering::SeqCst`: Sequentially consistent (global ordering)
//!
//! ## Release-Acquire Pairing
//! When thread A writes with Release, and thread B reads the same location with Acquire,
//! thread B will see all writes that thread A performed before the Release.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Use Release-Acquire semantics to safely pass data between two threads.
///
/// `produce` writes data first, then sets flag with Release;
/// `consume` reads flag with Acquire, ensuring it sees the data.
pub struct FlagChannel {
    data: AtomicU32,
    ready: AtomicBool,
}

impl FlagChannel {
    pub const fn new() -> Self {
        Self {
            data: AtomicU32::new(0),
            ready: AtomicBool::new(false),
        }
    }

    /// Producer: store data first, then set ready flag.
    ///
    /// Use for one producer/consumer exchange; reset only after both are done.
    pub fn produce(&self, value: u32) {
        self.data.store(value, Ordering::Relaxed);
        self.ready.store(true, Ordering::Release);
    }

    /// Consumer: spin-wait for ready flag, then read data.
    ///
    pub fn consume(&self) -> u32 {
        while !self.ready.load(Ordering::Acquire) {
            core::hint::spin_loop();
        }
        self.data.load(Ordering::Relaxed)
    }

    /// Reset channel state after the producer and consumer have finished.
    pub fn reset(&self) {
        self.ready.store(false, Ordering::Relaxed);
        self.data.store(0, Ordering::Relaxed);
    }
}

/// A simple once-initializer using SeqCst.
/// Guarantees `init` is executed only once, and all threads see the initialized value.
pub struct OnceCell {
    initializing: AtomicBool,
    initialized: AtomicBool,
    value: AtomicU32,
}

impl OnceCell {
    pub const fn new() -> Self {
        Self {
            initializing: AtomicBool::new(false),
            initialized: AtomicBool::new(false),
            value: AtomicU32::new(0),
        }
    }

    /// Attempt initialization. If not yet initialized, store value and return true.
    /// If already initialized, return false.
    ///
    /// Hint: use `compare_exchange` to ensure only one thread succeeds.
    pub fn init(&self, val: u32) -> bool {
        if self
            .initializing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        self.value.store(val, Ordering::SeqCst);
        // Claiming initialization and publishing completion are separate steps:
        // readers must never see Some(0) while the winner is still writing.
        self.initialized.store(true, Ordering::SeqCst);
        true
    }

    /// Get value. Returns Some if initialized, otherwise None.
    pub fn get(&self) -> Option<u32> {
        if self.initialized.load(Ordering::SeqCst) {
            Some(self.value.load(Ordering::SeqCst))
        } else {
            None
        }
    }
}

impl Default for FlagChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for OnceCell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_once_cell_does_not_publish_claimed_initialization() {
        let cell = OnceCell::new();
        // Simulate a winner paused after claiming initialization.
        cell.initializing.store(true, Ordering::SeqCst);
        assert_eq!(cell.get(), None);
        assert!(!cell.init(99));
        cell.value.store(42, Ordering::SeqCst);
        cell.initialized.store(true, Ordering::SeqCst);
        assert_eq!(cell.get(), Some(42));
    }

    #[test]
    fn test_once_cell_value_visible_before_publication() {
        for value in 1..=64 {
            let cell = OnceCell::new();
            let barrier = std::sync::Barrier::new(2);
            thread::scope(|scope| {
                scope.spawn(|| {
                    barrier.wait();
                    assert!(cell.init(value));
                });
                barrier.wait();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                loop {
                    if let Some(actual) = cell.get() {
                        assert_eq!(actual, value);
                        break;
                    }
                    assert!(
                        std::time::Instant::now() < deadline,
                        "initialization timed out"
                    );
                    thread::yield_now();
                }
            });
        }
    }

    #[test]
    fn test_once_cell_zero_is_valid() {
        let cell = OnceCell::default();
        assert!(cell.init(0));
        assert_eq!(cell.get(), Some(0));
        assert!(!cell.init(1));
    }

    #[test]
    fn test_flag_channel_reset_and_reuse() {
        let channel = FlagChannel::default();
        channel.produce(42);
        assert_eq!(channel.consume(), 42);
        channel.reset();
        thread::scope(|scope| {
            scope.spawn(|| channel.produce(99));
            assert_eq!(channel.consume(), 99);
        });
    }

    #[test]
    fn test_flag_channel() {
        let ch = Arc::new(FlagChannel::new());
        let ch2 = Arc::clone(&ch);

        let producer = thread::spawn(move || {
            ch2.produce(42);
        });

        let consumer = thread::spawn(move || ch.consume());

        producer.join().unwrap();
        let val = consumer.join().unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_flag_channel_large_value() {
        let ch = Arc::new(FlagChannel::new());
        let ch2 = Arc::clone(&ch);

        let producer = thread::spawn(move || {
            ch2.produce(0xDEAD_BEEF);
        });

        let val = ch.consume();
        producer.join().unwrap();
        assert_eq!(val, 0xDEAD_BEEF);
    }

    #[test]
    fn test_once_cell_init_once() {
        let cell = OnceCell::new();
        assert!(cell.init(42));
        assert!(!cell.init(100));
        assert_eq!(cell.get(), Some(42));
    }

    #[test]
    fn test_once_cell_not_initialized() {
        let cell = OnceCell::new();
        assert_eq!(cell.get(), None);
    }

    #[test]
    fn test_once_cell_concurrent() {
        let cell = Arc::new(OnceCell::new());
        let mut handles = vec![];

        for i in 0..10 {
            let c = Arc::clone(&cell);
            handles.push(thread::spawn(move || c.init(i)));
        }

        let results: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // Exactly one thread initializes successfully
        assert_eq!(results.iter().filter(|&&r| r).count(), 1);
        assert!(cell.get().is_some());
    }
}
