//! # Free-List Allocator
//!
//! Building on the bump allocator, implement a Free-List Allocator that supports memory reclamation.
//!
//! ## How It Works
//!
//! A Free-List Allocator uses a linked list to track all freed memory blocks.
//! On allocation, it first searches the list for a suitable block (first-fit strategy);
//! if none is found, it falls back to allocating from the unused region.
//! On deallocation, the block is inserted at the head of the list.
//!
//! ```text
//! free_list -> [block A: 64B] -> [block B: 128B] -> [block C: 32B] -> null
//! ```
//!
//! Each free block stores a `FreeBlock` struct at its head (containing block size and next pointer).
//!
//! ## Task
//!
//! Implement `FreeListAllocator`'s `alloc` and `dealloc` methods:
//!
//! ### alloc
//! 1. Traverse the free_list, find the first block with `size >= layout.size()` and proper alignment (first-fit)
//! 2. If found, remove it from the list and return it
//! 3. If not found, allocate from the `bump` region (same as bump allocator)
//!
//! ### dealloc
//! 1. Write `FreeBlock` header info at the freed block
//! 2. Insert it at the head of free_list
//!
//! ## Key Concepts
//!
//! - Intrusive linked list
//! - `*mut T` read/write: `ptr.write(val)` / `ptr.read()`
//! - Memory alignment checks

#![cfg_attr(not(test), no_std)]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Free block header, stored at the beginning of each free memory block
struct FreeBlock {
    size: usize,
    next: *mut FreeBlock,
}

pub struct FreeListAllocator {
    heap_end: usize,
    /// Bump pointer: unallocated region starts here
    bump_next: AtomicUsize,
    /// Protect the entire list operation, not just reading/updating the head.
    locked: AtomicBool,
    free_list: UnsafeCell<*mut FreeBlock>,
}

unsafe impl Send for FreeListAllocator {}
unsafe impl Sync for FreeListAllocator {}

struct FreeListGuard<'a>(&'a FreeListAllocator);

impl Drop for FreeListGuard<'_> {
    fn drop(&mut self) {
        self.0.locked.store(false, Ordering::Release);
    }
}

impl FreeListAllocator {
    /// # Safety
    /// `heap_start..heap_end` must be a valid readable and writable memory region
    /// exclusively owned by this allocator for its entire lifetime.
    pub unsafe fn new(heap_start: usize, heap_end: usize) -> Self {
        Self {
            heap_end,
            bump_next: AtomicUsize::new(heap_start),
            locked: AtomicBool::new(false),
            free_list: UnsafeCell::new(null_mut()),
        }
    }

    fn lock(&self) -> FreeListGuard<'_> {
        while self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        FreeListGuard(self)
    }

    fn block_size(layout: Layout) -> usize {
        let align = core::mem::align_of::<FreeBlock>();
        // Round up so that any split tail can hold an aligned FreeBlock header.
        (layout.size().max(core::mem::size_of::<FreeBlock>()) + align - 1) & !(align - 1)
    }
}

unsafe impl GlobalAlloc for FreeListAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Ensure block is at least large enough to hold a FreeBlock header (for future dealloc)
        let size = Self::block_size(layout);
        let align = layout.align().max(core::mem::align_of::<FreeBlock>());
        let _guard = self.lock();
        // `link` points either to the head or to the previous block's next field.
        // All header accesses are protected by the same lock in tests and no_std.
        let mut link = self.free_list.get();
        unsafe {
            while !(*link).is_null() {
                let block = *link;
                if (block as usize).is_multiple_of(align) && (*block).size >= size {
                    let remaining = (*block).size - size;
                    if remaining >= core::mem::size_of::<FreeBlock>() {
                        let tail = block.cast::<u8>().add(size).cast::<FreeBlock>();
                        tail.write(FreeBlock {
                            size: remaining,
                            next: (*block).next,
                        });
                        *link = tail;
                    } else {
                        *link = (*block).next;
                    }
                    return block.cast();
                }
                link = core::ptr::addr_of_mut!((*block).next);
            }
        }

        let next = self.bump_next.load(Ordering::Relaxed);
        let Some(aligned) = next.checked_add(align - 1) else {
            return null_mut();
        };
        let aligned = aligned & !(align - 1);
        let Some(end) = aligned.checked_add(size) else {
            return null_mut();
        };
        if end > self.heap_end {
            return null_mut();
        }
        // The list lock also serializes bump allocations.
        self.bump_next.store(end, Ordering::Relaxed);
        aligned as *mut u8
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = Self::block_size(layout);
        let _guard = self.lock();
        let block = ptr.cast::<FreeBlock>();
        unsafe {
            block.write(FreeBlock {
                size,
                next: *self.free_list.get(),
            });
            *self.free_list.get() = block;
        }
    }
}

// ============================================================
// Tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    const HEAP_SIZE: usize = 4096;

    fn make_allocator() -> (FreeListAllocator, Vec<u8>) {
        let mut heap = vec![0u8; HEAP_SIZE];
        let start = heap.as_mut_ptr() as usize;
        let alloc = unsafe { FreeListAllocator::new(start, start + HEAP_SIZE) };
        (alloc, heap)
    }

    #[test]
    fn test_small_block_can_store_header_and_be_reused() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(1, 1).unwrap();
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % core::mem::align_of::<FreeBlock>(), 0);
        unsafe {
            ptr.write(42);
            alloc.dealloc(ptr, layout);
        }
        assert_eq!(unsafe { alloc.alloc(layout) }, ptr);
    }

    #[test]
    fn test_first_fit_removes_non_head_block() {
        let (alloc, _heap) = make_allocator();
        let large = Layout::from_size_align(128, 8).unwrap();
        let small = Layout::from_size_align(32, 8).unwrap();
        let p = unsafe { alloc.alloc(large) };
        let q = unsafe { alloc.alloc(small) };
        unsafe {
            alloc.dealloc(p, large);
            alloc.dealloc(q, small);
        }
        assert_eq!(unsafe { alloc.alloc(large) }, p);
        assert_eq!(unsafe { alloc.alloc(small) }, q);
    }

    #[test]
    fn test_split_block_retains_remainder() {
        let (alloc, _heap) = make_allocator();
        let large = Layout::from_size_align(128, 8).unwrap();
        let small = Layout::from_size_align(32, 8).unwrap();
        let ptr = unsafe { alloc.alloc(large) };
        unsafe { alloc.dealloc(ptr, large) };
        for i in 0..4 {
            assert_eq!(unsafe { alloc.alloc(small) }, unsafe { ptr.add(i * 32) });
        }
    }

    #[test]
    fn test_reuse_checks_alignment() {
        let (alloc, _heap) = make_allocator();
        let low = Layout::from_size_align(32, 8).unwrap();
        let high = Layout::from_size_align(32, 64).unwrap();
        let ptr = unsafe { alloc.alloc(low) };
        unsafe { alloc.dealloc(ptr, low) };
        let aligned = unsafe { alloc.alloc(high) };
        assert!(!aligned.is_null());
        assert_eq!(aligned as usize % 64, 0);
        if !(ptr as usize).is_multiple_of(64) {
            assert_eq!(unsafe { alloc.alloc(low) }, ptr);
        }
    }

    #[test]
    fn test_reuse_after_heap_exhaustion() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(32, 8).unwrap();
        let mut blocks = Vec::new();
        loop {
            let ptr = unsafe { alloc.alloc(layout) };
            if ptr.is_null() {
                break;
            }
            blocks.push(ptr);
        }
        assert!(!blocks.is_empty());
        let ptr = blocks[blocks.len() / 2];
        unsafe { alloc.dealloc(ptr, layout) };
        assert_eq!(unsafe { alloc.alloc(layout) }, ptr);
        assert!(unsafe { alloc.alloc(layout) }.is_null());
    }

    #[test]
    fn test_concurrent_allocation_and_reclamation() {
        let (alloc, _heap) = make_allocator();
        let barrier = std::sync::Barrier::new(8);
        let layout = Layout::from_size_align(32, 8).unwrap();
        std::thread::scope(|scope| {
            for id in 0..8u8 {
                let alloc = &alloc;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..200 {
                        let ptr = unsafe { alloc.alloc(layout) };
                        assert!(!ptr.is_null());
                        unsafe {
                            for i in 0..layout.size() {
                                ptr.add(i).write(id);
                            }
                        }
                        std::thread::yield_now();
                        unsafe {
                            for i in 0..layout.size() {
                                assert_eq!(ptr.add(i).read(), id);
                            }
                            alloc.dealloc(ptr, layout);
                        }
                    }
                });
            }
        });
    }

    #[test]
    fn test_alloc_basic() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(32, 8).unwrap();
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_alloc_alignment() {
        let (alloc, _heap) = make_allocator();
        for align in [1, 2, 4, 8, 16] {
            let layout = Layout::from_size_align(8, align).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null());
            assert_eq!(ptr as usize % align, 0, "align={align}");
        }
    }

    #[test]
    fn test_dealloc_and_reuse() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(64, 8).unwrap();

        let p1 = unsafe { alloc.alloc(layout) };
        assert!(!p1.is_null());

        // After freeing, the next allocation should reuse the same block
        unsafe { alloc.dealloc(p1, layout) };
        let p2 = unsafe { alloc.alloc(layout) };
        assert!(!p2.is_null());
        assert_eq!(p1, p2, "should reuse the freed block");
    }

    #[test]
    fn test_multiple_alloc_dealloc() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(128, 8).unwrap();

        let p1 = unsafe { alloc.alloc(layout) };
        let p2 = unsafe { alloc.alloc(layout) };
        let p3 = unsafe { alloc.alloc(layout) };
        assert!(!p1.is_null() && !p2.is_null() && !p3.is_null());

        unsafe { alloc.dealloc(p2, layout) };
        unsafe { alloc.dealloc(p1, layout) };

        let q1 = unsafe { alloc.alloc(layout) };
        let q2 = unsafe { alloc.alloc(layout) };
        assert!(!q1.is_null() && !q2.is_null());
    }

    #[test]
    fn test_oom() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(HEAP_SIZE + 1, 1).unwrap();
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(ptr.is_null(), "should return null when exceeding heap");
    }
}
