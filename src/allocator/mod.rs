mod backend;
use backend::AllocatorBackend;
use std::{
    fmt::Debug,
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use ash::vk;

use super::context::Context;

pub const GLOBAL_MEMORY_SIZE: u64 = 2u64 << 30; // 2GB
pub const STAGING_MEMORY_SIZE: u64 = 100u64 << 20; // 100MB

pub struct Allocator {
    pub context: Arc<Context>,
    pub pending_transfers: Vec<PendingTransfer>,
    #[allow(unused)]
    pub pending_frees: Vec<PendingFree>,
    offset_allocator: offset_allocator::Allocator,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub sync2_pfn: ash::khr::synchronization2::Device,
    backend: AllocatorBackend,
}

impl Allocator {
    pub fn new(context: Arc<Context>, sync2_pfn: ash::khr::synchronization2::Device) -> Self {
        let backend = AllocatorBackend::new(context.clone());
        let offset_allocator = offset_allocator::Allocator::new(GLOBAL_MEMORY_SIZE as u32);

        Self {
            backend,
            context,
            offset_allocator,
            pending_frees: Default::default(),
            pending_transfers: Default::default(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            sync2_pfn,
        }
    }

    pub fn allocate_buffer<T: Sized>(
        &mut self,
        max_size: usize,
        usage_flags: vk::BufferUsageFlags,
    ) -> BufferAllocation<T> {
        let device = &self.context.device;
        let device_size = (max_size * std::mem::size_of::<T>()) as vk::DeviceSize;

        // Allocate an offset into our device local memory
        let offset = self
            .offset_allocator
            .allocate(device_size as u32)
            .expect("Unable to allocate memory. This should be impossible!");

        // Create the buffer
        let handle = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default().size(device_size).usage(
                    usage_flags
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_DST,
                ),
                None,
            )
        }
        .unwrap();

        // Bind its memory
        unsafe {
            device.bind_buffer_memory(handle, self.backend.device_memory(), offset.offset as _)
        }
        .unwrap();

        // Get its device address
        let device_address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(handle))
        };

        BufferAllocation {
            max_device_size: device_size,
            device_address,
            len: 0,
            handle,
            global_offset: offset,
            _phantom: PhantomData,
        }
    }

    pub fn stage_transfer<T: bytemuck::Pod + Debug>(
        &mut self,
        data: &[T],
        allocation: &mut BufferAllocation<T>,
    ) -> TransferToken {
        let bytes = bytemuck::cast_slice(data);
        let staging_buffer_offset = self.backend.stage_transfer(bytes);
        allocation.len += data.len();

        let (ours, theirs) = TransferToken::create_pair();

        self.pending_transfers.push(PendingTransfer {
            destination: allocation.handle,
            staging_buffer_offset,
            transfer_size: bytes.len() as _,
            global_offset: allocation.global_offset,
            transfer_token: ours,
        });

        theirs
    }

    pub fn execute_transfers(&mut self) {
        let mut barriers = vec![];

        for pending_transfer in self.pending_transfers.drain(..) {
            self.backend
                .execute_transfer(pending_transfer, &mut barriers);
        }

        self.backend.transfers_complete(&barriers);
    }

    #[allow(unused)]
    pub fn free<T: Sized>(&mut self, allocation: BufferAllocation<T>) {}
}

#[derive(Clone)]
pub struct TransferToken {
    complete: Arc<AtomicBool>,
}

impl TransferToken {
    /// TODO: need to be clear about under what conditions this is true
    pub fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Relaxed)
    }

    fn create_pair() -> (TransferToken, TransferToken) {
        let complete = Arc::new(AtomicBool::new(false));
        (
            TransferToken {
                complete: complete.clone(),
            },
            TransferToken { complete },
        )
    }

    fn mark_completed(&self) {
        self.complete.store(true, Ordering::Relaxed);
    }
}

pub struct PendingTransfer {
    destination: vk::Buffer,
    staging_buffer_offset: usize,
    transfer_size: vk::DeviceSize,
    global_offset: offset_allocator::Allocation, // offset into the global memory
    transfer_token: TransferToken,
}

pub struct PendingFree;
pub struct BufferAllocation<T> {
    #[allow(unused)]
    pub max_device_size: vk::DeviceSize,
    pub device_address: vk::DeviceAddress,
    pub handle: vk::Buffer,
    len: usize, // number of `T`s in this buffer
    #[allow(unused)]
    global_offset: offset_allocator::Allocation, // offset into the global memory
    _phantom: PhantomData<T>,
}

impl<T> BufferAllocation<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }
}
