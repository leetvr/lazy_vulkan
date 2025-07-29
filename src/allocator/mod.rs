mod device_buffer;
mod staging_buffer;
use device_buffer::DeviceBuffer;
use staging_buffer::StagingBuffer;
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
    backend: DeviceBuffer,
    staging_buffer: StagingBuffer,
}

impl Allocator {
    pub fn new(context: Arc<Context>) -> Self {
        let backend = DeviceBuffer::new(context.clone());
        let staging_buffer = StagingBuffer::new(&context);
        let offset_allocator = offset_allocator::Allocator::new(GLOBAL_MEMORY_SIZE as u32);

        Self {
            backend,
            context,
            offset_allocator,
            pending_frees: Default::default(),
            pending_transfers: Default::default(),
            staging_buffer,
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

    pub fn allocate_image(
        &mut self,
        data: &[u8],
        extent: vk::Extent2D,
        image: vk::Image,
    ) -> TransferToken {
        let device = &self.context.device;
        let memory_requirements = unsafe { device.get_image_memory_requirements(image) };
        let size = memory_requirements.size;

        // Allocate an offset into our device local memory
        let global_offset = self
            .offset_allocator
            .allocate(size as u32)
            .expect("Unable to allocate memory. This should be impossible!");

        // Bind the image to the memory at this offset
        unsafe {
            device.bind_image_memory(
                image,
                self.backend.device_memory(),
                global_offset.offset as _,
            )
        }
        .unwrap();

        // Stage the transfer
        let (ours, theirs) = TransferToken::create_pair();
        let staging_buffer_offset = self.staging_buffer.stage(data);
        self.pending_transfers.push(PendingTransfer {
            destination: TransferDestination::Image(image, extent),
            transfer_size: data.len() as _,
            transfer_token: ours,
            staging_buffer_offset,
            global_offset,
        });

        theirs
    }

    pub fn stage_buffer_transfer<T: bytemuck::Pod + Debug>(
        &mut self,
        data: &[T],
        allocation: &mut BufferAllocation<T>,
    ) -> TransferToken {
        let bytes = bytemuck::cast_slice(data);
        let staging_buffer_offset = self.staging_buffer.stage(bytes);
        allocation.len += data.len();

        let (ours, theirs) = TransferToken::create_pair();

        self.pending_transfers.push(PendingTransfer {
            destination: TransferDestination::Buffer(allocation.handle),
            staging_buffer_offset,
            transfer_size: bytes.len() as _,
            global_offset: allocation.global_offset,
            transfer_token: ours,
        });

        theirs
    }

    pub fn execute_transfers(&mut self) {
        self.backend.execute_transfers(
            &self.context,
            std::mem::take(&mut self.pending_transfers),
            &mut self.staging_buffer,
        );
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
    destination: TransferDestination,
    staging_buffer_offset: usize,
    transfer_size: vk::DeviceSize,
    global_offset: offset_allocator::Allocation, // offset into the global memory
    transfer_token: TransferToken,
}

enum TransferDestination {
    Buffer(vk::Buffer),
    Image(vk::Image, vk::Extent2D),
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
