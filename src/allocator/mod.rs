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

    /// Allocates a buffer of `max_size`
    pub fn allocate_buffer<T: Sized>(
        &mut self,
        max_size: usize,
        usage_flags: vk::BufferUsageFlags,
    ) -> BufferAllocation<T> {
        let device = &self.context.device;
        let device_size = (max_size * std::mem::size_of::<T>()) as vk::DeviceSize;

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

        let memory_requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let align = memory_requirements.alignment;
        let size = memory_requirements.size;

        self.allocate_buffer_inner(align, handle, size)
    }

    pub fn allocate_buffer_with_alignment<T: Sized>(
        &mut self,
        max_size: usize,
        align: u64,
        usage_flags: vk::BufferUsageFlags,
    ) -> BufferAllocation<T> {
        let device = &self.context.device;
        let device_size = (max_size * std::mem::size_of::<T>()) as vk::DeviceSize;

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

        let memory_requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let size = memory_requirements.size;

        self.allocate_buffer_inner(align, handle, size)
    }

    fn allocate_buffer_inner<T: Sized>(
        &mut self,
        align: u64,
        handle: vk::Buffer,
        size: u64,
    ) -> BufferAllocation<T> {
        let device = &self.context.device;
        let reserve = size + (align - 1);

        // Allocate an offset into our device local memory
        let offset = self
            .offset_allocator
            .allocate(reserve as u32)
            .expect("Unable to allocate memory. This should be impossible!");

        let label = format!(
            "[lazy_vulkan] BufferAllocation<{}> at offset {}",
            std::any::type_name::<T>(),
            offset.offset
        );
        self.context.set_debug_label(handle, &label);

        // Align the offset before binding
        let bind_offset = align_offset(align, offset);

        let pad = bind_offset - offset.offset as u64;
        println!(
            "size: {size}, align: {align}, reserve: {reserve}, offset: {}, bind_offset: {bind_offset}, pad: {pad}",
            offset.offset
        );

        // Bind its memory
        unsafe { device.bind_buffer_memory(handle, self.backend.device_memory(), bind_offset) }
            .unwrap();

        // Get its device address
        let device_address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(handle))
        };

        BufferAllocation {
            size,
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
        let align = memory_requirements.alignment;
        let reserve = size + (align - 1);

        // Allocate an offset into our device local memory
        let global_offset = self
            .offset_allocator
            .allocate(reserve as u32)
            .expect("Unable to allocate memory. This should be impossible!");

        let bind_offset = align_offset(align, global_offset);

        // Bind the image to the memory at this offset
        unsafe { device.bind_image_memory(image, self.backend.device_memory(), bind_offset) }
            .unwrap();

        // Stage the transfer
        let (ours, theirs) = TransferToken::create_pair();

        if !data.is_empty() {
            let staging_buffer_offset = self.staging_buffer.stage(data);
            self.pending_transfers.push(PendingTransfer {
                destination: TransferDestination::Image(image, extent),
                transfer_size: data.len() as _,
                transfer_token: ours,
                staging_buffer_offset,
                global_offset,
                allocation_offset: 0,
            });
        } else {
            // No data? Nothing to do
            ours.mark_completed();
            theirs.mark_completed();
        }

        theirs
    }

    pub fn append_to_buffer<T: bytemuck::Pod>(
        &mut self,
        data: &[T],
        allocation: &mut BufferAllocation<T>,
    ) -> TransferToken {
        let bytes = bytemuck::cast_slice(data);
        let staging_buffer_offset = self.staging_buffer.stage(bytes);

        let (ours, theirs) = TransferToken::create_pair();

        self.pending_transfers.push(PendingTransfer {
            destination: TransferDestination::Buffer(allocation.handle),
            staging_buffer_offset,
            transfer_size: bytes.len() as _,
            global_offset: allocation.global_offset,
            transfer_token: ours,
            allocation_offset: allocation.len(),
        });

        allocation.len += data.len();

        theirs
    }

    pub fn execute_transfers(&mut self, command_buffer: vk::CommandBuffer) {
        self.context
            .begin_marker("Execute Transfers", glam::vec4(0., 0., 1., 1.));
        self.backend.execute_transfers(
            &self.context,
            std::mem::take(&mut self.pending_transfers),
            &mut self.staging_buffer,
            command_buffer,
        );
        self.context.end_marker();
    }

    pub fn upload_to_slab<T: bytemuck::Pod + Debug>(&mut self, data: &[T]) -> SlabUpload<T> {
        let bytes = bytemuck::cast_slice(data);
        let size = bytes.len() as vk::DeviceSize;

        // Allocate an offset into our device local memory
        let global_offset = self
            .offset_allocator
            .allocate(size as u32)
            .expect("Unable to allocate memory. This should be impossible!");

        let staging_buffer_offset = self.staging_buffer.stage(bytes);
        let device_address = self.backend.get_device_address(global_offset);

        let (ours, theirs) = TransferToken::create_pair();

        self.pending_transfers.push(PendingTransfer {
            destination: TransferDestination::Slab,
            staging_buffer_offset,
            transfer_size: bytes.len() as _,
            global_offset,
            transfer_token: ours,
            allocation_offset: 0,
        });

        SlabUpload {
            device_address,
            size,
            offset: global_offset,
            transfer_token: theirs,
            _phantom: Default::default(),
        }
    }

    pub fn free<T: Sized>(&mut self, _allocation: BufferAllocation<T>) {
        unimplemented!("Free is not yet implemented");
    }

    pub fn free_from_slab<T: Sized>(&mut self, _allocation: BufferAllocation<T>) {
        unimplemented!("Free is not yet implemented");
    }

    pub unsafe fn append_unsafe<T: Copy>(
        &mut self,
        data: &[T],
        allocation: &mut BufferAllocation<T>,
    ) -> TransferToken {
        let bytes =
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data));
        let staging_buffer_offset = self.staging_buffer.stage(bytes);

        let (ours, theirs) = TransferToken::create_pair();

        self.pending_transfers.push(PendingTransfer {
            destination: TransferDestination::Buffer(allocation.handle),
            staging_buffer_offset,
            transfer_size: bytes.len() as _,
            global_offset: allocation.global_offset,
            transfer_token: ours,
            allocation_offset: allocation.len(),
        });

        allocation.len += data.len();

        theirs
    }
}

fn align_offset(align: u64, offset: offset_allocator::Allocation) -> u64 {
    (offset.offset as u64 + align - 1) & !(align - 1)
}

#[derive(Clone)]
pub struct SlabUpload<T> {
    pub device_address: vk::DeviceAddress,
    pub size: vk::DeviceSize,
    pub transfer_token: TransferToken,
    #[allow(unused)]
    offset: offset_allocator::Allocation,
    _phantom: PhantomData<T>,
}

#[derive(Clone, Debug, Default)]
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
    staging_buffer_offset: usize, // offset within the staging buffer
    global_offset: offset_allocator::Allocation, // offset into the global memory
    allocation_offset: usize,     // offset within the allocation
    transfer_size: vk::DeviceSize,
    transfer_token: TransferToken,
}

enum TransferDestination {
    Buffer(vk::Buffer),
    Image(vk::Image, vk::Extent2D),
    Slab,
}

pub struct PendingFree;
pub struct BufferAllocation<T> {
    #[allow(unused)]
    pub size: vk::DeviceSize,
    pub device_address: vk::DeviceAddress,
    pub handle: vk::Buffer,
    len: usize, // number of `T`s in this buffer
    #[allow(unused)]
    global_offset: offset_allocator::Allocation, // offset into the global memory
    _phantom: PhantomData<T>,
}
impl<T> BufferAllocation<T>
where
    T: Copy,
{
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }
    pub unsafe fn append_unsafe(&mut self, data: &[T], allocator: &mut Allocator) {
        allocator.append_unsafe(data, self);
    }
}

impl<T> BufferAllocation<T>
where
    T: bytemuck::Pod,
{
    pub fn append(&mut self, data: &[T], allocator: &mut Allocator) {
        allocator.append_to_buffer(data, self);
    }
}
