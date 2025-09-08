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
        // Allocate an offset into our device local memory
        let offset = self.allocate_offset(size, align);
        let device = &self.context.device;

        let label = format!(
            "[lazy_vulkan] BufferAllocation<{}> at offset {:?}",
            std::any::type_name::<T>(),
            offset.total_offset(),
        );
        self.context.set_debug_label(handle, &label);

        // Bind its memory
        unsafe {
            device.bind_buffer_memory(handle, self.backend.device_memory(), offset.total_offset())
        }
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
        let memory_requirements =
            unsafe { self.context.device.get_image_memory_requirements(image) };
        let size = memory_requirements.size;
        let align = memory_requirements.alignment;

        // Allocate an offset into our device local memory
        let global_offset = self.allocate_offset(size, align);
        let device = &self.context.device;

        // Bind the image to the memory at this offset
        unsafe {
            device.bind_image_memory(
                image,
                self.backend.device_memory(),
                global_offset.total_offset(),
            )
        }
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

    /// This should only be called when all transfers issued with `execute_transfers` have been
    /// actually completed.
    pub fn transfers_complete(&mut self) {
        self.staging_buffer.clear();
    }

    pub fn upload_to_slab<T: bytemuck::Pod + Debug>(&mut self, data: &[T]) -> SlabUpload<T> {
        let bytes = bytemuck::cast_slice(data);
        let size = bytes.len() as vk::DeviceSize;

        // Allocate an offset into our device local memory
        const SLAB_ALIGNMENT: u64 = 8;
        let global_offset = self.allocate_offset(size, SLAB_ALIGNMENT);

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

    fn allocate_offset(&mut self, size: u64, align: u64) -> Offset {
        let allocation = self
            .offset_allocator
            .allocate(size as u32)
            .expect("COULD NOT ALLOCATE AN OFFSET - THIS SHOULD BE IMPOSSIBLE");
        let aligned = align_offset(align, allocation);

        // Happy case: the offset is already aligned!
        if aligned == allocation.offset as u64 {
            log::trace!(
                "[ALIGNED]: offset:{}, align:{align}, size: {size}",
                allocation.offset
            );
            return Offset {
                allocation,
                bind_offset: 0,
            };
        }

        // Not aligned. First, see how much padding we need:
        let padding = align - (allocation.offset as u64 % align);

        log::trace!(
            "[NOT ALIGNED]: offset:{}, align:{align}, pad: {padding}, size: {size}",
            allocation.offset
        );

        // Free the offset we just got
        self.offset_allocator.free(allocation);

        // Ask for a new allocation with the padding we need
        let new_size = (padding + size) as u32;
        let allocation = self
            .offset_allocator
            .allocate(new_size)
            .expect("COULD NOT ALLOCATE AN OFFSET - THIS SHOULD BE IMPOSSIBLE");

        log::trace!(
            "[FIXED]: offset:{}, align:{align}, pad: {padding}, size: {new_size}",
            allocation.offset
        );

        Offset {
            allocation,
            bind_offset: padding,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Offset {
    pub allocation: offset_allocator::Allocation,
    pub bind_offset: vk::DeviceSize,
}

impl Offset {
    pub fn total_offset(&self) -> vk::DeviceSize {
        self.allocation.offset as u64 + self.bind_offset
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
    offset: Offset,
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
    global_offset: Offset,        // offset into the global memory
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
    len: usize,            // number of `T`s in this buffer
    global_offset: Offset, // offset into the global memory
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

#[cfg(test)]
mod tests {
    use crate::{allocator::STAGING_MEMORY_SIZE, Context, Core, LazyVulkan};
    use ash::vk;
    use std::{sync::Arc, u64};

    #[test]
    fn test_allocate_single_buffer_roundtrip() {
        let mut lazy_vulkan = get_vulkan();

        let context = &lazy_vulkan.context;
        let device = &context.device;
        let allocator = &mut lazy_vulkan.renderer.allocator;

        let command_buffer = context.draw_command_buffer;
        unsafe {
            device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .unwrap();

        let mut buffer_a = allocator.allocate_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC);
        let data_a: [u8; 4] = [1, 2, 3, 4];
        buffer_a.append(&data_a, allocator);
        allocator.execute_transfers(command_buffer);
        // Barrier
        unsafe {
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .buffer(buffer_a.handle)
                        .size(data_a.len() as _)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY),
                ]),
            )
        };

        let readback = create_readback_buffer(context);
        unsafe {
            device.cmd_copy_buffer(
                command_buffer,
                buffer_a.handle,
                readback.handle,
                &[vk::BufferCopy::default().size(data_a.len() as u64)],
            );
        }

        // Submit and wait
        submit_and_wait(context, command_buffer);
        allocator.transfers_complete();

        let readback_data =
            unsafe { std::slice::from_raw_parts(readback.ptr.as_ptr(), data_a.len()) };

        assert_eq!(&data_a, readback_data);
    }

    #[test]
    fn test_allocate_multiple_buffers_roundtrip() {
        let mut lazy_vulkan = get_vulkan();

        let context = &lazy_vulkan.context;
        let device = &context.device;
        let allocator = &mut lazy_vulkan.renderer.allocator;

        let command_buffer = context.draw_command_buffer;
        unsafe {
            device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .unwrap();

        let mut buffer_a = allocator.allocate_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC);
        let data_a: [u8; 4] = [1, 2, 3, 4];
        buffer_a.append(&data_a, allocator);

        allocator.execute_transfers(command_buffer);
        // Barrier
        unsafe {
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .buffer(buffer_a.handle)
                        .size(data_a.len() as _)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(
                            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
                        )
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY),
                ]),
            )
        };

        let mut buffer_b = allocator.allocate_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC);
        let data_b: [u8; 4] = [5, 6, 7, 8];
        buffer_b.append(&data_b, allocator);

        allocator.execute_transfers(command_buffer);
        // Barrier
        unsafe {
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .buffer(buffer_b.handle)
                        .size(data_b.len() as _)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(
                            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
                        )
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY),
                ]),
            )
        };

        let readback = create_readback_buffer(context);
        unsafe {
            device.cmd_copy_buffer(
                command_buffer,
                buffer_a.handle,
                readback.handle,
                &[vk::BufferCopy::default().size(data_a.len() as u64)],
            );
            device.cmd_copy_buffer(
                command_buffer,
                buffer_b.handle,
                readback.handle,
                &[vk::BufferCopy::default()
                    .dst_offset(data_a.len() as u64) // IMPORTANT! Offset by how much transferrred so far
                    .size(data_b.len() as u64)],
            );
        }

        // Submit and wait
        submit_and_wait(context, command_buffer);
        allocator.transfers_complete();

        let readback_data = unsafe { std::slice::from_raw_parts(readback.ptr.as_ptr(), 1024) };

        assert_eq!(&data_a, &readback_data[..data_a.len()]);
        assert_eq!(
            &data_b,
            &readback_data[data_a.len()..data_a.len() + data_b.len()]
        );
    }

    #[test]
    fn test_alignment() {
        let mut lazy_vulkan = get_vulkan();

        let context = &lazy_vulkan.context;
        let device = &context.device;
        let allocator = &mut lazy_vulkan.renderer.allocator;

        let command_buffer = context.draw_command_buffer;
        unsafe {
            device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .unwrap();

        let mut buffer_a = allocator.allocate_buffer(32, vk::BufferUsageFlags::TRANSFER_SRC);
        let data_a: [u8; 4] = [1, 2, 3, 4];
        buffer_a.append(&data_a, allocator);

        let mut buffer_b =
            allocator.allocate_buffer_with_alignment(1024, 64, vk::BufferUsageFlags::TRANSFER_SRC);
        let data_b: [u8; 4] = [5, 6, 7, 8];
        buffer_b.append(&data_b, allocator);

        allocator.execute_transfers(command_buffer);
        // Barrier
        unsafe {
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .buffer(buffer_a.handle)
                        .size(data_a.len() as _)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(
                            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
                        )
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY),
                    vk::BufferMemoryBarrier2::default()
                        .buffer(buffer_b.handle)
                        .size(data_b.len() as _)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(
                            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
                        )
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY),
                ]),
            )
        };

        let readback = create_readback_buffer(context);
        unsafe {
            device.cmd_copy_buffer(
                command_buffer,
                buffer_a.handle,
                readback.handle,
                &[vk::BufferCopy::default().size(data_a.len() as u64)],
            );
            device.cmd_copy_buffer(
                command_buffer,
                buffer_b.handle,
                readback.handle,
                &[vk::BufferCopy::default()
                    .dst_offset(data_a.len() as u64) // IMPORTANT! Offset by how much transferrred so far
                    .size(data_b.len() as u64)],
            );
        }

        // Submit and wait
        submit_and_wait(context, command_buffer);
        allocator.transfers_complete();

        let readback_data = unsafe { std::slice::from_raw_parts(readback.ptr.as_ptr(), 1024) };

        assert_eq!(&data_a, &readback_data[..data_a.len()]);
        assert_eq!(
            &data_b,
            &readback_data[data_a.len()..data_a.len() + data_b.len()]
        );
    }

    fn get_vulkan() -> LazyVulkan<()> {
        let core = Arc::new(Core::headless());
        let context = Arc::new(Context::new_headless(&core));
        LazyVulkan::headless(
            core,
            context,
            vk::Extent2D {
                width: 1,
                height: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
        )
    }

    fn submit_and_wait(context: &Context, command_buffer: vk::CommandBuffer) {
        let device = &context.device;
        unsafe {
            device.end_command_buffer(command_buffer).unwrap();
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            device
                .queue_submit(
                    context.graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                    fence,
                )
                .unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        }
    }

    struct ReadbackBuffer {
        handle: vk::Buffer,
        #[allow(unused)]
        memory: vk::DeviceMemory,
        ptr: std::ptr::NonNull<u8>,
    }

    fn create_readback_buffer(context: &Context) -> ReadbackBuffer {
        let device = &context.device;
        let memory_properties = &context.memory_properties;

        // Search through the available memory types to find the one we want
        let mut memory_type_index = None;
        for (index, memory_type) in memory_properties.memory_types_as_slice().iter().enumerate() {
            if memory_type.property_flags.contains(
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ) {
                memory_type_index = Some(index as u32);
                break;
            }
        }

        let memory_type_index = memory_type_index.expect("No global memory? Impossible");

        // Allocate our readback memory
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .memory_type_index(memory_type_index)
                    .allocation_size(STAGING_MEMORY_SIZE),
                None,
            )
        }
        .unwrap();

        // Create a readback buffer
        let handle = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(STAGING_MEMORY_SIZE)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
        }
        .unwrap();

        context.set_debug_label(handle, "[lazy_vulkan] Readback Buffer");

        // Bind its memory
        unsafe { device.bind_buffer_memory(handle, memory, 0) }.unwrap();

        // Map its memory
        let ptr = unsafe {
            std::ptr::NonNull::new_unchecked(
                device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .unwrap() as *mut u8,
            )
        };

        ReadbackBuffer {
            handle,
            memory,
            ptr,
        }
    }
}
