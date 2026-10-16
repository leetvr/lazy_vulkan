mod device_buffer;
mod error;
pub use error::AllocationError;
mod memory_blocks;
mod staging_buffer;
use memory_blocks::MemoryBlocks;
use staging_buffer::StagingBuffer;
use std::{
    collections::HashMap,
    fmt::Debug,
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use ash::vk;

use super::context::Context;

pub struct Allocator {
    pub context: Arc<Context>,
    pub pending_transfers: Vec<PendingTransfer>,
    pub pending_frees: Vec<PendingFree>,
    submitted_frees: Vec<PendingFree>,
    memory_blocks: MemoryBlocks,
    staging_buffer: StagingBuffer,
    pending_tokens: Vec<TransferToken>,
    image_allocations: HashMap<vk::Image, Offset>,
}

impl Allocator {
    pub fn new(
        context: Arc<Context>,
        pool_bytes: u64,
        staging_bytes: u64,
    ) -> Result<Self, AllocationError> {
        let mut memory_blocks = MemoryBlocks::new(context.clone(), pool_bytes)?;
        let staging_buffer = match StagingBuffer::new(&context, staging_bytes) {
            Ok(staging) => staging,
            Err(error) => {
                unsafe { memory_blocks.destroy() };
                return Err(error);
            }
        };

        Ok(Self {
            memory_blocks,
            context,
            pending_frees: Default::default(),
            submitted_frees: Default::default(),
            pending_transfers: Default::default(),
            staging_buffer,
            pending_tokens: Default::default(),
            image_allocations: Default::default(),
        })
    }

    /// Allocates a buffer of `max_size`.
    ///
    /// # NOTE
    /// Here `max_size` refers to the number of `T`s that can fit in this buffer, NOT bytes.
    pub fn allocate_buffer<T: Sized>(
        &mut self,
        max_size: usize,
        usage_flags: vk::BufferUsageFlags,
    ) -> Result<BufferAllocation<T>, AllocationError> {
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
        }?;

        self.allocate_buffer_inner(1, handle)
    }

    pub fn allocate_buffer_with_alignment<T: Sized>(
        &mut self,
        max_size: usize,
        align: u64,
        usage_flags: vk::BufferUsageFlags,
    ) -> Result<BufferAllocation<T>, AllocationError> {
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
        }?;

        self.allocate_buffer_inner(align, handle)
    }

    pub fn allocate_readback(&mut self, size: usize) -> Result<ReadbackBuffer, AllocationError> {
        let device = &self.context.device;
        let handle = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as vk::DeviceSize)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
        }?;
        self.context
            .set_debug_label(handle, "[lazy_vulkan] Readback Buffer");

        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let memory_type_index = match self.context.find_memory_type_index(
            &requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Some(index) => index,
            None => {
                unsafe { device.destroy_buffer(handle, None) };
                return Err(AllocationError::NoCompatibleMemoryType);
            }
        };
        let memory = match unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type_index),
                None,
            )
        } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_buffer(handle, None) };
                return Err(error.into());
            }
        };
        let result = (|| {
            unsafe { device.bind_buffer_memory(handle, memory, 0) }?;
            let ptr = std::ptr::NonNull::new(unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }?)
            .unwrap()
            .cast();
            Ok(ReadbackBuffer {
                handle,
                memory,
                ptr,
                size,
            })
        })();
        if result.is_err() {
            unsafe {
                device.destroy_buffer(handle, None);
                device.free_memory(memory, None);
            }
        }
        result
    }

    fn buffer_requirements(&self, buffer: vk::Buffer) -> (vk::MemoryRequirements, bool) {
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        unsafe {
            self.context.device.get_buffer_memory_requirements2(
                &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
                &mut requirements,
            );
        }
        (
            requirements.memory_requirements,
            dedicated.requires_dedicated_allocation == vk::TRUE,
        )
    }

    fn allocate_buffer_inner<T: Sized>(
        &mut self,
        align: u64,
        handle: vk::Buffer,
    ) -> Result<BufferAllocation<T>, AllocationError> {
        let (requirements, dedicated) = self.buffer_requirements(handle);
        let result = (|| {
            let offset = self
                .memory_blocks
                .allocate(requirements, align, dedicated)?;
            let device = &self.context.device;
            if let Err(error) = unsafe {
                device.bind_buffer_memory(
                    handle,
                    self.memory_blocks.memory(offset),
                    offset.total_offset(),
                )
            } {
                self.memory_blocks.free(offset);
                return Err(error.into());
            }
            self.context.set_debug_label(
                handle,
                &format!(
                    "[lazy_vulkan] BufferAllocation<{}>",
                    std::any::type_name::<T>()
                ),
            );
            let device_address = unsafe {
                device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(handle),
                )
            };
            Ok(BufferAllocation {
                size: requirements.size,
                device_address,
                len: 0,
                handle,
                global_offset: offset,
                _phantom: PhantomData,
            })
        })();
        if result.is_err() {
            unsafe { self.context.device.destroy_buffer(handle, None) };
        }
        result
    }

    pub fn allocate_image(
        &mut self,
        data: &[u8],
        extent: vk::Extent2D,
        image: vk::Image,
    ) -> Result<TransferToken, AllocationError> {
        self.allocate_image_with_mip_levels(data, extent, 1, image)
    }

    pub fn allocate_image_with_mip_levels(
        &mut self,
        data: &[u8],
        extent: vk::Extent2D,
        mip_levels: u32,
        image: vk::Image,
    ) -> Result<TransferToken, AllocationError> {
        self.allocate_image_with_mip_data(data, extent, mip_levels, image, None)
    }

    pub(crate) fn allocate_image_with_mip_data(
        &mut self,
        data: &[u8],
        extent: vk::Extent2D,
        mip_levels: u32,
        image: vk::Image,
        mip_offsets: Option<Vec<vk::DeviceSize>>,
    ) -> Result<TransferToken, AllocationError> {
        self.staging_buffer.check_capacity(data.len())?;
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        unsafe {
            self.context.device.get_image_memory_requirements2(
                &vk::ImageMemoryRequirementsInfo2::default().image(image),
                &mut requirements,
            );
        }
        let requirements = requirements.memory_requirements;
        let global_offset = self.memory_blocks.allocate(
            requirements,
            requirements.alignment,
            dedicated.requires_dedicated_allocation == vk::TRUE,
        )?;
        if let Err(error) = unsafe {
            self.context.device.bind_image_memory(
                image,
                self.memory_blocks.memory(global_offset),
                global_offset.total_offset(),
            )
        } {
            self.memory_blocks.free(global_offset);
            return Err(error.into());
        }
        self.image_allocations.insert(image, global_offset);
        self.stage_image_data(data, extent, mip_levels, image, mip_offsets)
    }

    pub(crate) fn stage_image_data(
        &mut self,
        data: &[u8],
        extent: vk::Extent2D,
        mip_levels: u32,
        image: vk::Image,
        mip_offsets: Option<Vec<vk::DeviceSize>>,
    ) -> Result<TransferToken, AllocationError> {
        let (ours, theirs) = TransferToken::create_pair();

        self.staging_buffer.check_capacity(data.len())?;
        let global_offset = self.image_allocations[&image];
        if !data.is_empty() {
            let staging_buffer_offset = self.staging_buffer.stage(data);
            self.pending_transfers.push(PendingTransfer {
                destination: TransferDestination::Image {
                    image,
                    extent,
                    mip_levels,
                    mip_offsets,
                },
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

        Ok(theirs)
    }

    pub fn append_to_buffer<T: bytemuck::Pod>(
        &mut self,
        data: &[T],
        allocation: &mut BufferAllocation<T>,
    ) -> TransferToken {
        let bytes = bytemuck::cast_slice(data);
        self.append_to_buffer_inner(bytes, allocation)
    }

    pub unsafe fn append_unsafe<T: Copy>(
        &mut self,
        data: &[T],
        allocation: &mut BufferAllocation<T>,
    ) -> TransferToken {
        let bytes =
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data));
        self.append_to_buffer_inner(bytes, allocation)
    }

    fn append_to_buffer_inner<T: Copy>(
        &mut self,
        bytes: &[u8],
        allocation: &mut BufferAllocation<T>,
    ) -> TransferToken {
        let staging_buffer_offset = self.staging_buffer.stage(bytes);

        let (ours, theirs) = TransferToken::create_pair();

        self.pending_transfers.push(PendingTransfer {
            destination: TransferDestination::Buffer(allocation.handle),
            staging_buffer_offset,
            transfer_size: bytes.len() as _,
            global_offset: allocation.global_offset,
            transfer_token: ours,
            allocation_offset: allocation.current_size() as usize,
        });

        allocation.len += bytes.len() as vk::DeviceSize;

        theirs
    }

    pub fn execute_transfers(&mut self, command_buffer: vk::CommandBuffer) {
        self.context
            .begin_marker("Execute Transfers", glam::vec4(0., 0., 1., 1.));

        for transfer in &self.pending_transfers {
            self.pending_tokens.push(transfer.transfer_token.clone());
        }

        device_buffer::execute_transfers(
            &self.context,
            &self.memory_blocks,
            std::mem::take(&mut self.pending_transfers),
            &mut self.staging_buffer,
            command_buffer,
        );

        self.context.end_marker();
    }

    pub fn check_staging_capacity(&self, bytes: usize) -> Result<(), AllocationError> {
        self.staging_buffer.check_capacity(bytes)
    }

    /// This should only be called when all transfers issued with `execute_transfers` have been
    /// actually completed.
    pub fn transfers_complete(&mut self) {
        for token in self.pending_tokens.drain(..) {
            token.mark_completed();
        }

        if let Some(start) = self
            .pending_transfers
            .iter()
            .map(|t| t.staging_buffer_offset)
            .min()
        {
            self.staging_buffer.retain_from(start);
            for transfer in &mut self.pending_transfers {
                transfer.staging_buffer_offset -= start;
            }
        } else {
            self.staging_buffer.clear();
        }
    }

    pub fn upload_to_slab<T: bytemuck::Pod + Debug>(
        &mut self,
        data: &[T],
    ) -> Result<SlabUpload<T>, AllocationError> {
        let bytes = bytemuck::cast_slice(data);
        let size = bytes.len() as vk::DeviceSize;

        // Allocate an offset into our device local memory
        const SLAB_ALIGNMENT: u64 = 8;
        self.staging_buffer.check_capacity(bytes.len())?;
        let global_offset = self.memory_blocks.allocate_slab(size, SLAB_ALIGNMENT)?;

        let staging_buffer_offset = self.staging_buffer.stage(bytes);
        let device_address = self.memory_blocks.slab_address(global_offset);

        let (ours, theirs) = TransferToken::create_pair();

        self.pending_transfers.push(PendingTransfer {
            destination: TransferDestination::Slab,
            staging_buffer_offset,
            transfer_size: bytes.len() as _,
            global_offset,
            transfer_token: ours,
            allocation_offset: 0,
        });

        Ok(SlabUpload {
            device_address,
            size,
            offset: global_offset,
            transfer_token: theirs,
            _phantom: Default::default(),
        })
    }

    // Caller has destroyed the image after all uses completed.
    pub(crate) unsafe fn release_image_memory(&mut self, image: vk::Image) {
        let offset = self
            .image_allocations
            .remove(&image)
            .expect("owned image allocation");
        self.memory_blocks.free(offset);
    }

    pub fn free<T: Sized>(&mut self, allocation: BufferAllocation<T>) {
        self.pending_frees.push(PendingFree::Buffer {
            handle: allocation.handle,
            offset: allocation.global_offset,
        });
    }

    pub fn free_from_slab<T>(&mut self, allocation: SlabUpload<T>) {
        self.pending_frees
            .push(PendingFree::Slab(allocation.offset));
    }

    pub fn free_readback(&mut self, readback: ReadbackBuffer) {
        self.pending_frees.push(PendingFree::Readback(readback));
    }

    /// Allocates an acceleration structure of `size` bytes over the pool. Record its build
    /// with `Context::acceleration_structure_pfn` before anything traverses it.
    #[cfg(feature = "rtx_on")]
    pub fn allocate_acceleration_structure(
        &mut self,
        ty: vk::AccelerationStructureTypeKHR,
        size: vk::DeviceSize,
    ) -> Result<AccelerationStructureAllocation, AllocationError> {
        // The backing buffer offset must be a multiple of 256.
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkAccelerationStructureCreateInfoKHR.html
        let buffer = self.allocate_buffer_with_alignment::<u8>(
            size as usize,
            256,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        )?;
        let handle = match unsafe {
            self.context
                .acceleration_structure_pfn
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(buffer.handle)
                        .size(size)
                        .ty(ty),
                    None,
                )
        } {
            Ok(handle) => handle,
            Err(error) => {
                self.free(buffer);
                return Err(error.into());
            }
        };
        self.context.set_debug_label(
            handle,
            &format!("[lazy_vulkan] AccelerationStructure {ty:?}"),
        );
        let device_address = unsafe {
            self.context
                .acceleration_structure_pfn
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(handle),
                )
        };
        Ok(AccelerationStructureAllocation {
            handle,
            device_address,
            size,
            buffer: buffer.handle,
            global_offset: buffer.global_offset,
        })
    }

    #[cfg(feature = "rtx_on")]
    pub fn free_acceleration_structure(&mut self, allocation: AccelerationStructureAllocation) {
        self.pending_frees.push(PendingFree::AccelerationStructure {
            handle: allocation.handle,
            buffer: allocation.buffer,
            offset: allocation.global_offset,
        });
    }

    pub(crate) fn free_image(&mut self, image: crate::Image) {
        let offset = self
            .image_allocations
            .remove(&image.handle)
            .expect("owned image allocation");
        self.pending_frees
            .push(PendingFree::Image { image, offset });
    }

    // Call after submitting the batch covering the final uses of the pending frees.
    pub fn submission_submitted(&mut self) {
        self.submitted_frees.append(&mut self.pending_frees);
    }

    // Call only after every submission registered above has completed.
    // Returns sampled texture IDs that can now be reused by ImageManager.
    pub unsafe fn submission_complete(&mut self) -> Vec<u32> {
        let mut texture_ids = Vec::new();
        for resource in std::mem::take(&mut self.submitted_frees) {
            let offset = match resource {
                PendingFree::Buffer { handle, offset } => {
                    self.context.device.destroy_buffer(handle, None);
                    offset
                }
                PendingFree::Slab(offset) => offset,
                PendingFree::Readback(readback) => {
                    self.context.device.destroy_buffer(readback.handle, None);
                    self.context.device.unmap_memory(readback.memory);
                    self.context.device.free_memory(readback.memory, None);
                    continue;
                }
                PendingFree::Image { image, offset } => {
                    self.context.device.destroy_image_view(image.view, None);
                    self.context.device.destroy_sampler(image.sampler, None);
                    self.context.device.destroy_image(image.handle, None);
                    if image.id != u32::MAX {
                        texture_ids.push(image.id);
                    }
                    offset
                }
                #[cfg(feature = "rtx_on")]
                PendingFree::AccelerationStructure {
                    handle,
                    buffer,
                    offset,
                } => {
                    self.context
                        .acceleration_structure_pfn
                        .destroy_acceleration_structure(handle, None);
                    self.context.device.destroy_buffer(buffer, None);
                    offset
                }
            };
            self.memory_blocks.free(offset);
        }
        texture_ids
    }
    // Call after destroying resources and completing all GPU work using the allocator.
    pub unsafe fn destroy(&mut self) {
        self.memory_blocks.destroy();
        self.staging_buffer.destroy(&self.context);
    }
}

#[derive(Clone, Copy)]
pub struct Offset {
    pub block: usize,
    pub allocation: offset_allocator::Allocation,
    pub unit_size: u64,
    pub bind_offset: vk::DeviceSize,
}

impl Offset {
    pub fn total_offset(&self) -> vk::DeviceSize {
        u64::from(self.allocation.offset) * self.unit_size + self.bind_offset
    }
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
    Image {
        image: vk::Image,
        extent: vk::Extent2D,
        mip_levels: u32,
        mip_offsets: Option<Vec<vk::DeviceSize>>,
    },
    Slab,
}

pub enum PendingFree {
    Buffer {
        handle: vk::Buffer,
        offset: Offset,
    },
    Slab(Offset),
    Readback(ReadbackBuffer),
    Image {
        image: crate::Image,
        offset: Offset,
    },
    #[cfg(feature = "rtx_on")]
    AccelerationStructure {
        handle: vk::AccelerationStructureKHR,
        buffer: vk::Buffer,
        offset: Offset,
    },
}

pub struct ReadbackBuffer {
    pub handle: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: std::ptr::NonNull<u8>,
    size: usize,
}

impl ReadbackBuffer {
    pub unsafe fn read_vec<T: bytemuck::Pod>(&self) -> Vec<T> {
        assert!(std::mem::size_of::<T>() > 0 && self.size % std::mem::size_of::<T>() == 0);
        let bytes = std::slice::from_raw_parts(self.ptr.as_ptr(), self.size);
        bytes
            .chunks_exact(std::mem::size_of::<T>())
            .map(bytemuck::pod_read_unaligned)
            .collect()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// # Safety
    /// The caller must ensure that GPU writes to this buffer have completed.
    pub unsafe fn read<T: bytemuck::Pod>(&self) -> T {
        assert_eq!(self.size, std::mem::size_of::<T>());
        let bytes = unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) };
        bytemuck::pod_read_unaligned(bytes)
    }
}

pub struct BufferAllocation<T> {
    #[allow(unused)]
    pub size: vk::DeviceSize,
    pub device_address: vk::DeviceAddress,
    pub handle: vk::Buffer,
    len: vk::DeviceSize,   // number of bytes in the buffer
    global_offset: Offset, // offset into the global memory
    _phantom: PhantomData<T>,
}

impl<T> BufferAllocation<T>
where
    T: Copy,
{
    /// Returns the number of `T`s in the buffer
    pub fn len(&self) -> usize {
        self.len as usize / std::mem::size_of::<T>()
    }

    /// Returns the number of *bytes* in the buffer
    pub fn current_size(&self) -> vk::DeviceSize {
        self.len
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub unsafe fn append_unsafe(&mut self, data: &[T], allocator: &mut Allocator) {
        allocator.append_unsafe(data, self);
    }

    pub fn tip_address(&self) -> vk::DeviceAddress {
        self.device_address + self.current_size()
    }
}

impl<T> BufferAllocation<T>
where
    T: bytemuck::Pod,
{
    pub fn append(&mut self, data: &[T], allocator: &mut Allocator) {
        allocator.append_to_buffer(data, self);
    }

    pub fn append_one(&mut self, data: &T, allocator: &mut Allocator) {
        allocator.append_to_buffer(std::slice::from_ref(data), self);
    }
}

/// An acceleration structure and the pool buffer it lives in.
#[cfg(feature = "rtx_on")]
pub struct AccelerationStructureAllocation {
    pub handle: vk::AccelerationStructureKHR,
    /// The acceleration structure's own address: what instances and shaders reference.
    pub device_address: vk::DeviceAddress,
    pub size: vk::DeviceSize,
    /// The backing buffer, for memory barriers.
    pub buffer: vk::Buffer,
    global_offset: Offset,
}

#[cfg(test)]
mod tests {
    use crate::{Context, Core, LazyVulkan};
    use ash::vk;
    use std::{sync::Arc, u64};

    const READBACK_MEMORY_SIZE: u64 = 500 << 20;

    #[cfg(feature = "rtx_on")]
    #[test]
    fn acceleration_structures_allocate_over_the_pool_and_free() {
        let mut lazy_vulkan = get_vulkan();
        let allocator = &mut lazy_vulkan.renderer.allocator;
        let allocation = allocator
            .allocate_acceleration_structure(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, 4096)
            .unwrap();
        assert_ne!(allocation.device_address, 0);
        allocator.free_acceleration_structure(allocation);
        allocator.submission_submitted();
        unsafe { allocator.submission_complete() };
    }

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

        let mut buffer_a = allocator
            .allocate_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC)
            .unwrap();
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

        let readback = allocator.allocate_readback(data_a.len()).unwrap();
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

        let readback_data = unsafe { readback.read::<[u8; 4]>() };

        assert_eq!(data_a, readback_data);
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

        let mut buffer_a = allocator
            .allocate_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC)
            .unwrap();
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

        let mut buffer_b = allocator
            .allocate_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC)
            .unwrap();
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
    fn test_append_buffer() {
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

        let mut buffer_a = allocator
            .allocate_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC)
            .unwrap();
        let data_a: [u64; 4] = [1, 2, 3, 4];
        buffer_a.append(&data_a, allocator);
        assert_eq!(buffer_a.len(), data_a.len());
        assert_eq!(
            buffer_a.current_size() as usize,
            std::mem::size_of_val(&data_a)
        );

        let data_b: [u64; 4] = [5, 6, 7, 8];
        buffer_a.append(&data_b, allocator);
        assert_eq!(buffer_a.len(), data_a.len() + data_b.len());
        assert_eq!(
            buffer_a.current_size() as usize,
            std::mem::size_of_val(&data_a) + std::mem::size_of_val(&data_b)
        );

        allocator.execute_transfers(command_buffer);
        let total_size = std::mem::size_of_val(&data_a) + std::mem::size_of_val(&data_b);

        // Barrier
        unsafe {
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .buffer(buffer_a.handle)
                        .size(total_size as u64)
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
                &[vk::BufferCopy::default().size(total_size as _)],
            );
        }

        // Submit and wait
        submit_and_wait(context, command_buffer);
        allocator.transfers_complete();

        let readback_data = unsafe {
            std::slice::from_raw_parts(readback.ptr.as_ptr().cast(), data_a.len() + data_b.len())
        };

        assert_eq!(&data_a, &readback_data[..data_a.len()]);
        assert_eq!(
            &data_b,
            &readback_data[data_a.len()..data_a.len() + data_b.len()]
        );
    }

    #[test]
    fn test_append_buffer_unsafe() {
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

        let mut buffer_a = allocator
            .allocate_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC)
            .unwrap();
        let data_a: [u64; 4] = [1, 2, 3, 4];
        unsafe { buffer_a.append_unsafe(&data_a, allocator) };

        let data_b: [u64; 4] = [5, 6, 7, 8];
        unsafe { buffer_a.append_unsafe(&data_b, allocator) };

        allocator.execute_transfers(command_buffer);
        let total_size = std::mem::size_of_val(&data_a) + std::mem::size_of_val(&data_b);

        // Barrier
        unsafe {
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .buffer(buffer_a.handle)
                        .size(total_size as u64)
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
                &[vk::BufferCopy::default().size(total_size as _)],
            );
        }

        // Submit and wait
        submit_and_wait(context, command_buffer);
        allocator.transfers_complete();

        let readback_data = unsafe {
            std::slice::from_raw_parts(readback.ptr.as_ptr().cast(), data_a.len() + data_b.len())
        };

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

        let mut buffer_a = allocator
            .allocate_buffer(32, vk::BufferUsageFlags::TRANSFER_SRC)
            .unwrap();
        let data_a: [u8; 4] = [1, 2, 3, 4];
        buffer_a.append(&data_a, allocator);

        let mut buffer_b = allocator
            .allocate_buffer_with_alignment(1024, 64, vk::BufferUsageFlags::TRANSFER_SRC)
            .unwrap();
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

    #[test]
    fn attachment_memory_types_match_resource_requirements() {
        let mut lazy_vulkan = get_vulkan();
        let renderer = &mut lazy_vulkan.renderer;
        let extent = vk::Extent2D {
            width: 16,
            height: 16,
        };
        for (format, usage) in [
            (
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ),
            (
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
        ] {
            let image = renderer
                .create_image("attachment allocation test", format, extent, [], usage)
                .unwrap();
            let requirements = unsafe {
                renderer
                    .context
                    .device
                    .get_image_memory_requirements(image.handle)
            };
            let offset = renderer.allocator.image_allocations[&image.handle];
            let block = &renderer.allocator.memory_blocks.blocks[offset.block];
            // VkBindImageMemory validation checks the selected memory type and alignment.
            assert_eq!(offset.total_offset() % requirements.alignment, 0);
            assert_ne!(
                requirements.memory_type_bits & (1 << block.memory_type_index),
                0
            );
            assert!(image.transfer_complete.is_complete());
        }
    }

    #[test]
    fn test_mipmapped_image_upload_completes() {
        let mut lazy_vulkan = get_vulkan();
        let extent = vk::Extent2D {
            width: 8,
            height: 4,
        };
        let pixels = vec![128_u8; (extent.width * extent.height * 4) as usize];
        let image = lazy_vulkan
            .renderer
            .create_mipmapped_sampled_image(
                "mip upload test",
                vk::Format::R8G8B8A8_UNORM,
                extent,
                &pixels,
            )
            .unwrap();
        assert_eq!(image.mip_levels, 4);

        let context = &lazy_vulkan.context;
        let command_buffer = context.draw_command_buffer;
        unsafe {
            context
                .device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }
        lazy_vulkan
            .renderer
            .allocator
            .execute_transfers(command_buffer);
        submit_and_wait(context, command_buffer);
        lazy_vulkan.renderer.allocator.transfers_complete();

        assert!(image.transfer_complete.is_complete());
    }

    #[test]
    fn fixed_reservation_exceeds_two_gib_and_stays_reserved() {
        let mut vulkan = get_vulkan_with_pool(6 << 30);
        let context = vulkan.context.clone();
        let blocks = vulkan.renderer.allocator.memory_blocks.blocks.len();
        let mut buffers = Vec::new();
        for value in 0..9_u32 {
            let allocator = &mut vulkan.renderer.allocator;
            let mut buffer = allocator
                .allocate_buffer::<u32>((512 << 20) / 4, vk::BufferUsageFlags::TRANSFER_SRC)
                .unwrap();
            buffer.append(&[value; 4], allocator);
            buffers.push(buffer);
        }
        let readback = vulkan.renderer.allocator.allocate_readback(9 * 16).unwrap();
        let drawable = vulkan.get_drawable();
        vulkan.begin_commands();
        vulkan
            .renderer
            .allocator
            .execute_transfers(context.draw_command_buffer);
        unsafe {
            context.cmd_pipeline_barrier2(
                context.draw_command_buffer,
                &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)]),
            );
            for (index, buffer) in buffers.iter().enumerate() {
                context.device.cmd_copy_buffer(
                    context.draw_command_buffer,
                    buffer.handle,
                    readback.handle,
                    &[vk::BufferCopy::default()
                        .dst_offset(index as u64 * 16)
                        .size(16)],
                );
            }
        }
        for buffer in buffers {
            vulkan.renderer.allocator.free(buffer);
        }
        vulkan.submit_and_present(drawable);
        vulkan.wait_for_previous_frame();
        assert_eq!(
            unsafe { readback.read_vec::<u32>() },
            (0..9_u32).flat_map(|value| [value; 4]).collect::<Vec<_>>()
        );
        vulkan.begin_commands();
        assert_eq!(vulkan.renderer.allocator.memory_blocks.blocks.len(), blocks);
        let entire_first_block = vulkan
            .renderer
            .allocator
            .allocate_buffer::<u8>(512 << 20, vk::BufferUsageFlags::STORAGE_BUFFER)
            .unwrap();
        vulkan.renderer.allocator.free(entire_first_block);
        vulkan.renderer.allocator.free_readback(readback);
        let drawable = vulkan.get_drawable();
        vulkan.submit_and_present(drawable);
        vulkan.begin_commands();
        unsafe { vulkan.renderer.allocator.destroy() };
    }

    #[test]
    fn exhaustion_preserves_reservations_and_release_waits_for_its_upload_submission() {
        let mut vulkan = get_vulkan_with_memory(64 << 10, 20);
        let context = vulkan.context.clone();
        let mut buffer = vulkan
            .renderer
            .allocator
            .allocate_buffer::<u32>(
                (64 << 10) / 4,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .unwrap();
        let address = buffer.device_address;
        let readback = vulkan.renderer.allocator.allocate_readback(20).unwrap();
        assert!(matches!(
            vulkan
                .renderer
                .allocator
                .allocate_buffer::<u32>(4, vk::BufferUsageFlags::STORAGE_BUFFER),
            Err(super::AllocationError::PoolExhausted)
        ));
        let drawable = vulkan.get_drawable();
        vulkan.begin_commands();
        vulkan.submit_and_present(drawable);
        buffer.append(&[1, 2, 3, 4, 5], &mut vulkan.renderer.allocator);
        assert!(matches!(
            vulkan.renderer.allocator.upload_to_slab(&[6_u32]),
            Err(super::AllocationError::StagingExhausted)
        ));
        let handle = buffer.handle;
        vulkan.renderer.allocator.free(buffer);
        let drawable = vulkan.get_drawable();
        vulkan.begin_commands();
        assert!(matches!(
            vulkan
                .renderer
                .allocator
                .allocate_buffer::<u32>(4, vk::BufferUsageFlags::STORAGE_BUFFER),
            Err(super::AllocationError::PoolExhausted)
        ));
        vulkan
            .renderer
            .allocator
            .execute_transfers(context.draw_command_buffer);
        unsafe {
            context.cmd_pipeline_barrier2(
                context.draw_command_buffer,
                &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)]),
            );
            context.device.cmd_copy_buffer(
                context.draw_command_buffer,
                handle,
                readback.handle,
                &[vk::BufferCopy::default().size(20)],
            );
        }
        vulkan.submit_and_present(drawable);
        vulkan.begin_commands();
        assert_eq!(unsafe { readback.read_vec::<u32>() }, [1, 2, 3, 4, 5]);
        let mut replacement = vulkan
            .renderer
            .allocator
            .allocate_buffer::<u32>((64 << 10) / 4, vk::BufferUsageFlags::STORAGE_BUFFER)
            .unwrap();
        assert_eq!(replacement.device_address, address);
        replacement.append(&[5, 4, 3, 2, 1], &mut vulkan.renderer.allocator);
        vulkan
            .renderer
            .allocator
            .execute_transfers(context.draw_command_buffer);
        vulkan.renderer.allocator.free(replacement);
        vulkan.renderer.allocator.free_readback(readback);
        let drawable = vulkan.get_drawable();
        vulkan.submit_and_present(drawable);
        vulkan.begin_commands();
        unsafe { vulkan.renderer.allocator.destroy() };
    }

    #[test]
    fn failed_image_allocation_preserves_pool_and_texture_id() {
        let mut vulkan = get_vulkan_with_pool(64 << 10);
        let whole = vulkan
            .renderer
            .allocator
            .allocate_buffer::<u8>(64 << 10, vk::BufferUsageFlags::STORAGE_BUFFER)
            .unwrap();
        let extent = vk::Extent2D {
            width: 16,
            height: 16,
        };
        assert!(matches!(
            vulkan.renderer.create_image(
                "failed image",
                vk::Format::R8G8B8A8_UNORM,
                extent,
                [],
                vk::ImageUsageFlags::SAMPLED
            ),
            Err(super::AllocationError::PoolExhausted)
        ));
        assert!(vulkan.renderer.allocator.image_allocations.is_empty());
        assert!(vulkan.renderer.allocator.pending_transfers.is_empty());
        let drawable = vulkan.get_drawable();
        vulkan.begin_commands();
        vulkan.renderer.allocator.free(whole);
        vulkan.submit_and_present(drawable);
        vulkan.begin_commands();
        let image = vulkan
            .renderer
            .create_image(
                "successful image",
                vk::Format::R8G8B8A8_UNORM,
                extent,
                [],
                vk::ImageUsageFlags::SAMPLED,
            )
            .unwrap();
        assert_eq!(image.id, 0);
        vulkan
            .renderer
            .image_manager
            .free_image(image.handle, &mut vulkan.renderer.allocator);
        let drawable = vulkan.get_drawable();
        vulkan.submit_and_present(drawable);
        vulkan.begin_commands();
        unsafe { vulkan.renderer.allocator.destroy() };
    }

    #[test]
    fn mixed_resources_reuse_aligned_holes_and_retire_texture_ids() {
        let mut vulkan = get_vulkan_with_pool(64 << 10);
        let anchor = vulkan
            .renderer
            .allocator
            .allocate_buffer::<u8>(13, vk::BufferUsageFlags::STORAGE_BUFFER)
            .unwrap();
        let hole = vulkan
            .renderer
            .allocator
            .allocate_buffer_with_alignment::<u8>(8192, 4096, vk::BufferUsageFlags::STORAGE_BUFFER)
            .unwrap();
        let hole_address = hole.device_address;
        let extent = vk::Extent2D {
            width: 16,
            height: 16,
        };
        let image = vulkan
            .renderer
            .create_image(
                "mixed pool test",
                vk::Format::R8G8B8A8_UNORM,
                extent,
                [],
                vk::ImageUsageFlags::SAMPLED,
            )
            .unwrap();
        let image_offset = vulkan.renderer.allocator.image_allocations[&image.handle];
        assert_eq!(image_offset.block, anchor.global_offset.block);
        let granularity = vulkan
            .context
            .device_properties
            .limits
            .buffer_image_granularity;
        let buffer_end = hole.global_offset.total_offset() + hole.size - 1;
        assert!(buffer_end / granularity < image_offset.total_offset() / granularity);

        let drawable = vulkan.get_drawable();
        vulkan.begin_commands();
        vulkan.renderer.allocator.free(hole);
        vulkan.submit_and_present(drawable);
        vulkan.begin_commands();
        let replacement = vulkan
            .renderer
            .allocator
            .allocate_buffer_with_alignment::<u8>(8192, 4096, vk::BufferUsageFlags::STORAGE_BUFFER)
            .unwrap();
        assert_eq!(replacement.device_address, hole_address);
        assert_eq!(replacement.global_offset.total_offset() % 4096, 0);
        vulkan
            .renderer
            .image_manager
            .free_image(image.handle, &mut vulkan.renderer.allocator);
        let next_image = vulkan
            .renderer
            .create_image(
                "pending ID test",
                vk::Format::R8G8B8A8_UNORM,
                extent,
                [],
                vk::ImageUsageFlags::SAMPLED,
            )
            .unwrap();
        assert_ne!(image.id, next_image.id);
        vulkan.renderer.allocator.free(anchor);
        vulkan.renderer.allocator.free(replacement);
        let drawable = vulkan.get_drawable();
        vulkan.submit_and_present(drawable);
        vulkan.begin_commands();
        let reused_image = vulkan
            .renderer
            .create_image(
                "reused ID test",
                vk::Format::R8G8B8A8_UNORM,
                extent,
                [],
                vk::ImageUsageFlags::SAMPLED,
            )
            .unwrap();
        assert_eq!(image.id, reused_image.id);
        vulkan
            .renderer
            .image_manager
            .free_image(next_image.handle, &mut vulkan.renderer.allocator);
        vulkan
            .renderer
            .image_manager
            .free_image(reused_image.handle, &mut vulkan.renderer.allocator);
        let drawable = vulkan.get_drawable();
        vulkan.submit_and_present(drawable);
        vulkan.begin_commands();
        assert_eq!(vulkan.renderer.allocator.memory_blocks.blocks.len(), 1);
        unsafe { vulkan.renderer.allocator.destroy() };
    }

    #[test]
    fn slab_uploads_roundtrip_across_reserved_blocks() {
        let core = Arc::new(Core::headless());
        let mut context = Context::new_headless(&core);
        context.max_memory_allocation_size = 64 << 10;
        let context = Arc::new(context);
        let mut vulkan: LazyVulkan<()> = LazyVulkan::headless(
            core,
            context.clone(),
            vk::Extent2D {
                width: 1,
                height: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            128 << 10,
            64 << 10,
        )
        .unwrap();
        let first_data = vec![0x11223344_u32; 10240];
        let second_data = vec![0x55667788_u32; 10240];
        let first = vulkan
            .renderer
            .allocator
            .upload_to_slab(&first_data)
            .unwrap();
        let drawable = vulkan.get_drawable();
        vulkan.begin_commands();
        vulkan
            .renderer
            .allocator
            .execute_transfers(context.draw_command_buffer);
        vulkan.submit_and_present(drawable);
        vulkan.wait_for_previous_frame();
        vulkan.renderer.allocator.transfers_complete();
        let second = vulkan
            .renderer
            .allocator
            .upload_to_slab(&second_data)
            .unwrap();
        assert_ne!(first.offset.block, second.offset.block);
        let readback = vulkan
            .renderer
            .allocator
            .allocate_readback((first.size + second.size) as usize)
            .unwrap();
        let mut aliases = Vec::new();
        for upload in [&first, &second] {
            // Read the slab allocation through an explicitly bound buffer alias.
            let buffer = unsafe {
                context.device.create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(upload.size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    None,
                )
            }
            .unwrap();
            let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
            let block = &vulkan.renderer.allocator.memory_blocks.blocks[upload.offset.block];
            assert_ne!(
                requirements.memory_type_bits & (1 << block.memory_type_index),
                0
            );
            assert_eq!(upload.offset.total_offset() % requirements.alignment, 0);
            unsafe {
                context.device.bind_buffer_memory(
                    buffer,
                    block.memory,
                    upload.offset.total_offset(),
                )
            }
            .unwrap();
            aliases.push(buffer);
        }
        let drawable = vulkan.get_drawable();
        vulkan.begin_commands();
        vulkan
            .renderer
            .allocator
            .execute_transfers(context.draw_command_buffer);
        unsafe {
            context.cmd_pipeline_barrier2(
                context.draw_command_buffer,
                &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)]),
            );
            for (index, &buffer) in aliases.iter().enumerate() {
                context.device.cmd_copy_buffer(
                    context.draw_command_buffer,
                    buffer,
                    readback.handle,
                    &[vk::BufferCopy::default()
                        .dst_offset(index as u64 * first.size)
                        .size(first.size)],
                );
            }
        }
        vulkan.renderer.allocator.free_from_slab(first);
        vulkan.renderer.allocator.free_from_slab(second);
        vulkan.submit_and_present(drawable);
        vulkan.wait_for_previous_frame();
        assert_eq!(
            unsafe { readback.read_vec::<u32>() },
            first_data
                .into_iter()
                .chain(second_data)
                .collect::<Vec<_>>()
        );
        for buffer in aliases {
            unsafe { context.device.destroy_buffer(buffer, None) };
        }
        vulkan.begin_commands();
        assert_eq!(vulkan.renderer.allocator.memory_blocks.blocks.len(), 2);
        vulkan.renderer.allocator.free_readback(readback);
        let drawable = vulkan.get_drawable();
        vulkan.submit_and_present(drawable);
        vulkan.begin_commands();
        assert_eq!(vulkan.renderer.allocator.memory_blocks.blocks.len(), 2);
        unsafe { vulkan.renderer.allocator.destroy() };
    }

    fn get_vulkan() -> LazyVulkan<()> {
        get_vulkan_with_pool(1 << 20)
    }

    fn get_vulkan_with_pool(pool_bytes: u64) -> LazyVulkan<()> {
        get_vulkan_with_memory(pool_bytes, 1 << 20)
    }

    fn get_vulkan_with_memory(pool_bytes: u64, staging_bytes: u64) -> LazyVulkan<()> {
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
            pool_bytes,
            staging_bytes,
        )
        .unwrap()
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
                    .allocation_size(READBACK_MEMORY_SIZE),
                None,
            )
        }
        .unwrap();

        // Create a readback buffer
        let handle = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(READBACK_MEMORY_SIZE)
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
