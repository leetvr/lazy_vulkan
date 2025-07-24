use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use ash::vk;

use super::context::Context;

const GLOBAL_MEMORY_SIZE: u64 = 2u64 << 30; // 2GB
const STAGING_MEMORY_SIZE: u64 = 100u64 << 20; // 100MB

pub struct Allocator {
    pub context: Arc<Context>,
    #[allow(unused)]
    staging_memory: vk::DeviceMemory,
    staging_buffer: vk::Buffer,
    staging_buffer_size: vk::DeviceSize,
    staging_address: std::ptr::NonNull<u8>,
    pub global_memory: vk::DeviceMemory,
    pub pending_transfers: Vec<PendingTransfer>,
    #[allow(unused)]
    pub pending_frees: Vec<PendingFree>,
    offset_allocator: offset_allocator::Allocator,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub sync2_pfn: ash::khr::synchronization2::Device,
}

impl Allocator {
    pub fn new(context: Arc<Context>, sync2_pfn: ash::khr::synchronization2::Device) -> Self {
        let device = &context.device;
        let memory_properties = &context.memory_properties;

        // Search through the available memory types to find the one we want
        let mut memory_type_index = None;
        let mut memory_heap_index = None;
        for (index, memory_type) in memory_properties.memory_types_as_slice().iter().enumerate() {
            if memory_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            {
                memory_type_index = Some(index as u32);
                memory_heap_index = Some(memory_type.heap_index);
                break;
            }
        }

        let memory_type_index = memory_type_index.expect("No device memory? Impossible");
        let memory_heap_index = memory_heap_index.expect("No device memory? Impossible");

        let device_memory = unsafe {
            println!("Allocating {GLOBAL_MEMORY_SIZE} from memory type / heap : {memory_type_index}, {memory_heap_index}");
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .memory_type_index(memory_type_index)
                    .allocation_size(GLOBAL_MEMORY_SIZE).push_next(&mut vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS)),
                None,
            )
        }
        .unwrap();

        let mut memory_type_index = None;
        let mut memory_heap_index = None;
        for (index, memory_type) in memory_properties.memory_types_as_slice().iter().enumerate() {
            if memory_type.property_flags.contains(
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ) {
                memory_type_index = Some(index as u32);
                memory_heap_index = Some(memory_type.heap_index);
                break;
            }
        }

        // TODO: We can get reaaaaaally fancy on different devices here.
        // Allocate our staging memory
        let staging_memory = unsafe {
            let memory_type_index = memory_type_index.expect("No host local memory? Impossible");
            let memory_heap_index = memory_heap_index.expect("No host local memory? Impossible");


            println!("Allocating {STAGING_MEMORY_SIZE} from memory type / heap : {memory_type_index}, {memory_heap_index}");
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .memory_type_index(memory_type_index)
                    .allocation_size(STAGING_MEMORY_SIZE),
                None,
            )
        }
        .unwrap();

        // Create a staging buffer
        let staging_buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(STAGING_MEMORY_SIZE)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
        }
        .unwrap();

        // Bind its memory
        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }.unwrap();

        // Map its memory
        let staging_address = unsafe {
            std::ptr::NonNull::new_unchecked(
                device
                    .map_memory(
                        staging_memory,
                        0,
                        vk::WHOLE_SIZE,
                        vk::MemoryMapFlags::empty(),
                    )
                    .unwrap() as *mut u8,
            )
        };

        let offset_allocator = offset_allocator::Allocator::new(GLOBAL_MEMORY_SIZE as u32);

        Self {
            context,
            global_memory: device_memory,
            staging_memory,
            staging_buffer,
            staging_buffer_size: 0,
            staging_address,
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
        unsafe { device.bind_buffer_memory(handle, self.global_memory, offset.offset as _) }
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

    /// SAFETY:
    ///
    /// - [`data`] must be POD
    pub fn stage_transfer<T: Sized + PartialEq + Debug>(
        &mut self,
        data: &[T],
        allocation: &mut BufferAllocation<T>,
    ) {
        // **ACHTUNG**:
        // All references to sizes in this function are in terms of *BYTES*, not `[T]`.

        // Step one: copy the data into the staging buffer
        let staging_buffer_offset = self.staging_buffer_size;
        let transfer_size = std::mem::size_of_val(data) as vk::DeviceSize;

        // We get the staging pointer by taking the base address and adding the current size of
        // the buffer.
        let staging_ptr = unsafe {
            self.staging_address
                .add(staging_buffer_offset as usize)
                .as_ptr()
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                staging_ptr,
                transfer_size as usize, // BYTES
            );
        };

        // Step two: record the amount of data transferred
        self.staging_buffer_size += transfer_size;
        allocation.len += data.len();

        // DEBUG: Ensure the data was copied correctly.
        unsafe {
            let staging_data = std::slice::from_raw_parts(staging_ptr as *const T, data.len());
            debug_assert_eq!(staging_data, data);
        };

        // Step three: stage the transfer to device local memory
        self.pending_transfers.push(PendingTransfer {
            destination: allocation.handle,
            staging_buffer_offset,
            transfer_size,
        });
    }

    pub fn execute_transfers(&mut self) {
        let device = &self.context.device;
        let command_buffer = self.context.draw_command_buffer;

        // Step one: record transfers and barriers
        let mut barriers = vec![];

        for PendingTransfer {
            destination,
            staging_buffer_offset,
            transfer_size,
        } in self.pending_transfers.drain(..)
        {
            unsafe {
                device.cmd_copy_buffer(
                    command_buffer,
                    self.staging_buffer,
                    destination,
                    &[vk::BufferCopy::default()
                        .src_offset(staging_buffer_offset)
                        .dst_offset(0)
                        .size(transfer_size)],
                );
            }

            barriers.push(
                vk::BufferMemoryBarrier2::default()
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
                    .buffer(destination)
                    .size(transfer_size),
            )
        }

        // Step two: set the barriers
        unsafe {
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
            );
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            self.sync2_pfn.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
            );
        }

        // Step three: reset the staging buffer
        self.staging_buffer_size = 0;
    }

    #[allow(unused)]
    pub fn free<T: Sized>(&mut self, allocation: BufferAllocation<T>) {}
}

pub struct PendingTransfer {
    destination: vk::Buffer,
    staging_buffer_offset: vk::DeviceSize,
    transfer_size: vk::DeviceSize,
}

pub struct PendingFree;
pub struct BufferAllocation<T> {
    #[allow(unused)]
    pub max_device_size: vk::DeviceSize,
    pub device_address: vk::DeviceAddress,
    len: usize, // number of `T`s in this buffer
    pub handle: vk::Buffer,
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
