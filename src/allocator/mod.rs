use std::{fmt::Debug, marker::PhantomData, ptr::NonNull, sync::Arc};

use ash::vk;

use super::context::Context;

const GLOBAL_MEMORY_SIZE: u64 = 2u64 << 30; // 2GB
const STAGING_MEMORY_SIZE: u64 = 100u64 << 20; // 100MB

enum AllocatorBackend {
    Discrete(DiscreteAllocator),
    Integrated(IntegratedAllocator),
}

impl AllocatorBackend {
    fn new(context: Arc<Context>) -> AllocatorBackend {
        match context.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => {
                AllocatorBackend::Discrete(DiscreteAllocator::new(context))
            }
            vk::PhysicalDeviceType::INTEGRATED_GPU => {
                AllocatorBackend::Integrated(IntegratedAllocator::new(context))
            }
            _ => unreachable!("Impossible device type"),
        }
    }

    fn device_memory(&self) -> vk::DeviceMemory {
        match self {
            AllocatorBackend::Discrete(discrete_allocator) => discrete_allocator.device_memory,
            AllocatorBackend::Integrated(integrated_allocator) => {
                integrated_allocator.global_memory
            }
        }
    }

    fn stage_transfer(&mut self, data: &[u8]) -> usize {
        match self {
            AllocatorBackend::Discrete(discrete_allocator) => {
                // Step one: copy the data into the staging buffer
                let staging_buffer_offset = discrete_allocator.staging_buffer_size;
                let transfer_size = std::mem::size_of_val(data) as vk::DeviceSize;

                // We get the staging pointer by taking the base address and adding the current size of
                // the buffer.
                let staging_ptr = unsafe {
                    discrete_allocator
                        .staging_address
                        .add(staging_buffer_offset as usize)
                        .as_ptr()
                };

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        staging_ptr,
                        transfer_size as usize,
                    );
                };

                // Step two: record the amount of data transferred
                discrete_allocator.staging_buffer_size += transfer_size;

                // DEBUG: Ensure the data was copied correctly.
                unsafe {
                    let staging_data = std::slice::from_raw_parts(staging_ptr, data.len());
                    debug_assert_eq!(staging_data, data);
                };

                staging_buffer_offset as usize
            }
            AllocatorBackend::Integrated(integrated_allocator) => {
                let offset = integrated_allocator.staging_buffer.len();
                integrated_allocator.staging_buffer.extend_from_slice(data);

                offset
            }
        }
    }

    fn execute_transfer(
        &mut self,
        PendingTransfer {
            destination,
            staging_buffer_offset,
            transfer_size,
            global_offset,
        }: PendingTransfer,
        barriers: &mut Vec<vk::BufferMemoryBarrier2>,
    ) {
        match self {
            AllocatorBackend::Discrete(discrete_allocator) => {
                let device = &discrete_allocator.context.device;
                let command_buffer = discrete_allocator.context.draw_command_buffer;

                unsafe {
                    device.cmd_copy_buffer(
                        command_buffer,
                        discrete_allocator.staging_buffer,
                        destination,
                        &[vk::BufferCopy::default()
                            .src_offset(staging_buffer_offset as _)
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
            AllocatorBackend::Integrated(integrated_allocator) => {
                let source = integrated_allocator.staging_buffer[staging_buffer_offset..].as_ptr();
                // We get the destination pointer by taking the base address and adding the allocated offset
                let destination = unsafe {
                    integrated_allocator
                        .global_ptr
                        .add(global_offset.offset as usize)
                        .as_ptr()
                };

                unsafe {
                    std::ptr::copy_nonoverlapping(source, destination, transfer_size as usize);
                };
            }
        }
    }

    fn transfers_complete(&mut self, barriers: &[vk::BufferMemoryBarrier2]) {
        match self {
            AllocatorBackend::Discrete(discrete_allocator) => {
                discrete_allocator.staging_buffer_size = 0;
                let device = &discrete_allocator.context.device;
                let command_buffer = discrete_allocator.context.draw_command_buffer;

                unsafe {
                    device.cmd_pipeline_barrier2(
                        command_buffer,
                        &vk::DependencyInfo::default().buffer_memory_barriers(barriers),
                    )
                };
            }
            AllocatorBackend::Integrated(integrated_allocator) => {
                integrated_allocator.staging_buffer.clear();
            }
        }
    }
}

pub struct DiscreteAllocator {
    #[allow(unused)]
    staging_memory: vk::DeviceMemory,
    staging_buffer: vk::Buffer,
    staging_buffer_size: vk::DeviceSize,
    staging_address: std::ptr::NonNull<u8>,
    device_memory: vk::DeviceMemory,
    context: Arc<Context>,
}

impl DiscreteAllocator {
    pub fn new(context: Arc<Context>) -> DiscreteAllocator {
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

        Self {
            device_memory,
            staging_memory,
            staging_buffer,
            staging_buffer_size: 0,
            staging_address,
            context,
        }
    }
}

struct IntegratedAllocator {
    staging_buffer: Vec<u8>,
    global_memory: vk::DeviceMemory,
    global_ptr: NonNull<u8>,
}

impl IntegratedAllocator {
    pub fn new(context: Arc<Context>) -> IntegratedAllocator {
        let device = &context.device;
        let memory_properties = &context.memory_properties;

        // Search through the available memory types to find the one we want
        let mut memory_type_index = None;
        let mut memory_heap_index = None;
        for (index, memory_type) in memory_properties.memory_types_as_slice().iter().enumerate() {
            if memory_type.property_flags.contains(
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ) {
                memory_type_index = Some(index as u32);
                memory_heap_index = Some(memory_type.heap_index);
                break;
            }
        }

        let memory_type_index = memory_type_index.expect("No global memory? Impossible");
        let memory_heap_index = memory_heap_index.expect("No global memory? Impossible");

        let global_memory = unsafe {
            println!("Allocating {GLOBAL_MEMORY_SIZE} from memory type / heap : {memory_type_index}, {memory_heap_index}");
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .memory_type_index(memory_type_index)
                    .allocation_size(GLOBAL_MEMORY_SIZE).push_next(&mut vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS)),
                None,
            )
        }
        .unwrap();

        // Map its memory
        let global_ptr = unsafe {
            std::ptr::NonNull::new_unchecked(
                device
                    .map_memory(
                        global_memory,
                        0,
                        vk::WHOLE_SIZE,
                        vk::MemoryMapFlags::empty(),
                    )
                    .unwrap() as *mut u8,
            )
        };

        IntegratedAllocator {
            staging_buffer: Vec::with_capacity(STAGING_MEMORY_SIZE as usize),
            global_memory,
            global_ptr,
        }
    }
}

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
    ) {
        let bytes = bytemuck::cast_slice(data);
        let staging_buffer_offset = self.backend.stage_transfer(bytes);
        allocation.len += data.len();

        // Step three: stage the transfer to device local memory
        self.pending_transfers.push(PendingTransfer {
            destination: allocation.handle,
            staging_buffer_offset,
            transfer_size: bytes.len() as _,
            global_offset: allocation.global_offset,
        });
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

pub struct PendingTransfer {
    destination: vk::Buffer,
    staging_buffer_offset: usize,
    transfer_size: vk::DeviceSize,
    global_offset: offset_allocator::Allocation, // offset into the global memory
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
