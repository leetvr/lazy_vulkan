use std::ptr::NonNull;

use ash::vk;

use crate::{allocator::STAGING_MEMORY_SIZE, Context};

pub struct StagingBuffer {
    pub handle: vk::Buffer,
    #[allow(unused)]
    pub memory: vk::DeviceMemory,
    pub ptr: NonNull<u8>,
    size: vk::DeviceSize,
}

impl StagingBuffer {
    pub fn new(context: &Context) -> StagingBuffer {
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

        // Allocate our staging memory
        let memory = unsafe {
            log::debug!("[STAGING BUFFER] Allocating {STAGING_MEMORY_SIZE} from memory type / heap : {memory_type_index}, {memory_heap_index}");
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .memory_type_index(memory_type_index)
                    .allocation_size(STAGING_MEMORY_SIZE),
                None,
            )
        }
        .unwrap();

        // Create a staging buffer
        let handle = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(STAGING_MEMORY_SIZE)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
        }
        .unwrap();

        context.set_debug_label(handle, "[lazy_vulkan] Staging Buffer");

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

        StagingBuffer {
            handle,
            memory,
            ptr,
            size: 0,
        }
    }

    pub fn stage(&mut self, data: &[u8]) -> usize {
        // Step one: copy the data into the staging buffer
        let staging_buffer_offset = self.size as usize;

        let transfer_size = data.len();

        if (staging_buffer_offset + transfer_size) > STAGING_MEMORY_SIZE as usize {
            panic!("Staging buffer overflow. Transfer size: {transfer_size}, current staging buffer size: {}", self.size);
        }

        // We get the staging pointer by taking the base address and adding the current size of
        // the buffer.
        let staging_ptr = unsafe { self.ptr.add(staging_buffer_offset).as_ptr() };

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, staging_ptr, transfer_size);
        };

        // Step two: record the amount of data transferred
        self.size += transfer_size as vk::DeviceSize;

        staging_buffer_offset as usize
    }

    pub fn clear(&mut self) {
        self.size = 0;
    }
}
