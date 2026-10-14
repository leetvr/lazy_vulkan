use std::ptr::NonNull;

use ash::vk;

use crate::{allocator::AllocationError, Context};

pub struct StagingBuffer {
    pub handle: vk::Buffer,
    #[allow(unused)]
    pub memory: vk::DeviceMemory,
    pub ptr: NonNull<u8>,
    capacity: vk::DeviceSize,
    size: vk::DeviceSize,
}

impl StagingBuffer {
    pub fn new(
        context: &Context,
        capacity: vk::DeviceSize,
    ) -> Result<StagingBuffer, AllocationError> {
        if capacity == 0 || capacity > context.max_buffer_size {
            return Err(AllocationError::InvalidStagingConfiguration);
        }
        let device = &context.device;
        let handle = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(capacity)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
        }?;
        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        if requirements.size > context.max_memory_allocation_size {
            unsafe { device.destroy_buffer(handle, None) };
            return Err(AllocationError::InvalidStagingConfiguration);
        }
        let memory_type_index = match context.find_memory_type_index(
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
            let ptr = NonNull::new(unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }?)
            .unwrap()
            .cast();
            context.set_debug_label(handle, "[lazy_vulkan] Staging Buffer");
            Ok(StagingBuffer {
                handle,
                memory,
                ptr,
                capacity,
                size: 0,
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

    pub fn check_capacity(&self, size: usize) -> Result<(), AllocationError> {
        if size == 0 {
            return Ok(());
        }
        let offset = (self.size + 15) & !15;
        if offset > self.capacity || size as u64 > self.capacity - offset {
            return Err(AllocationError::StagingExhausted);
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, context: &Context) {
        context.device.destroy_buffer(self.handle, None);
        context.device.unmap_memory(self.memory);
        context.device.free_memory(self.memory, None);
    }

    pub fn stage(&mut self, data: &[u8]) -> usize {
        if data.is_empty() {
            return self.size as usize;
        }
        self.check_capacity(data.len())
            .expect("staging buffer overflow");
        // Step one: copy the data into the staging buffer
        // Compressed image copies require texel-block-aligned offsets (up to 16 bytes).
        self.size = (self.size + 15) & !15;
        let staging_buffer_offset = self.size as usize;

        let transfer_size = data.len();

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

    pub fn retain_from(&mut self, start: usize) {
        let retained = self.size as usize - start;
        unsafe {
            std::ptr::copy(self.ptr.as_ptr().add(start), self.ptr.as_ptr(), retained);
        }
        self.size = retained as vk::DeviceSize;
    }

    pub fn clear(&mut self) {
        self.size = 0;
    }
}
