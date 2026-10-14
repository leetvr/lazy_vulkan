use std::{ptr::NonNull, sync::Arc};

use ash::vk;

use super::{AllocationError, Offset};
use crate::Context;

pub(super) struct MemoryBlocks {
    context: Arc<Context>,
    pub(super) blocks: Vec<MemoryBlock>,
}

pub(super) struct MemoryBlock {
    pub(super) memory: vk::DeviceMemory,
    pub(super) memory_type_index: u32,
    pub(super) offsets: offset_allocator::Allocator,
    unit_size: u64,
    mapped: Option<NonNull<u8>>,
    slab_buffer: vk::Buffer,
    slab_address: vk::DeviceAddress,
}

impl MemoryBlocks {
    pub(super) fn new(context: Arc<Context>, pool_bytes: u64) -> Result<Self, AllocationError> {
        let mut pool = Self {
            context,
            blocks: Vec::new(),
        };
        let block_limit = pool
            .context
            .max_memory_allocation_size
            .min(pool.context.max_buffer_size);
        if block_limit == 0 {
            return Err(AllocationError::InvalidPoolConfiguration);
        }
        let mut remaining = pool_bytes;
        while remaining != 0 {
            let size = remaining.min(block_limit);
            match pool.create_block(size) {
                Ok(block) => pool.blocks.push(block),
                Err(error) => {
                    unsafe { pool.destroy() };
                    return Err(error);
                }
            }
            remaining -= size;
        }
        Ok(pool)
    }

    fn create_block(&self, size: u64) -> Result<MemoryBlock, AllocationError> {
        let context = &self.context;
        let device = &context.device;
        let mut buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_DST,
                ),
                None,
            )
        }?;
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        unsafe {
            device.get_buffer_memory_requirements2(
                &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
                &mut requirements,
            );
        }
        let mut requirements = requirements.memory_requirements;
        if requirements.size > size {
            unsafe { device.destroy_buffer(buffer, None) };
            let buffer_size = size & !(requirements.alignment - 1);
            if buffer_size == 0 {
                return Err(AllocationError::InvalidPoolConfiguration);
            }
            buffer = unsafe {
                device.create_buffer(
                    &vk::BufferCreateInfo::default().size(buffer_size).usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::TRANSFER_DST,
                    ),
                    None,
                )
            }?;
            let mut queried = vk::MemoryRequirements2::default().push_next(&mut dedicated);
            unsafe {
                device.get_buffer_memory_requirements2(
                    &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
                    &mut queried,
                );
            }
            requirements = queried.memory_requirements;
        }
        let result = (|| {
            if dedicated.requires_dedicated_allocation == vk::TRUE {
                return Err(AllocationError::DedicatedAllocationRequired);
            }
            if requirements.size > size {
                return Err(AllocationError::InvalidPoolConfiguration);
            }
            let required_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let preferred_flags = if context.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                required_flags
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
            } else {
                required_flags
            };
            let memory_type_index = context
                .find_memory_type_index(&requirements, preferred_flags)
                .or_else(|| context.find_memory_type_index(&requirements, required_flags))
                .ok_or(AllocationError::NoCompatibleMemoryType)?;
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = unsafe {
                device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(size)
                        .memory_type_index(memory_type_index)
                        .push_next(&mut flags),
                    None,
                )
            }?;
            let result = (|| {
                unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
                let memory_flags = context.memory_properties.memory_types
                    [memory_type_index as usize]
                    .property_flags;
                let mapped = if context.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                    && memory_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    ) {
                    Some(
                        NonNull::new(unsafe {
                            device.map_memory(
                                memory,
                                0,
                                vk::WHOLE_SIZE,
                                vk::MemoryMapFlags::empty(),
                            )
                        }?)
                        .unwrap()
                        .cast(),
                    )
                } else {
                    None
                };
                let slab_address = unsafe {
                    device.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::default().buffer(buffer),
                    )
                };
                // Counts aligned units so the existing 32-bit range allocator can manage >4 GiB.
                let unit_size = size
                    .div_ceil(1 << 31)
                    .next_power_of_two()
                    .max(requirements.alignment)
                    .max(context.device_properties.limits.buffer_image_granularity);
                Ok(MemoryBlock {
                    memory,
                    memory_type_index,
                    offsets: offset_allocator::Allocator::new((size / unit_size) as u32),
                    unit_size,
                    mapped,
                    slab_buffer: buffer,
                    slab_address,
                })
            })();
            if result.is_err() {
                unsafe { device.free_memory(memory, None) };
            }
            result
        })();
        if result.is_err() {
            unsafe { device.destroy_buffer(buffer, None) };
        } else {
            context.set_debug_label(buffer, "[lazy_vulkan] Pool Slab");
        }
        result
    }

    pub(super) fn allocate(
        &mut self,
        requirements: vk::MemoryRequirements,
        alignment: u64,
        requires_dedicated: bool,
    ) -> Result<Offset, AllocationError> {
        if requires_dedicated {
            return Err(AllocationError::DedicatedAllocationRequired);
        }
        let alignment = alignment.max(requirements.alignment);
        let mut compatible = self.blocks.is_empty();
        for (index, block) in self.blocks.iter_mut().enumerate() {
            if requirements.memory_type_bits & (1 << block.memory_type_index) == 0 {
                continue;
            }
            compatible = true;
            if let Some(offset) = block.allocate(index, requirements.size, alignment) {
                return Ok(offset);
            }
        }
        Err(if compatible {
            AllocationError::PoolExhausted
        } else {
            AllocationError::NoCompatibleMemoryType
        })
    }

    pub(super) fn allocate_slab(
        &mut self,
        size: u64,
        alignment: u64,
    ) -> Result<Offset, AllocationError> {
        self.allocate(
            vk::MemoryRequirements {
                size,
                alignment,
                memory_type_bits: u32::MAX,
            },
            alignment,
            false,
        )
    }

    pub(super) fn memory(&self, offset: Offset) -> vk::DeviceMemory {
        self.blocks[offset.block].memory
    }

    pub(super) fn mapped_ptr(&self, offset: Offset) -> Option<NonNull<u8>> {
        self.blocks[offset.block].mapped.map(|ptr| unsafe {
            NonNull::new_unchecked(ptr.as_ptr().add(offset.total_offset() as usize))
        })
    }

    pub(super) fn slab_buffer(&self, offset: Offset) -> vk::Buffer {
        self.blocks[offset.block].slab_buffer
    }

    pub(super) fn slab_address(&self, offset: Offset) -> vk::DeviceAddress {
        self.blocks[offset.block].slab_address + offset.total_offset()
    }

    // Resource destruction and GPU completion are the caller's responsibility.
    pub(super) fn free(&mut self, offset: Offset) {
        self.blocks[offset.block].offsets.free(offset.allocation);
    }

    pub(super) unsafe fn destroy(&mut self) {
        for block in self.blocks.drain(..) {
            self.context.device.destroy_buffer(block.slab_buffer, None);
            if block.mapped.is_some() {
                self.context.device.unmap_memory(block.memory);
            }
            self.context.device.free_memory(block.memory, None);
        }
    }
}

impl MemoryBlock {
    fn allocate(&mut self, block: usize, size: u64, alignment: u64) -> Option<Offset> {
        let units = u32::try_from(size.div_ceil(self.unit_size)).ok()?;
        if units > (1 << 31) {
            return None;
        }
        let allocation = self.offsets.allocate(units)?;
        let byte_offset = u64::from(allocation.offset) * self.unit_size;
        let aligned = align_up(byte_offset, alignment);
        if aligned == byte_offset {
            return Some(Offset {
                block,
                allocation,
                unit_size: self.unit_size,
                bind_offset: 0,
            });
        }
        self.offsets.free(allocation);
        let padded = u64::from(units) + alignment.div_ceil(self.unit_size) - 1;
        if padded > (1 << 31) {
            return None;
        }
        // Use a complete size bin so the same aligned request can reuse this freed range.
        let padded = offset_allocator::ext::min_allocator_size(padded as u32);
        let allocation = self.offsets.allocate(padded)?;
        let byte_offset = u64::from(allocation.offset) * self.unit_size;
        Some(Offset {
            block,
            allocation,
            unit_size: self.unit_size,
            bind_offset: align_up(byte_offset, alignment) - byte_offset,
        })
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_units_address_ranges_above_four_gib() {
        let mut block = MemoryBlock {
            memory: vk::DeviceMemory::null(),
            memory_type_index: 0,
            offsets: offset_allocator::Allocator::new(6 << 20),
            unit_size: 1024,
            mapped: None,
            slab_buffer: vk::Buffer::null(),
            slab_address: 0,
        };
        let first = block.allocate(0, 5 << 30, 1024).unwrap();
        let second = block.allocate(0, 1 << 30, 1024).unwrap();
        assert_eq!(second.total_offset(), 5 << 30);
        assert!(block.allocate(0, 1024, 1024).is_none());
        block.offsets.free(first.allocation);
        let reused = block.allocate(0, 5 << 30, 1024).unwrap();
        assert_eq!(reused.total_offset(), first.total_offset());
        assert_eq!(second.total_offset(), 5 << 30);
    }
}
