use crate::allocator::GLOBAL_MEMORY_SIZE;
use crate::FULL_IMAGE;
use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;

use crate::Context;

use super::staging_buffer::StagingBuffer;
use super::PendingTransfer;
use super::TransferDestination;

pub enum DeviceBuffer {
    Discrete(DiscreteDeviceBuffer),
    Integrated(IntegratedDeviceBuffer),
}

impl DeviceBuffer {
    pub fn new(context: Arc<Context>) -> DeviceBuffer {
        match context.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => {
                DeviceBuffer::Discrete(DiscreteDeviceBuffer::new(context))
            }
            vk::PhysicalDeviceType::INTEGRATED_GPU => {
                DeviceBuffer::Integrated(IntegratedDeviceBuffer::new(context))
            }
            _ => unreachable!("Impossible device type"),
        }
    }

    pub fn device_memory(&self) -> vk::DeviceMemory {
        match self {
            DeviceBuffer::Discrete(discrete_allocator) => discrete_allocator.device_memory,
            DeviceBuffer::Integrated(integrated_allocator) => integrated_allocator.global_memory,
        }
    }

    pub fn get_device_address(&self, offset: offset_allocator::Allocation) -> vk::DeviceAddress {
        let base_address = match self {
            DeviceBuffer::Discrete(discrete_allocator) => discrete_allocator.slab_address,
            DeviceBuffer::Integrated(integrated_allocator) => integrated_allocator.slab_address,
        };

        base_address + offset.offset as vk::DeviceAddress
    }

    pub fn execute_transfers(
        &mut self,
        context: &Context,
        mut pending_transfers: Vec<PendingTransfer>,
        staging_buffer: &mut StagingBuffer,
    ) {
        for pending in pending_transfers.drain(..) {
            match pending.destination {
                TransferDestination::Slab | TransferDestination::Buffer(_) => {
                    match self {
                        DeviceBuffer::Discrete(discrete_allocator) => {
                            discrete_allocator.buffer_transfer(context, pending, staging_buffer)
                        }
                        DeviceBuffer::Integrated(integrated_allocator) => {
                            integrated_allocator.buffer_transfer(pending, staging_buffer)
                        }
                    };
                }
                TransferDestination::Image(image, extent) => {
                    let device = &context.device;
                    let command_buffer = context.draw_command_buffer;

                    unsafe {
                        context.cmd_pipeline_barrier2(
                            command_buffer,
                            &vk::DependencyInfo::default().image_memory_barriers(&[
                                // Swapchain image
                                vk::ImageMemoryBarrier2::default()
                                    .subresource_range(FULL_IMAGE)
                                    .image(image)
                                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                    .old_layout(vk::ImageLayout::UNDEFINED)
                                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                            ]),
                        );
                        device.cmd_copy_buffer_to_image(
                            command_buffer,
                            staging_buffer.handle,
                            image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[vk::BufferImageCopy::default()
                                .buffer_offset(pending.staging_buffer_offset as _)
                                .image_subresource(
                                    vk::ImageSubresourceLayers::default()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .layer_count(1),
                                )
                                .image_extent(extent.into())],
                        );

                        context.cmd_pipeline_barrier2(
                            command_buffer,
                            &vk::DependencyInfo::default().image_memory_barriers(&[
                                // Swapchain image
                                vk::ImageMemoryBarrier2::default()
                                    .subresource_range(FULL_IMAGE)
                                    .image(image)
                                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                            ]),
                        );
                    };

                    pending.transfer_token.mark_completed();
                }
            }
        }

        staging_buffer.clear();
    }
}

pub struct DiscreteDeviceBuffer {
    device_memory: vk::DeviceMemory,
    slab_buffer: vk::Buffer,
    slab_address: vk::DeviceAddress,
}

impl DiscreteDeviceBuffer {
    pub fn new(context: Arc<Context>) -> DiscreteDeviceBuffer {
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
            log::debug!("Allocating {GLOBAL_MEMORY_SIZE} from memory type / heap : {memory_type_index}, {memory_heap_index}");
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .memory_type_index(memory_type_index)
                    .allocation_size(GLOBAL_MEMORY_SIZE).push_next(&mut vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS)),
                None,
            )
        }
        .unwrap();

        let (slab_buffer, slab_address) = create_slab_buffer(device);

        Self {
            device_memory,
            slab_buffer,
            slab_address,
        }
    }

    pub fn buffer_transfer(
        &mut self,
        context: &Context,
        PendingTransfer {
            destination,
            staging_buffer_offset,
            allocation_offset,
            transfer_size,
            transfer_token,
            ..
        }: PendingTransfer,
        staging_buffer: &mut StagingBuffer,
    ) {
        let device = &context.device;
        let command_buffer = context.draw_command_buffer;

        let buffer = match destination {
            TransferDestination::Buffer(buffer) => buffer,
            _ => return,
        };

        // Issue the transfer
        unsafe {
            device.cmd_copy_buffer(
                command_buffer,
                staging_buffer.handle,
                buffer,
                &[vk::BufferCopy::default()
                    .src_offset(staging_buffer_offset as _)
                    .dst_offset(allocation_offset as _)
                    .size(transfer_size)],
            );
        }

        // Place a barrier
        unsafe {
            device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
                        .buffer(buffer)
                        .size(transfer_size),
                ]),
            )
        };

        transfer_token.mark_completed();
    }
}

fn create_slab_buffer(device: &ash::Device) -> (vk::Buffer, u64) {
    // Create the buffer
    let slab_buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(GLOBAL_MEMORY_SIZE)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_DST,
                ),
            None,
        )
    }
    .unwrap();

    let slab_address = unsafe {
        device
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(slab_buffer))
    };
    (slab_buffer, slab_address)
}

pub struct IntegratedDeviceBuffer {
    global_memory: vk::DeviceMemory,
    global_ptr: NonNull<u8>,
    slab_buffer: vk::Buffer,
    slab_address: vk::DeviceAddress,
}

impl IntegratedDeviceBuffer {
    pub fn new(context: Arc<Context>) -> IntegratedDeviceBuffer {
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
            log::debug!("Allocating {GLOBAL_MEMORY_SIZE} from memory type / heap : {memory_type_index}, {memory_heap_index}");
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

        let (slab_buffer, slab_address) = create_slab_buffer(device);

        IntegratedDeviceBuffer {
            global_memory,
            global_ptr,
            slab_buffer,
            slab_address,
        }
    }

    pub fn buffer_transfer(
        &mut self,
        PendingTransfer {
            allocation_offset,
            staging_buffer_offset,
            transfer_size,
            global_offset,
            transfer_token,
            ..
        }: PendingTransfer,
        staging_buffer: &mut StagingBuffer,
    ) {
        // We get the source pointer by taking the base address of the **staging buffer** and
        // adding the offset
        let source = unsafe { staging_buffer.ptr.add(staging_buffer_offset).as_ptr() };

        // We get the destination pointer by taking the base address of the **global buffer**,
        // and then finally adding the offset within the allocation itself
        let destination = unsafe {
            self.global_ptr
                .add(global_offset.offset as usize + allocation_offset)
                .as_ptr()
        };

        unsafe {
            std::ptr::copy_nonoverlapping(source, destination, transfer_size as usize);
        };

        transfer_token.mark_completed();
    }
}
