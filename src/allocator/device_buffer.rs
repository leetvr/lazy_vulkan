use super::{
    memory_blocks::MemoryBlocks, staging_buffer::StagingBuffer, PendingTransfer,
    TransferDestination,
};
use crate::{Context, FULL_IMAGE};
use ash::vk;

pub(super) fn execute_transfers(
    context: &Context,
    memory_blocks: &MemoryBlocks,
    pending_transfers: Vec<PendingTransfer>,
    staging_buffer: &mut StagingBuffer,
    command_buffer: vk::CommandBuffer,
) {
    for pending in pending_transfers {
        match pending.destination {
            TransferDestination::Slab | TransferDestination::Buffer(_) => {
                buffer_transfer(
                    context,
                    memory_blocks,
                    pending,
                    staging_buffer,
                    command_buffer,
                );
            }
            TransferDestination::Image {
                image,
                extent,
                mip_levels,
                ref mip_offsets,
            } => {
                let mip_offsets = mip_offsets.clone();
                image_transfer(
                    context,
                    staging_buffer,
                    command_buffer,
                    pending,
                    image,
                    extent,
                    mip_levels,
                    mip_offsets.as_deref(),
                );
            }
        }
    }
}

fn image_transfer(
    context: &Context,
    staging_buffer: &mut StagingBuffer,
    command_buffer: vk::CommandBuffer,
    pending: PendingTransfer,
    image: vk::Image,
    extent: vk::Extent2D,
    mip_levels: u32,
    mip_offsets: Option<&[vk::DeviceSize]>,
) {
    let device = &context.device;

    unsafe {
        // Transition the image into the TRANSFER DST layout
        context.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(&[
                vk::ImageMemoryBarrier2::default()
                    .subresource_range(FULL_IMAGE)
                    .image(image)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
            ]),
        );
        let regions = if let Some(offsets) = mip_offsets {
            offsets
                .iter()
                .enumerate()
                .map(|(level, offset)| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(pending.staging_buffer_offset as u64 + offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(level as u32)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: (extent.width >> level).max(1),
                            height: (extent.height >> level).max(1),
                            depth: 1,
                        })
                })
                .collect::<Vec<_>>()
        } else {
            vec![vk::BufferImageCopy::default()
                .buffer_offset(pending.staging_buffer_offset as u64)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(extent.into())]
        };
        device.cmd_copy_buffer_to_image(
            command_buffer,
            staging_buffer.handle,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
        if mip_offsets.is_some() {
            transition_mip_to_shader_read(
                context,
                command_buffer,
                image,
                FULL_IMAGE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags2::TRANSFER_WRITE,
            );
        } else {
            generate_mipmaps(context, command_buffer, image, extent, mip_levels);
        }
    };

    pending.transfer_token.mark_completed();
}

fn generate_mipmaps(
    context: &Context,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    extent: vk::Extent2D,
    mip_levels: u32,
) {
    let device = &context.device;
    let mut source_width = extent.width as i32;
    let mut source_height = extent.height as i32;

    for destination_level in 1..mip_levels {
        let source_level = destination_level - 1;
        let destination_width = (source_width / 2).max(1);
        let destination_height = (source_height / 2).max(1);
        let source_range = mip_range(source_level);

        unsafe {
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .subresource_range(source_range)
                        .image(image)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
                ]),
            );
            device.cmd_blit_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(source_level)
                            .layer_count(1),
                    )
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: source_width,
                            y: source_height,
                            z: 1,
                        },
                    ])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(destination_level)
                            .layer_count(1),
                    )
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: destination_width,
                            y: destination_height,
                            z: 1,
                        },
                    ])],
                vk::Filter::LINEAR,
            );
            transition_mip_to_shader_read(
                context,
                command_buffer,
                image,
                source_range,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags2::TRANSFER_READ,
            );
        }

        source_width = destination_width;
        source_height = destination_height;
    }

    unsafe {
        transition_mip_to_shader_read(
            context,
            command_buffer,
            image,
            mip_range(mip_levels - 1),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
    }
}

unsafe fn transition_mip_to_shader_read(
    context: &Context,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    old_layout: vk::ImageLayout,
    source_access: vk::AccessFlags2,
) {
    context.cmd_pipeline_barrier2(
        command_buffer,
        &vk::DependencyInfo::default().image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
            .subresource_range(range)
            .image(image)
            .src_access_mask(source_access)
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]),
    );
}

fn mip_range(mip_level: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(mip_level)
        .level_count(1)
        .layer_count(1)
}

fn buffer_transfer(
    context: &Context,
    memory_blocks: &MemoryBlocks,
    PendingTransfer {
        destination,
        staging_buffer_offset,
        transfer_size,
        allocation_offset,
        global_offset,
        ..
    }: PendingTransfer,
    staging_buffer: &mut StagingBuffer,
    command_buffer: vk::CommandBuffer,
) {
    if context.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
        if let Some(destination) = memory_blocks.mapped_ptr(global_offset) {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    staging_buffer.ptr.as_ptr().add(staging_buffer_offset),
                    destination.as_ptr().add(allocation_offset),
                    transfer_size as usize,
                );
            }
            return;
        }
    }
    context.begin_marker("Buffer Transfer", glam::vec4(0., 1., 1., 1.));
    let device = &context.device;

    let (destination_offset, destination_buffer) = match destination {
        TransferDestination::Buffer(buffer) => (allocation_offset, buffer),
        TransferDestination::Slab => (
            global_offset.total_offset() as usize,
            memory_blocks.slab_buffer(global_offset),
        ),
        _ => return,
    };

    log::trace!(
        "TRANSFER: {transfer_size} [src: {staging_buffer_offset}] -> [dst: {destination_offset}]"
    );

    // Issue the transfer
    unsafe {
        device.cmd_copy_buffer(
            command_buffer,
            staging_buffer.handle,
            destination_buffer,
            &[vk::BufferCopy::default()
                .src_offset(staging_buffer_offset as vk::DeviceSize)
                .dst_offset(destination_offset as vk::DeviceSize)
                .size(transfer_size)],
        );
    }

    // Place a barrier
    // TODO(these are a bit pessimistic, they could potentially be inferred from usage flags)
    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().buffer_memory_barriers(&[
                vk::BufferMemoryBarrier2::default()
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(
                        vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COPY,
                    )
                    .dst_access_mask(
                        vk::AccessFlags2::INDEX_READ
                            | vk::AccessFlags2::SHADER_READ
                            | vk::AccessFlags2::TRANSFER_WRITE,
                    )
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::COPY
                            | vk::PipelineStageFlags2::VERTEX_SHADER
                            | vk::PipelineStageFlags2::INDEX_INPUT,
                    )
                    .buffer(destination_buffer)
                    .offset(destination_offset as _)
                    .size(transfer_size),
            ]),
        )
    };

    context.end_marker();
}
