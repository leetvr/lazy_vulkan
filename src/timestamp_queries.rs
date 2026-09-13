use std::sync::Arc;

use ash::vk;

use crate::Context;

#[derive(Debug, Clone)]
pub struct TimestampQueryResults {
    frame_index: u32,
    timestamp_period_nanoseconds: f64,
    timestamp_valid_bits: u32,
    timestamps: Vec<u64>,
}

impl TimestampQueryResults {
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    pub fn timestamp_period_nanoseconds(&self) -> f64 {
        self.timestamp_period_nanoseconds
    }

    pub fn query_count(&self) -> usize {
        self.timestamps.len()
    }

    pub fn duration_nanoseconds(&self, start_query: u32, end_query: u32) -> Option<f64> {
        let start = *self.timestamps.get(start_query as usize)?;
        let end = *self.timestamps.get(end_query as usize)?;
        Some(
            timestamp_delta_ticks(start, end, self.timestamp_valid_bits) as f64
                * self.timestamp_period_nanoseconds,
        )
    }
}

pub(crate) struct TimestampQueries {
    context: Arc<Context>,
    pool: vk::QueryPool,
    capacity: u32,
    submitted_frame: Option<u32>,
    submitted_count: u32,
}

impl TimestampQueries {
    pub(crate) fn new(context: Arc<Context>, capacity: u32) -> Self {
        assert!(capacity > 0);
        let pool = unsafe {
            context.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(capacity),
                None,
            )
        }
        .unwrap();
        context.set_debug_label(pool, "Frame timestamps");
        Self {
            context,
            pool,
            capacity,
            submitted_frame: None,
            submitted_count: 0,
        }
    }

    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    pub(crate) fn completed_results(&self) -> Option<TimestampQueryResults> {
        let frame_index = self.submitted_frame?;
        if self.submitted_count == 0 {
            return None;
        }
        let mut timestamps = vec![0_u64; self.submitted_count as usize];
        unsafe {
            self.context.device.get_query_pool_results(
                self.pool,
                0,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64,
            )
        }
        .unwrap();
        Some(TimestampQueryResults {
            frame_index,
            timestamp_period_nanoseconds: f64::from(
                self.context.device_properties.limits.timestamp_period,
            ),
            timestamp_valid_bits: self.context.timestamp_valid_bits,
            timestamps,
        })
    }

    pub(crate) fn begin_frame(&mut self, frame_index: u32, query_count: u32) {
        debug_assert!(query_count <= self.capacity);
        if query_count > 0 {
            unsafe {
                self.context.device.cmd_reset_query_pool(
                    self.context.draw_command_buffer,
                    self.pool,
                    0,
                    query_count,
                );
            }
        }
        self.submitted_frame = Some(frame_index);
        self.submitted_count = query_count;
    }

    pub(crate) fn write(&self, query: u32, stage: vk::PipelineStageFlags2) {
        debug_assert!(query < self.submitted_count);
        unsafe {
            self.context.device.cmd_write_timestamp2(
                self.context.draw_command_buffer,
                stage,
                self.pool,
                query,
            );
        }
    }
}

impl Drop for TimestampQueries {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_query_pool(self.pool, None);
        }
    }
}

fn timestamp_delta_ticks(start: u64, end: u64, valid_bits: u32) -> u64 {
    let mask = if valid_bits >= u64::BITS {
        u64::MAX
    } else {
        (1_u64 << valid_bits) - 1
    };
    end.wrapping_sub(start) & mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_delta_uses_all_64_bits() {
        assert_eq!(timestamp_delta_ticks(100, 175, 64), 75);
    }

    #[test]
    fn timestamp_delta_handles_counter_wrap() {
        assert_eq!(timestamp_delta_ticks(250, 5, 8), 11);
    }

    #[test]
    fn results_convert_ticks_with_the_device_timestamp_period() {
        let results = TimestampQueryResults {
            frame_index: 7,
            timestamp_period_nanoseconds: 2.5,
            timestamp_valid_bits: 64,
            timestamps: vec![100, 112],
        };

        assert_eq!(results.frame_index(), 7);
        assert_eq!(results.query_count(), 2);
        assert_eq!(results.timestamp_period_nanoseconds(), 2.5);
        assert_eq!(results.duration_nanoseconds(0, 1), Some(30.0));
        assert_eq!(results.duration_nanoseconds(1, 2), None);
    }
}
