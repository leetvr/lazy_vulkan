use ash::vk;
use winit::raw_window_handle::HasDisplayHandle;

pub struct Core {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
}
impl Core {
    pub(crate) fn new(window: &winit::window::Window) -> Self {
        let entry = unsafe { ash::Entry::load().unwrap() };

        let display_handle = window.display_handle().unwrap().as_raw();

        #[allow(unused_mut)]
        let mut instance_extensions = ash_window::enumerate_required_extensions(display_handle)
            .unwrap()
            .to_vec();

        let version;
        let instance_create_flags;

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            instance_extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            instance_extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
            version = vk::API_VERSION_1_2;
            instance_create_flags = vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            version = vk::API_VERSION_1_3;
            create_flags = vk::InstanceCreateFlags::default();
        }

        let instance = unsafe {
            entry
                .create_instance(
                    &vk::InstanceCreateInfo::default()
                        .flags(instance_create_flags)
                        .enabled_extension_names(&instance_extensions)
                        .application_info(&vk::ApplicationInfo::default().api_version(version)),
                    None,
                )
                .unwrap()
        };

        let physical_device = unsafe { instance.enumerate_physical_devices() }
            .unwrap()
            .first()
            .copied()
            .unwrap();

        Self {
            entry,
            instance,
            physical_device,
        }
    }
}
