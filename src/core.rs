use ash::vk::{self, LayerSettingTypeEXT};
use winit::raw_window_handle::HasDisplayHandle;

pub struct Core {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
}

impl Core {
    pub(crate) fn from_window(window: &winit::window::Window) -> Self {
        // #[cfg(any(target_os = "windows", target_vendor = "apple"))]
        let entry = ash::Entry::linked();

        // #[cfg(not(any(target_os = "windows", target_vendor = "apple")))]
        // let entry = unsafe { ash::Entry::load() }.unwrap();
        println!("Layers:");
        for layer in unsafe { entry.enumerate_instance_layer_properties().unwrap() } {
            let name = layer.layer_name_as_c_str().unwrap();
            println!("Layer name: {name:?}");
        }

        let display_handle = window.display_handle().unwrap().as_raw();

        let mut instance_extensions = ash_window::enumerate_required_extensions(display_handle)
            .unwrap()
            .to_vec();

        // TODO: Make this optional
        instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());

        let version;
        let instance_create_flags;

        #[cfg(target_vendor = "apple")]
        {
            instance_extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            instance_extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
            instance_extensions.push(ash::ext::layer_settings::NAME.as_ptr());
            version = vk::API_VERSION_1_3;
            instance_create_flags = vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            version = vk::API_VERSION_1_3;
            instance_create_flags = vk::InstanceCreateFlags::default();
        }

        let validation_layer = c"VK_LAYER_KHRONOS_validation";

        let layer_setting = vk::LayerSettingEXT::default()
            .layer_name(c"khronos_validation")
            .setting_name(c"validate_core")
            .ty(LayerSettingTypeEXT::BOOL32)
            .values(&[1]);

        let binding = [layer_setting];
        let mut layer_settings_create_info =
            vk::LayerSettingsCreateInfoEXT::default().settings(&binding);

        let instance = unsafe {
            entry
                .create_instance(
                    &vk::InstanceCreateInfo::default()
                        .flags(instance_create_flags)
                        .enabled_extension_names(&instance_extensions)
                        .enabled_layer_names(&[validation_layer.as_ptr()])
                        .application_info(&vk::ApplicationInfo::default().api_version(version))
                        .push_next(&mut layer_settings_create_info),
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

    pub fn headless() -> Self {
        let entry = unsafe { ash::Entry::load().unwrap() };

        let mut instance_extensions = Vec::new();

        instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
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
            instance_create_flags = vk::InstanceCreateFlags::default();
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
