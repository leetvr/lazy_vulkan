use std::ffi::CStr;

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

        let mut debug_messenger_create_info = debug_messenger_create_info();

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
                        .push_next(&mut layer_settings_create_info)
                        .push_next(&mut debug_messenger_create_info),
                    None,
                )
                .unwrap()
        };

        let debug_messenger = DebugMessenger::new(&entry, &instance).unwrap();

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
use ash::ext::debug_utils;
use log::{debug, error, info, trace, warn};
use std::borrow::Cow;
use std::os::raw::c_void;

/// Basic owned wrapper so cleanup is obvious.
pub struct DebugMessenger {
    loader: debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugMessenger {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self, vk::Result> {
        let loader = debug_utils::Instance::new(entry, instance);
        let create_info = debug_messenger_create_info();

        let messenger = unsafe { loader.create_debug_utils_messenger(&create_info, None)? };

        Ok(Self { loader, messenger })
    }

    pub fn raw(&self) -> vk::DebugUtilsMessengerEXT {
        self.messenger
    }

    pub fn destroy(&self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}

/// Call this when building your instance if you want debug messages during instance creation too.
pub fn debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO, // Add VERBOSE if you want the firehose.
                                                               // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback))
}

/// Optional convenience: chain this into InstanceCreateInfo via push_next(...)
pub fn instance_create_info_with_debug<'a>(
    app_info: &'a vk::ApplicationInfo<'a>,
    enabled_layers: &'a [*const i8],
    enabled_extensions: &'a [*const i8],
    debug_ci: &'a mut vk::DebugUtilsMessengerCreateInfoEXT<'a>,
    flags: vk::InstanceCreateFlags,
) -> vk::InstanceCreateInfo<'a> {
    vk::InstanceCreateInfo::default()
        .application_info(app_info)
        .enabled_layer_names(enabled_layers)
        .enabled_extension_names(enabled_extensions)
        .flags(flags)
        .push_next(debug_ci)
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { &*p_callback_data };

    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::Borrowed("<no-id>")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message_id_name) }.to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::Borrowed("<no-message>")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message) }.to_string_lossy()
    };

    let text = format!(
        "[Vulkan][{:?}][{:?}][{}:{}] {}",
        message_severity, message_type, message_id_name, message_id_number, message
    );

    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        error!("{text}");
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!("{text}");
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        info!("{text}");
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE) {
        debug!("{text}");
    } else {
        trace!("{text}");
    }

    vk::FALSE
}
