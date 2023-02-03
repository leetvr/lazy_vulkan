use lazy_vulkan::{DrawCall, LazyVulkan, Vertex, NO_TEXTURE_ID};
/// Compile your own damn shaders! LazyVulkan is just as lazy as you are!
static FRAGMENT_SHADER: &'static [u8] = include_bytes!("shaders/triangle.frag.spv");
static VERTEX_SHADER: &'static [u8] = include_bytes!("shaders/triangle.vert.spv");

pub fn main() {
    env_logger::init();

    /// Oh, you thought you could supply your own Vertex type? What is this, a rendergraph?!
    /// Better make sure those shaders use the right layout!
    /// **LAUGHS IN VULKAN**
    let vertices = [
        Vertex::new([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([1.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0]),
    ];

    /// Your own index type?! What are you going to use, `u16`?
    let indices = [0, 1, 2];

    let mut lazy_vulkan = LazyVulkan::builder()
        .initial_vertices(&vertices)
        .initial_indices(&indices)
        .fragment_shader(FRAGMENT_SHADER)
        .vertex_shader(VERTEX_SHADER)
        .build();

    lazy_vulkan.run(|| {
        lazy_vulkan.draw(&[DrawCall::new(
            0,
            3,
            NO_TEXTURE_ID,
            lazy_vulkan::Workflow::Main,
        )]);
    });
}
