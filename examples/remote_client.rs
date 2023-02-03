use lazy_vulkan::Vertex;
use std::io::Write;
use std::net::TcpStream;

use log::info;

/// Compile your own damn shaders! LazyVulkan is just as lazy as you are!
static FRAGMENT_SHADER: &'static [u8] = include_bytes!("shaders/triangle.frag.spv");
static VERTEX_SHADER: &'static [u8] = include_bytes!("shaders/triangle.vert.spv");

pub fn main() -> std::io::Result<()> {
    let mut entry = ash::Entry::linked();
    env_logger::init();

    let vertices = [
        Vertex::new([1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([-1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0]),
    ];

    // Your own index type?! What are you going to use, `u16`?
    let indices = [0, 1, 2];

    // Alright, let's build some stuff

    info!("Conecting to server..");
    let mut stream = TcpStream::connect("127.0.0.1:8000")?;
    info!("Connected!");
    let value: u32 = 42;
    stream.write(&value.to_be_bytes())?;

    Ok(())
}
