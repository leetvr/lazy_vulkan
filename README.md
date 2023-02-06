# LazyVulkan
## Motivation
Sometimes you just want to render some things, you know? **You** know that Vulkan is powerful,
and feature rich, **I** know it too. That beautifully rich, explicit API, getting down and dirty
with the GPU - really getting in there, rolling up your sleeves and writing some super verbose,
close to the metal GPU stuff. You're looking at the spec and thinking "woah, that's a lot of
things I can do", some real gooey and caramelly insides there, real sweet and satisfying,
but sometimes you just want to say "hey, Vulkan, you know what? I've got to be real with you.
Just put some pixels on the screen for me? Think you can do that"? And then Vulkan is all like
"sure, that'll be one THOUSAND lines of code, thx" and you're all like "what!?! that's so many
lines" and then Vulkan just sorta shrugs at you, you know what I mean, and then you're like
"look buddy, I don't have a lot of time here" and you just like tap your watch as if you're
sort of showing Vulkan that like, you know, you don't really have a lot of time here at all
and so then Vulkan is sorta like "ugh fine" and then like rolls its eyes so I guess what I'm
trying to say is this crate lets you do some things I guess.

## Examples
### An Triangles
Let's say we want to get a triangle on the screen. Who really knows why - we just want one.

```rust
pub fn main() {
  use lazy_vulkan::{LazyVulkan, Vertex};
  // Compile your own damn shaders! LazyVulkan is just as lazy as you are!
  static FRAGMENT_SHADER: &'static [u8]  = include_bytes!("shaders/triangle.frag");
  static VERTEX_SHADER: &'static [u8]  = include_bytes!("shaders/triangle.vert");

  // Oh, you thought you could supply your own Vertex type? What is this, a rendergraph?!
  // Better make sure those shaders use the right layout!
  // **LAUGHS IN VULKAN**
  let vertices = [
      Vertex::new([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
      Vertex::new([1.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0]),
      Vertex::new([-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0]),
  ];

  // Your own index type?! What are you going to use, `u16`?
  let indices = [0, 1, 2];

  // Alright, let's get on with it.
  // 
  // You want a different window size? Just resize it!
  let (mut lazy_vulkan, mut lazy_renderer, mut event_loop) = LazyVulkan::builder()
      .initial_vertices(&vertices)
      .initial_indices(&indices)
      .fragment_shader(FRAGMENT_SHADER)
      .vertex_shader(VERTEX_SHADER)
      .build();

  // *whistling*
  event_loop.run_return(|event, _, control_flow| {
    // Note that there's no need to ever close the window because triangles are forever.
    match event {
      Event::MainEventsCleared => {
          let framebuffer_index = lazy_vulkan.render_begin();
          lazy_renderer.render(
              &lazy_vulkan.context(),
              framebuffer_index,
              &[DrawCall::new(0, 3, NO_TEXTURE_ID, Workflow::Main)],
          );

          // Don't forget your semaphores!
          lazy_vulkan
              .render_end(framebuffer_index, &[lazy_vulkan.present_complete_semaphore]);
      }
      _ => {},
    }
  }

  // I guess we better do this or else the Dreaded Validation Layers will complain
  unsafe {
      lazy_renderer.cleanup(&lazy_vulkan.context().device);
  }
}
```

The full source for this miraculous creation is available [here](examples/triangle.rs).

### Remote rendering
A logical next step to getting a triangle on the screen is rendering a triangle rendered
in another process. You can find an extremely safe, robust and clean implementation of this
extremely trivial process in the following places:

- [Remote Host](examples/remote_host.rs)
- [Remote Client](examples/remote_client.rs)

Please note that these will only run on the Microsoft Windows(TM) operating system. 

## NOTE
This repository is a joke and its contents should never, ever be used as the basis for a AAA rendering engine.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.