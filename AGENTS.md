## Working on Lazy Vulkan
Lazy Vulkan is exactly what it sounds like. It does not hide Vulkan from the user, and VERY IMPORTANTLY
it does not try to extend Rust's ownership model onto the GPU. It makes no attempt to perform magic,
it does not ensure validation safety, nothing. It is just a wrapper around common Vulkan functionality
and that's it.

Don't write unit tests just because Uncle Bob told you to. Sure, test a complicated bit of logic
or make sure that some complex GPU operation succeeded, but you don't need to test that Vulkan
itself works - smart people are already on that one.

Inspiration is drawn from the "No Graphics API" concept.

## IMPORTANT
DO NOT modify or create any documentation without human permission.

## GPU usage
- Never test with llvmpipe. Use a hardware GPU for GPU tests.
- Any GPU usage related to the current task is permitted without asking for additional approval.
