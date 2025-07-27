#version 450

layout(push_constant) uniform Registers {
    vec4 colour;
} registers;
layout(location = 0) out vec3 vColor;

vec2 positions[3] = vec2[](
        vec2(0.0, -0.5), // top
        vec2(-0.5, 0.5), // bottom-left
        vec2(0.5, 0.5) // bottom-right
    );

void main() {
    // Pick the position based on the built-in vertex index
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);

    vColor = registers.colour.xyz;
}
