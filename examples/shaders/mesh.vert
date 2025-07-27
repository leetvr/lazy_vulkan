#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable

struct Vertex {
    vec4 position;
};

layout(scalar, buffer_reference, buffer_reference_align = 16) readonly buffer VertexBuffer
{
    Vertex vertices[];
};

layout(scalar, push_constant) uniform Registers {
    mat4 mvp;
    vec4 colour;
    VertexBuffer vertexBuffer;
} registers;
layout(location = 0) out vec3 vColor;

void main() {
    // Pick the position based on the built-in vertex index
    gl_Position = registers.mvp * registers.vertexBuffer.vertices[gl_VertexIndex].position;

    vColor = registers.colour.xyz;
}
