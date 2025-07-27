#version 450

layout(location = 0) in vec3 vColor; // from the VS
layout(location = 0) out vec4 fColor; // final output

void main() {
    fColor = vec4(vColor, 1.0);
}
