#version 330 core

// 深度渲染只需要位置属性，它必须在 location = 0
layout (location = 0) in vec3 a_position;

uniform mat4 model;

void main()
{
    // 将顶点变换到世界坐标，并传递给几何着色器
    gl_Position = model * vec4(a_position, 1.0);
}