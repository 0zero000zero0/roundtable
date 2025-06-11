#version 330 core
// shaders/depth_vertex.glsl
// 接收顶点位置
layout (location = 0) in vec3 a_position;

// 模型矩阵
uniform mat4 model;

// 将世界坐标位置传递给几何着色器
void main()
{
    gl_Position = model * vec4(a_position, 1.0);
}
