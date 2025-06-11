#version 330 core
// shaders/depth_geometry.glsl
// 启用这个扩展来使用 gl_Layer
#extension GL_ARB_geometry_shader4 : enable

// 输入是一个三角形
layout (triangles) in;
// 输出是一系列的三角形条带，最多18个顶点
layout (triangle_strip, max_vertices = 18) out;

// 接收从光源视角看的6个方向的 视图*投影 矩阵
uniform mat4 shadowMatrices[6];

// 将世界坐标位置输出到片元着色器
out vec4 FragPos;

void main()
{
    // 遍历立方体贴图的6个面
    for(int face = 0; face < 6; ++face)
    {
        // gl_Layer 指定了图元将被发送到哪个面
        gl_Layer = face;

        // 遍历输入三角形的3个顶点
        for(int i = 0; i < 3; ++i)
        {
            // 从顶点着色器获取顶点的世界坐标
            FragPos = gl_in[i].gl_Position;
            // 使用对应面的 shadowMatrix 对顶点进行变换
            gl_Position = shadowMatrices[face] * FragPos;
            // 发射这个顶点
            EmitVertex();
        }
        // 结束当前面的三角形条带
        EndPrimitive();
    }
}
