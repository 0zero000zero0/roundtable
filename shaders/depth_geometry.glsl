// shaders/depth_geometry.glsl (已修正)
#version 330 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 18) out;

uniform mat4 shadowMatrices[6];

out vec4 worldPos;

void main()
{
    for(int face = 0; face < 6; ++face)
    {
        gl_Layer = face;

        for(int i = 0; i < 3; ++i)
        {
            // 1. 从顶点着色器获取世界坐标，并将其传递给片元着色器
            worldPos = gl_in[i].gl_Position;

            // 2. 将世界坐标变换到光源的裁切空间
            gl_Position = shadowMatrices[face] * worldPos;
            EmitVertex();
        }
        EndPrimitive();
    }
}