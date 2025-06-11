#version 330 core

// 顶点数据输入
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec2 a_texCoord;
layout (location = 2) in vec3 a_normal;

// MVP 矩阵
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// 输出到片元着色器
out vec2 v_texCoord;
out vec3 v_normal;
out vec3 v_fragPos;

void main()
{
    // 将顶点位置变换到世界坐标
    v_fragPos = vec3(model * vec4(a_position, 1.0));
    // 将法线变换到世界坐标 (使用法线矩阵避免非等比缩放导致的问题)
    v_normal = mat3(transpose(inverse(model))) * a_normal;
    // 传递纹理坐标
    v_texCoord = a_texCoord;

    // 计算最终的裁剪空间坐标
    gl_Position = projection * view * vec4(v_fragPos, 1.0);
}