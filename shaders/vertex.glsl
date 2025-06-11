#version 330 core

// 这里的 location 必须与 app.py 中的 glEnableVertexAttribArray(location) 严格对应
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec2 a_texCoord;
layout (location = 2) in vec3 a_normal;

out vec2 v_texCoord;
out vec3 v_normal;
out vec3 v_fragPos; // 片元在世界空间的位置

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // 将顶点位置变换到世界坐标，并传递给片元着色器
    v_fragPos = vec3(model * vec4(a_position, 1.0));

    // 将法线变换到世界坐标（使用逆转置矩阵以正确处理非统一缩放）
    v_normal = mat3(transpose(inverse(model))) * a_normal;

    // 直接传递纹理坐标
    v_texCoord = a_texCoord;

    // 计算顶点的最终裁剪空间位置
    gl_Position = projection * view * vec4(v_fragPos, 1.0);
}