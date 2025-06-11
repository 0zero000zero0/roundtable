// shaders/depth_fragment.glsl (已修正)
#version 330 core

in vec4 worldPos;

uniform vec3 lightPos;
uniform float far_plane;

void main()
{
    // 1. 计算片元到光源的距离 (现在 worldPos 有正确的值了)
    float lightDistance = length(worldPos.xyz - lightPos);

    // 2. 归一化距离并写入深度 (现在 far_plane 也有正确的值了)
    gl_FragDepth = lightDistance / far_plane;
}