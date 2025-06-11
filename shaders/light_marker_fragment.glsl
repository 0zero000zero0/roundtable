#version 330 core

out vec4 FragColor;

// 接收来自Python代码的光源颜色
uniform vec3 lightColor;

void main()
{
    // 直接输出光源的颜色，使其看起来像一个发光体
    FragColor = vec4(lightColor, 1.0);
}
