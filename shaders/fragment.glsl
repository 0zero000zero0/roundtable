#version 330 core

in vec2 v_texCoord;
in vec3 v_normal;
in vec3 v_fragPos;

out vec4 FragColor;


uniform sampler2D u_texture;
uniform samplerCube u_depthMap;

uniform vec3 u_viewPos;
uniform vec3 u_ambient_light;
uniform float u_far_plane;
uniform vec3 u_lightPos;

uniform int u_visualize_normals;


struct PointLight {
    vec3 position;
    vec3 color;
    float constant;
    float linear;
    float quadratic;
};

#define NUM_LIGHTS 11
uniform PointLight u_lights[NUM_LIGHTS];


float CalculateShadow(vec3 fragPos, vec3 normal);
vec3 CalculatePointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec4 texColor, float shadow);

void main()
{
    // --- 调试模式A：可视化深度图 ---
    // vec3 fragToLight = v_fragPos - u_lightPos;
    // float depthValue = texture(u_depthMap, fragToLight).r;
    // FragColor = vec4(vec3(depthValue), 1.0);
    // return;

    // --- 正常光照计算 ---
    vec3 norm = normalize(v_normal);
    vec3 viewDir = normalize(u_viewPos - v_fragPos);
    vec4 texColor = texture(u_texture, v_texCoord);

    // --- 调试模式B：可视化法线 ---
    if (u_visualize_normals == 1) {
        FragColor = vec4(norm * 0.5 + 0.5, 1.0);
        return;
    }

    // 1. 计算主光源的阴影
    float shadow = CalculateShadow(v_fragPos, norm);

    // 2. 累加所有光源的光照贡献
    vec3 final_color = u_ambient_light * texColor.rgb;
    final_color += CalculatePointLight(u_lights[0], norm, v_fragPos, viewDir, texColor, shadow);

    for(int i = 1; i < NUM_LIGHTS; i++)
    {
        final_color += CalculatePointLight(u_lights[i], norm, v_fragPos, viewDir, texColor, 0.0);
    }

    // 3. 伽马校正
    final_color = pow(final_color, vec3(1.0/2.2));
    FragColor = vec4(final_color, 1.0);
}

// =======================================================
//                辅助函数
// =======================================================

float CalculateShadow(vec3 fragPos, vec3 normal)
{
    vec3 fragToLight = fragPos - u_lightPos; // 修正

    vec3 sample_offsets[] = vec3[](
       vec3(0.38, 0.14, 0.22), vec3(0.26, -0.35, -0.19), vec3(-0.45, 0.31, 0.09),
       vec3(0.12, 0.48, -0.34), vec3(-0.27, -0.16, 0.49), vec3(-0.08, 0.23, -0.42),
       vec3(0.41, -0.07, 0.37), vec3(-0.15, -0.43, -0.28), vec3(0.05, -0.29, 0.47),
       vec3(0.33, 0.46, 0.11), vec3(-0.49, 0.03, -0.36), vec3(-0.32, 0.21, 0.43),
       vec3(0.18, -0.41, 0.25), vec3(0.44, 0.19, -0.48), vec3(-0.01, -0.04, -0.26),
       vec3(-0.23, -0.25, 0.17)
    );

    float currentDepth = length(fragToLight);
    float shadow = 0.0;
    int samples = 16;
    float bias = max(0.05 * (1.0 - dot(normal, normalize(fragToLight))), 0.005);
    float viewDistance = length(u_viewPos - fragPos);
    float diskRadius = (1.0 + viewDistance / u_far_plane) * 0.03; // 修正

    for(int i = 0; i < samples; ++i)
    {
        float closestDepth_normalized = texture(u_depthMap, fragToLight + sample_offsets[i] * diskRadius).r;
        float closestDepth_world = closestDepth_normalized * u_far_plane; // 修正
        if(currentDepth - bias > closestDepth_world)
            shadow += 1.0;
    }
    shadow /= float(samples);
    return shadow;
}

vec3 CalculatePointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec4 texColor, float shadow)
{
    vec3 lightDir = normalize(light.position - fragPos);

    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * light.color * texColor.rgb;

    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = spec * light.color * vec3(0.5);

    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

    diffuse  *= attenuation * (1.0 - shadow);
    specular *= attenuation * (1.0 - shadow);

    return (diffuse + specular);
}