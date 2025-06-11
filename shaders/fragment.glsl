#version 330 core

// -----------------------------------------------------
// 1. 输入/输出变量
// -----------------------------------------------------
// 从顶点着色器传入的、经过插值的数据
in vec2 v_texCoord;
in vec3 v_normal;
in vec3 v_fragPos;

// 输出到屏幕的最终像素颜色
out vec4 FragColor;

// -----------------------------------------------------
// 2. Uniforms (由Python代码在每一帧设置)
// -----------------------------------------------------
// 纹理采样器
uniform sampler2D u_texture;
uniform samplerCube u_depthMap; // 用于阴影计算的深度立方体贴图

// 场景与相机属性
uniform vec3 u_viewPos;         // 相机在世界空间的位置
uniform vec3 u_ambient_light;   // 全局环境光的颜色和强度
uniform float u_far_plane;        // 光源视锥体的远裁切平面距离
uniform vec3 u_lightPos;        // 主光源在世界空间的位置 (用于阴影计算)

// 调试开关
uniform int u_visualize_normals;

// -----------------------------------------------------
// 3. 结构体定义
// -----------------------------------------------------
struct PointLight {
    vec3 position;
    vec3 color;
    float constant;
    float linear;
    float quadratic;
};

#define NUM_LIGHTS 11 // 确保这个值和你的lights.py中的光源数量一致
uniform PointLight u_lights[NUM_LIGHTS];


// -----------------------------------------------------
// 4. 函数原型声明
// -----------------------------------------------------
float CalculateShadow(vec3 fragPos, vec3 lightPos);
vec3 CalculatePointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec4 texColor, float shadow);


// =======================================================
//                       主函数 (Main)
// =======================================================
void main()
{

    // =======================================================
    // --- 调试模式A：可视化深度图 ---
    // =======================================================
    // 这段代码会绕过所有光照计算，直接将从阴影图中采样的深度值显示为颜色

    // vec3 fragToLight = v_fragPos - u_lightPos;
    // // 从深度立方体贴图中采样，获取[0,1]范围的线性深度值
    // float depthValue = texture(u_depthMap, fragToLight).r;

    // // 将深度值作为灰度颜色输出
    // FragColor = vec4(vec3(depthValue), 1.0);

    // // 注意：在这个调试模式下，我们不需要下面的任何代码，所以可以提前结束
    // return;


    // 准备基础向量
    vec3 norm = normalize(v_normal);

    // --- 【调试模式】 ---
    if (u_visualize_normals == 1) {
        // 如果开关打开，直接输出法线颜色并退出
        // 将法线从[-1, 1]范围映射到[0, 1]的颜色范围以便可视化
        FragColor = vec4(norm * 0.5 + 0.5, 1.0);
        return;
    }

    // --- 正常的光照计算代码 ---
    vec3 viewDir = normalize(u_viewPos - v_fragPos);
    vec4 texColor = texture(u_texture, v_texCoord);

    // 计算环境光
    vec3 ambient = u_ambient_light * texColor.rgb;


    // 1. 计算主光源的阴影
    float shadow = CalculateShadow(v_fragPos, u_lightPos);

    // 2. 计算光照并应用阴影
    vec3 lighting_result = vec3(0.0);
    lighting_result += CalculatePointLight(u_lights[0], normalize(v_normal), v_fragPos, normalize(u_viewPos - v_fragPos), texture(u_texture, v_texCoord), shadow);
    for(int i = 1; i < NUM_LIGHTS; i++)
    {
        lighting_result += CalculatePointLight(u_lights[i], normalize(v_normal), v_fragPos, normalize(u_viewPos - v_fragPos), texture(u_texture, v_texCoord), 0.0);
    }

    vec3 final_color = u_ambient_light * texture(u_texture, v_texCoord).rgb + lighting_result;

    // 3. 伽马校正
    final_color = pow(final_color, vec3(1.0/2.2));
    FragColor = vec4(final_color, 1.0);
}


// =======================================================
//                       辅助函数
// =======================================================

/**
 * @brief 计算一个点是否在主光源的阴影中。
 * @param fragPos 当前片元的世界坐标。
 * @param lightPos 主光源的世界坐标。
 * @return 阴影系数 (0.0代表完全照亮，1.0代表完全在阴影中)。
 */
float CalculateShadow(vec3 fragPos, vec3 lightPos)
{
    vec3 fragToLight = fragPos - lightPos;
    float currentDepth = length(fragToLight);

    float shadow = 0.0;
    float bias = 0.15;
    int samples = 4;
    float viewDistance = length(u_viewPos - fragPos);
    // 根据距离调整采样半径，实现近处阴影锐利，远处阴影模糊的效果
    float diskRadius = (1.0 + viewDistance / u_far_plane) / 25.0;

    // 对周围进行多次采样
    for(int i = 0; i < samples; ++i)
    {
        for(int j = 0; j < samples; ++j)
        {
             for(int k = 0; k < samples; ++k)
             {
                float closestDepth = texture(u_depthMap, fragToLight + vec3(i,j,k) * diskRadius).r;
                closestDepth *= u_far_plane;
                if(currentDepth - bias > closestDepth)
                    shadow += 1.0;
             }
        }
    }

    // 取平均值
    shadow /= float(samples * samples * samples);

    return shadow;
}
/**
 * @brief 计算单个点光源对片元的贡献（冯氏模型）。
 * @param shadow 阴影系数，由CalculateShadow函数计算得出。
 */
vec3 CalculatePointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec4 texColor, float shadow)
{
    vec3 lightDir = normalize(light.position - fragPos);

    // 漫反射 (Diffuse)
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * light.color * texColor.rgb;

    // 镜面光 (Specular) - Blinn-Phong
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = spec * light.color * vec3(0.5);

    // 衰减 (Attenuation)
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

    // 使用 (1.0 - shadow) 来应用阴影
    diffuse  *= attenuation * (1.0 - shadow);
    specular *= attenuation * (1.0 - shadow);

    return (diffuse + specular);
}
