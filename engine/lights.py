# Blender (x1,y1,z1) -> -Z forward, Y up ->OpenGL(x1,-z1,y1)
def get_lights():
    # 1. 定义一个火焰色板
    main_flame_color = [1, 1, 1]  # 中央火焰，更亮、偏黄白
    torch_color_A = [1.0, 0.6, 0.2]  # 火炬颜色A，经典的橙黄色
    torch_color_B = [1.0, 0.5, 0.1]  # 火炬颜色B，更深、偏红的橙色
    # 2. 定义不同的强度级别
    main_flame_strength = 1  # 主光源应该非常强
    torch_strength_strong = 0.00  # 较亮的火炬
    torch_strength_weak = 0.00  # 稍暗的火炬
    lights = [
        # --- 主光源 ---
        # Blender: [-21.91, -48.59, 32.39] => OpenGL: [-21.91, 32.39, 48.59]
        {
            'position': [-21.91, 32.39, 48.59],
            'color': main_flame_color,
            'strength': main_flame_strength,
        },
        # --- 周围的火炬 ---
        {
            # 1
            'position': [-164.848, 14.5924, 179.305],
            'color': torch_color_A,
            'strength': torch_strength_strong,
        },
        {
            # 2
            'position': [-164.948, 14.3205, 64.2643],
            'color': torch_color_B,
            'strength': torch_strength_weak,
        },
        {
            # 3
            'position': [-161.317, 15.1098, -55.4458],
            'color': torch_color_A,
            'strength': torch_strength_strong,
        },
        {
            # 4
            'position': [-170.189, 12.0041, -157.297],
            'color': torch_color_B,
            'strength': torch_strength_weak,
        },
        {
            # 5
            'position': [-85.6, 13.6, -146.7],
            'color': torch_color_A,
            'strength': torch_strength_strong,
        },
        {
            # 6
            'position': [22.2, 14.03, -143],
            'color': torch_color_B,
            'strength': torch_strength_strong,
        },
        {
            # 7
            'position': [106.5, 13.78, -157.4],
            'color': torch_color_A,
            'strength': torch_strength_weak,
        },
        {
            # 8
            'position': [103.4, 15.53, -59.22],
            'color': torch_color_B,
            'strength': torch_strength_strong,
        },
        {
            # 9
            'position': [102.5, 15.81, 56.44],
            'color': torch_color_A,
            'strength': torch_strength_weak,
        },
        {
            # 10
            'position': [105.6, 13.05, 176.3],
            'color': torch_color_B,
            'strength': torch_strength_weak,
        },
    ]
    return lights


def get_vertex():
    # list[[x,y,z]]
    points = [
        [125, 175, 55],
        [-169, 175, 55],
        [-169, 175, -41],
        [129, 175, -41],
        [-169, -238, 55],
        [-167, -238, -43],
        [125, -238, 43],
        [125, -238, 55],
    ]
    return points
    pass
