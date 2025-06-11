# engine/app.py

# -----------------------------------------------------------------------------
# 1. 导入必要的库
# -----------------------------------------------------------------------------
import ctypes  # 用于设置顶点属性指针的偏移量
import os
import pickle

import glfw
import numpy as np

# 推荐的OpenGL导入方式，并使用gl前缀
import OpenGL.GL as gl

# 图像处理库，用于加载纹理
import PIL.Image
import pywavefront

# 导入你自己的引擎模块
from .camera import Camera
from .lights import get_lights
from .shader import Shader

# -----------------------------------------------------------------------------
# 2. 定义全局常量
# -----------------------------------------------------------------------------
CACHE_DIR = 'cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'model_cache.pkl')
SHADOW_WIDTH, SHADOW_HEIGHT = 1024, 1024

# --- 【调试开关】 ---
# 将其设置为True，可以可视化法线，用于检查模型渲染和矩阵变换是否正确
DEBUG_VISUALIZE_NORMALS = True


# -----------------------------------------------------------------------------
# 3. 主应用程序类 App
# -----------------------------------------------------------------------------
class App:
    """
    主应用程序类，负责管理窗口、主循环、资源和渲染。
    """

    def __init__(
        self, width=2560, height=1600, title='Roundtable Hold', reload_assets=False
    ):
        # --- 窗口与时间属性 ---
        self.width = width
        self.height = height
        self.title = title
        self.last_frame_time = 0.0
        self.delta_time = 0.0

        # --- 输入状态属性 ---
        self.last_mouse_x = self.width / 2
        self.last_mouse_y = self.height / 2
        self.first_mouse = True

        # --- 渲染属性 ---
        self.near_plane = 1.0
        self.far_plane = 3000.0

        # --- 初始化窗口和OpenGL上下文 ---
        if not glfw.init():
            raise Exception("GLFW can't be initialized")

        self.window = glfw.create_window(
            self.width, self.height, self.title, None, None
        )
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")

        glfw.make_context_current(self.window)
        self._set_glfw_callbacks()
        self._initialize_opengl_states()
        self._initialize_components(reload=reload_assets)

    # -------------------------------------------------------------------------
    # 辅助方法 (Initialization Helpers)
    # -------------------------------------------------------------------------

    def _set_glfw_callbacks(self):
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        glfw.set_cursor_pos_callback(self.window, self._mouse_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

    def _initialize_opengl_states(self):
        """设置固定的OpenGL状态。"""
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0.05, 0.05, 0.08, 1.0)
        gl.glDisable(gl.GL_CULL_FACE)
        # 禁用背面剔除，因为我们在室内

    def _create_default_texture(self):
        """创建一个1x1的纯白色纹理，作为无纹理材质的备用。"""
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        white_pixel = np.array([255, 255, 255, 255], dtype=np.uint8)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            1,
            1,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            white_pixel,
        )
        return texture_id

    def _load_texture_from_image(self, image):
        """辅助方法：从Pillow图像对象创建OpenGL纹理，并处理sRGB格式。"""
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        img_data = image.convert('RGBA').tobytes()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_SRGB_ALPHA,
            image.width,
            image.height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            img_data,
        )
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        return texture_id

    def _initialize_components(self, reload=False):
        """【已重构】加载所有资源，并为光源标识创建额外的VAO/VBO。"""
        # 1. 加载所有着色器程序
        self.shader = Shader('shaders/vertex.glsl', 'shaders/fragment.glsl')
        self.depth_shader = Shader(
            'shaders/depth_vertex.glsl',
            'shaders/depth_fragment.glsl',
            'shaders/depth_geometry.glsl',
        )
        self.light_marker_shader = Shader(
            'shaders/light_marker_vertex.glsl', 'shaders/light_marker_fragment.glsl'
        )

        # 2. 创建阴影渲染所需的FBO和深度立方体贴图
        self.depth_cubemap_fbo = gl.glGenFramebuffers(1)
        self.depth_cubemap = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.depth_cubemap)
        for i in range(6):
            gl.glTexImage2D(
                gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,  # type: ignore
                0,
                gl.GL_DEPTH_COMPONENT,
                SHADOW_WIDTH,
                SHADOW_HEIGHT,
                0,
                gl.GL_DEPTH_COMPONENT,
                gl.GL_FLOAT,
                None,
            )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depth_cubemap_fbo)
        gl.glFramebufferTexture(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, self.depth_cubemap, 0
        )
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise Exception('Framebuffer is not complete!')
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # 3. 加载模型数据（带缓存逻辑）
        self.default_texture_id = self._create_default_texture()
        self.texture_cache = {}
        serializable_draw_info = []

        if reload or not os.path.exists(CACHE_FILE):
            print('Loading assets from source file (.obj)...')
            scene = pywavefront.Wavefront(
                'assets/rountdtableroom.obj', create_materials=True, collect_faces=True
            )
            all_vertices = []
            for _name, material in scene.materials.items():
                all_vertices.extend(material.vertices)
                texture_path = None
                if material.texture:
                    texture_path = material.texture.path
                draw_info = {
                    'texture_path': texture_path,
                    'vertex_count': len(material.vertices) // 8,
                }
                serializable_draw_info.append(draw_info)
            self.vertex_data = np.array(all_vertices, dtype=np.float32)
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_data = {
                'vertex_data': self.vertex_data,
                'draw_info_list': serializable_draw_info,
            }
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f'Assets cached to {CACHE_FILE}')
        else:
            print(f'Loading assets from cache ({CACHE_FILE})...')
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
            self.vertex_data = cache_data['vertex_data']
            serializable_draw_info = cache_data['draw_info_list']

        # 4. 创建VBO和专用的VAO
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.vertex_data.nbytes,
            self.vertex_data,
            gl.GL_STATIC_DRAW,
        )

        stride = 8 * 4

        self.main_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.main_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(5 * 4)
        )
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0)
        )
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(2 * 4)
        )
        gl.glEnableVertexAttribArray(2)

        self.depth_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.depth_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(5 * 4)
        )
        gl.glEnableVertexAttribArray(0)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # 5. 准备分批次绘制指令和纹理
        self.draw_calls = []
        vertex_offset = 0
        for info in serializable_draw_info:
            current_texture_id = self.default_texture_id
            if info['texture_path']:
                texture_path = info['texture_path']
                if texture_path in self.texture_cache:
                    current_texture_id = self.texture_cache[texture_path]
                else:
                    try:
                        image = PIL.Image.open(texture_path).transpose(
                            PIL.Image.Transpose.FLIP_TOP_BOTTOM
                        )
                        new_texture_id = self._load_texture_from_image(image)
                        self.texture_cache[texture_path] = new_texture_id
                        current_texture_id = new_texture_id
                    except Exception as e:
                        print(
                            f"WARNING: Failed to load texture '{texture_path}' due to error: {e}. Using default texture."
                        )
            self.draw_calls.append(
                {
                    'texture_id': current_texture_id,
                    'start_index': vertex_offset,
                    'vertex_count': info['vertex_count'],
                }
            )
            vertex_offset += info['vertex_count']

        self.total_vertex_count = vertex_offset
        print(
            f'Loaded a total of {self.total_vertex_count} vertices across {len(self.draw_calls)} materials.'
        )
        # 6. 【新增】为光源标识创建一个立方体模型
        cube_vertices = np.array(
            [
                -0.25,
                -0.25,
                -0.25,
                0.25,
                -0.25,
                -0.25,
                0.25,
                0.25,
                -0.25,
                0.25,
                0.25,
                -0.25,
                -0.25,
                0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                0.25,
                0.25,
                -0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                -0.25,
                0.25,
                0.25,
                -0.25,
                -0.25,
                0.25,
                -0.25,
                0.25,
                0.25,
                -0.25,
                0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                0.25,
                -0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                -0.25,
                0.25,
                -0.25,
                -0.25,
                0.25,
                -0.25,
                -0.25,
                0.25,
                -0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                -0.25,
                -0.25,
                -0.25,
                0.25,
                -0.25,
                -0.25,
                0.25,
                -0.25,
                0.25,
                0.25,
                0.25,
                -0.25,
                0.25,
                -0.25,
                -0.25,
                0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                0.25,
                -0.25,
                0.25,
                0.25,
                -0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                -0.25,
                0.25,
                0.25,
                -0.25,
                0.25,
                -0.25,
            ],
            dtype=np.float32,
        )
        self.light_marker_vao = gl.glGenVertexArrays(1)
        self.light_marker_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.light_marker_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.light_marker_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, gl.GL_STATIC_DRAW
        )
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, ctypes.c_void_p(0)
        )
        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # 7. 初始化相机和光照
        self.camera = Camera()
        self.ambient_light = [0.001] * 3  # 环境光
        self.lights = get_lights()

    # -------------------------------------------------------------------------
    # 输入处理方法 (Input Handlers)
    # -------------------------------------------------------------------------

    def _mouse_callback(self, window, xpos, ypos):
        if self.first_mouse:
            self.last_mouse_x = xpos
            self.last_mouse_y = ypos
            self.first_mouse = False
        x_offset = xpos - self.last_mouse_x
        y_offset = self.last_mouse_y - ypos
        self.last_mouse_x = xpos
        self.last_mouse_y = ypos
        self.camera.process_mouse_movement(x_offset, y_offset)

    def _scroll_callback(self, window, x_offset, y_offset):
        self.camera.process_mouse_scroll(y_offset)

    def _process_input(self):
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.camera.process_keyboard('FORWARD', self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.camera.process_keyboard('BACKWARD', self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.camera.process_keyboard('LEFT', self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.camera.process_keyboard('RIGHT', self.delta_time)

    # -------------------------------------------------------------------------
    # 辅助方法 (Matrix Helpers)
    # -------------------------------------------------------------------------

    def _create_projection_matrix(self, fov_degrees, aspect_ratio, near, far):
        fov = np.radians(fov_degrees)
        f = 1.0 / np.tan(fov / 2.0)
        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0] = f / aspect_ratio
        projection[1, 1] = f
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = -1.0
        projection[3, 2] = (2.0 * far * near) / (near - far)
        return projection

    def _create_look_at_matrix(self, position, target, world_up):
        """【已修正】一个独立的、数学上正确的LookAt矩阵辅助函数"""
        z_axis = (position - target) / np.linalg.norm(position - target)
        x_axis = np.cross(world_up, z_axis) / np.linalg.norm(np.cross(world_up, z_axis))
        y_axis = np.cross(z_axis, x_axis)

        rotation = np.identity(4, dtype=np.float32)
        rotation[0, 0:3] = x_axis
        rotation[1, 0:3] = y_axis
        rotation[2, 0:3] = z_axis

        translation = np.identity(4, dtype=np.float32)
        translation[3, 0] = -np.dot(x_axis, position)
        translation[3, 1] = -np.dot(y_axis, position)
        translation[3, 2] = -np.dot(z_axis, position)

        return translation @ rotation

    # -------------------------------------------------------------------------
    # 渲染不同场景
    # -------------------------------------------------------------------------
    def render_depth_pass(self, model_matrix):
        """执行深度渲染遍，并使用正确的面剔除设置。"""
        main_light_pos = self.lights[0]['position']
        shadow_projection = self._create_projection_matrix(90.0, SHADOW_WIDTH / SHADOW_HEIGHT, self.near_plane, self.far_plane)
        shadow_transforms = [ self._create_look_at_matrix(main_light_pos, main_light_pos + d, u) for d, u in [
            (np.array([1,0,0]), np.array([0,-1,0])), (np.array([-1,0,0]), np.array([0,-1,0])),
            (np.array([0,1,0]), np.array([0,0,1])), (np.array([0,-1,0]), np.array([0,0,-1])),
            (np.array([0,0,1]), np.array([0,-1,0])), (np.array([0,0,-1]), np.array([0,-1,0]))
        ]]
        final_shadow_transforms = [shadow_projection @ vt for vt in shadow_transforms]

        gl.glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depth_cubemap_fbo)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        # --- 【核心修正】在深度遍中，我们暂时完全禁用面剔除 ---
        # 这确保了无论我们看到的是正面还是背面，都会被写入深度图
        gl.glDisable(gl.GL_CULL_FACE)

        self.depth_shader.use()
        for i in range(6):
            self.depth_shader.set_mat4(f"shadowMatrices[{i}]", final_shadow_transforms[i])
        self.depth_shader.set_mat4("model", model_matrix)
        self.depth_shader.set_vec3("lightPos", main_light_pos)
        self.depth_shader.set_float("far_plane", self.far_plane)

        gl.glBindVertexArray(self.depth_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.total_vertex_count)

        # 恢复默认状态，为主渲染做准备
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def render_main_pass(self, projection_matrix, view_matrix, model_matrix):
        """执行主场景渲染遍"""
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # type: ignore
        gl.glDisable(gl.GL_CULL_FACE)  # 主渲染禁用剔除
        self.shader.use()

        self.shader.set_mat4('view', view_matrix)
        self.shader.set_mat4('projection', projection_matrix)
        self.shader.set_mat4('model', model_matrix)
        self.shader.set_vec3('u_viewPos', self.camera.position)
        self.shader.set_vec3('u_ambient_light', self.ambient_light)
        self.shader.set_float('u_far_plane', self.far_plane)
        self.shader.set_vec3('u_lightPos', self.lights[0]['position'])

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.depth_cubemap)
        self.shader.set_int('u_depthMap', 1)

        for i, light in enumerate(self.lights):
            light_color = np.array(light['color']) * light.get('strength', 1.0)
            self.shader.set_vec3(f'u_lights[{i}].position', light['position'])
            self.shader.set_vec3(f'u_lights[{i}].color', light_color)
            self.shader.set_float(f'u_lights[{i}].constant', 1.0)
            self.shader.set_float(f'u_lights[{i}].linear', 0.0014)
            self.shader.set_float(f'u_lights[{i}].quadratic', 0.000007)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        self.shader.set_int('u_texture', 0)

        gl.glBindVertexArray(self.main_vao)
        for call in self.draw_calls:
            gl.glBindTexture(gl.GL_TEXTURE_2D, call['texture_id'])
            gl.glDrawArrays(gl.GL_TRIANGLES, call['start_index'], call['vertex_count'])
        gl.glBindVertexArray(0)

    def render_light_markers(self, projection_matrix, view_matrix):
        """执行光源标识渲染遍"""
        self.light_marker_shader.use()
        self.light_marker_shader.set_mat4('view', view_matrix)
        self.light_marker_shader.set_mat4('projection', projection_matrix)

        gl.glBindVertexArray(self.light_marker_vao)
        for light in self.lights:
            translate_matrix = np.identity(4, dtype=np.float32)
            translate_matrix[3, 0:3] = light['position']
            scale_matrix = np.diag([10.0, 10.0, 10.0, 1.0])
            marker_model_matrix = scale_matrix @ translate_matrix

            self.light_marker_shader.set_mat4('model', marker_model_matrix)
            self.light_marker_shader.set_vec3('lightColor', light['color'])
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)
        gl.glBindVertexArray(0)

    # -------------------------------------------------------------------------
    # 主循环与清理
    # -------------------------------------------------------------------------

    def run(self):
        model_matrix = np.identity(4, dtype=np.float32)
        while not glfw.window_should_close(self.window):
            current_frame_time = glfw.get_time()
            self.delta_time = current_frame_time - self.last_frame_time
            self.last_frame_time = current_frame_time
            glfw.poll_events()
            self._process_input()

            projection_matrix = self._create_projection_matrix(
                self.camera.zoom,
                self.width / self.height,
                self.near_plane,
                self.far_plane,
            )
            view_matrix = self.camera.get_view_matrix()

            self.render_depth_pass(model_matrix)
            self.render_main_pass(projection_matrix, view_matrix, model_matrix)
            self.render_light_markers(projection_matrix, view_matrix)

            # 渲染HUD
            gl.glDisable(gl.GL_DEPTH_TEST)

            gl.glEnable(gl.GL_DEPTH_TEST)

            glfw.swap_buffers(self.window)
        self._cleanup()

    def _cleanup(self):
        """程序退出时清理所有OpenGL对象。"""
        print('Cleaning up resources...')
        gl.glDeleteVertexArrays(2, [self.main_vao, self.depth_vao])
        gl.glDeleteBuffers(1, [self.vbo])

        all_texture_ids = list(self.texture_cache.values())
        all_texture_ids.append(self.default_texture_id)
        gl.glDeleteTextures(all_texture_ids)

        gl.glDeleteTextures(1, [self.depth_cubemap])
        gl.glDeleteFramebuffers(1, [self.depth_cubemap_fbo])
        glfw.terminate()
