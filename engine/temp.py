# engine/app.py

import ctypes
import os

import glfw
import numpy as np
import OpenGL.GL as gl

from .camera import Camera
from .lights import get_lights
from .shader import Shader

# from .text_renderer import TextRenderer # 已移除

# 定义常量
CACHE_DIR = 'cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'model_cache.pkl')
SHADOW_WIDTH, SHADOW_HEIGHT = 1024, 1024


class App:
    def __init__(
        self, width=2560, height=1600, title='Roundtable Hold', reload_assets=False
    ):
        # ... (属性初始化无变化) ...
        self.width = width
        self.height = height
        self.title = title
        self.last_frame_time = 0.0
        self.delta_time = 0.0
        self.last_mouse_x = self.width / 2
        self.last_mouse_y = self.height / 2
        self.first_mouse = True
        self.near_plane = 1.0
        self.far_plane = 3000.0

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

    def _set_glfw_callbacks(self):
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        glfw.set_cursor_pos_callback(self.window, self._mouse_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

    def _initialize_opengl_states(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0.05, 0.05, 0.08, 1.0)
        gl.glDisable(gl.GL_CULL_FACE)

    def _create_default_texture(self):
        # ... (无变化) ...
        pass

    def _load_texture_from_image(self, image):
        # ... (无变化) ...
        pass

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

        # 2. 创建阴影渲染的FBO和深度立方体贴图
        # ... (这部分代码无变化) ...

        # 3. 加载主场景模型数据
        # ... (缓存逻辑无变化) ...

        # 4. 创建主场景的VBO和专用的VAO
        # ... (这部分代码无变化，确保创建了 self.main_vao 和 self.depth_vao) ...

        # 5. 准备主场景的分批次绘制指令
        # ... (这部分代码无变化，确保创建了 self.draw_calls) ...

        # 6. 【新增】为光源标识创建一个立方体模型
        cube_vertices = np.array(
            [
                -1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
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
        self.ambient_light = [0.15, 0.15, 0.18]
        self.lights = get_lights()

    # ... (_create_projection_matrix, _create_look_at_matrix, _mouse_callback, _scroll_callback, _process_input 无变化) ...

    def run(self):
        """【已重构】主循环，严格按照 深度->主场景->光源标识 的顺序渲染"""
        model_matrix = np.identity(4, dtype=np.float32)

        while not glfw.window_should_close(self.window):
            current_frame_time = glfw.get_time()
            self.delta_time = current_frame_time - self.last_frame_time
            self.last_frame_time = current_frame_time

            glfw.poll_events()
            self._process_input()

            # --- 准备每一帧都需要的矩阵 ---
            projection_matrix = self._create_projection_matrix(
                self.camera.zoom,
                self.width / self.height,
                self.near_plane,
                self.far_plane,
            )
            view_matrix = self.camera.get_view_matrix()

            # --- PASS 1: 渲染深度立方体贴图 ---
            self.render_depth_pass(model_matrix)

            # --- PASS 2: 正常渲染主场景 ---
            self.render_main_pass(projection_matrix, view_matrix, model_matrix)

            # --- PASS 3: 渲染光源标识 ---
            self.render_light_markers(projection_matrix, view_matrix)

            glfw.swap_buffers(self.window)

        self._cleanup()

    def render_depth_pass(self, model_matrix):
        """执行深度渲染遍"""
        main_light_pos = self.lights[0]['position']
        shadow_projection = self._create_projection_matrix(
            90.0, SHADOW_WIDTH / SHADOW_HEIGHT, self.near_plane, self.far_plane
        )
        shadow_transforms = [
            self._create_look_at_matrix(main_light_pos, main_light_pos + d, u)
            for d, u in [
                (np.array([1, 0, 0]), np.array([0, -1, 0])),
                (np.array([-1, 0, 0]), np.array([0, -1, 0])),
                (np.array([0, 1, 0]), np.array([0, 0, 1])),
                (np.array([0, -1, 0]), np.array([0, 0, -1])),
                (np.array([0, 0, 1]), np.array([0, -1, 0])),
                (np.array([0, 0, -1]), np.array([0, -1, 0])),
            ]
        ]
        final_shadow_transforms = [shadow_projection @ vt for vt in shadow_transforms]

        gl.glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depth_cubemap_fbo)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        self.depth_shader.use()
        for i in range(6):
            self.depth_shader.set_mat4(
                f'shadowMatrices[{i}]', final_shadow_transforms[i]
            )
        self.depth_shader.set_mat4('model', model_matrix)
        self.depth_shader.set_vec3('lightPos', main_light_pos)
        self.depth_shader.set_float('far_plane', self.far_plane)

        gl.glBindVertexArray(self.depth_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.total_vertex_count)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def render_main_pass(self, projection_matrix, view_matrix, model_matrix):
        """执行主场景渲染遍"""
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

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

    def _cleanup(self):
        """程序退出时清理所有OpenGL对象。"""
        print('Cleaning up resources...')
        gl.glDeleteVertexArrays(
            3, [self.main_vao, self.depth_vao, self.light_marker_vao]
        )
        gl.glDeleteBuffers(2, [self.vbo, self.light_marker_vbo])
        all_texture_ids = list(self.texture_cache.values())
        all_texture_ids.append(self.default_texture_id)
        gl.glDeleteTextures(all_texture_ids)
        gl.glDeleteTextures(1, [self.depth_cubemap])
        gl.glDeleteFramebuffers(1, [self.depth_cubemap_fbo])
        glfw.terminate()
