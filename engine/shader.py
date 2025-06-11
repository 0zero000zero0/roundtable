# engine/shader.py
import OpenGL.GL as gl


class Shader:
    def __init__(
        self, vertex_path, fragment_path, geometry_path=None
    ):
        """
        初始化Shader类，可以额外加载几何着色器。
        """
        try:
            with open(vertex_path, encoding='utf-8') as f:
                vertex_source = f.read()
            with open(fragment_path, encoding='utf-8') as f:
                fragment_source = f.read()

            geometry_source = None
            if geometry_path:
                with open(geometry_path, encoding='utf-8') as f:
                    geometry_source = f.read()
        except FileNotFoundError as e:
            print(f'Error: Shader file not found. {e}')
            raise

        vertex_shader = self._compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
        fragment_shader = self._compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)

        geometry_shader = None
        if geometry_source:
            geometry_shader = self._compile_shader(
                geometry_source, gl.GL_GEOMETRY_SHADER
            )

        self.program_id = self._link_program(
            vertex_shader, fragment_shader, geometry_shader
        )

        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        if geometry_shader:
            gl.glDeleteShader(geometry_shader)

    def _compile_shader(self, source, shader_type):
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            info_log = gl.glGetShaderInfoLog(shader).decode('utf-8')
            shader_type_str = (
                'Vertex'
                if shader_type == gl.GL_VERTEX_SHADER
                else 'Fragment'
                if shader_type == gl.GL_FRAGMENT_SHADER
                else 'Geometry'
            )
            raise Exception(
                f'ERROR::SHADER::{shader_type_str}::COMPILATION_FAILED\n{info_log}'
            )
        return shader

    def _link_program(self, vertex_shader, fragment_shader, geometry_shader=None):
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        if geometry_shader:
            gl.glAttachShader(program, geometry_shader)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            info_log = gl.glGetProgramInfoLog(program).decode('utf-8')
            raise Exception(f'ERROR::PROGRAM::LINKING_FAILED\n{info_log}')
        return program

    def use(self):
        gl.glUseProgram(self.program_id)

    def set_mat4(self, name, matrix):
        location = gl.glGetUniformLocation(self.program_id, name)
        if location != -1:
            gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE, matrix)

    def set_vec3(self, name, vector):
        location = gl.glGetUniformLocation(self.program_id, name)
        if location != -1:
            gl.glUniform3fv(location, 1, vector)

    def set_float(self, name, value):
        location = gl.glGetUniformLocation(self.program_id, name)
        if location != -1:
            gl.glUniform1f(location, value)

    def set_int(self, name, value):
        location = gl.glGetUniformLocation(self.program_id, name)
        if location != -1:
            gl.glUniform1i(location, value)
