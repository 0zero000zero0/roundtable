# engine/camera.py
import numpy as np

# 定义一些常量
YAW = -90.0
PITCH = 0.0
SPEED = 500.0
SENSITIVITY = 0.1
ZOOM = 45.0


class Camera:
    def __init__(self, position=None, up=None, yaw=YAW, pitch=PITCH):
        if position is None:
            position = [-15, 15, 270]

        self.position = np.array(position, dtype=np.float32)
        self.world_up = np.array([0, 1, 0], dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.movement_speed = SPEED
        self.mouse_sensitivity = SENSITIVITY
        self.zoom = ZOOM
        self.last_mouse_x = None
        self.last_mouse_y = None
        self.first_mouse = True
        self.front = np.zeros(3, dtype=np.float32)
        self.right = np.zeros(3, dtype=np.float32)
        self.up = np.zeros(3, dtype=np.float32)
        self._update_camera_vectors()
        print(f'Camera initialized for a Y-up world at position: {self.position}')

    def get_view_matrix(self):
        """
        【已修正】根据经典LookAt矩阵的数学原理构建视图矩阵。
        """
        # 旋转部分
        rotation = np.identity(4, dtype=np.float32)
        rotation[0, 0:3] = self.right
        rotation[1, 0:3] = self.up
        rotation[2, 0:3] = -self.front

        # 平移部分
        translation = np.identity(4, dtype=np.float32)
        translation[3, 0] = -np.dot(self.right, self.position)
        translation[3, 1] = -np.dot(self.up, self.position)
        translation[3, 2] = -np.dot(-self.front, self.position)

        # 最终的视图矩阵 = 平移矩阵 * 旋转矩阵 (注意顺序)
        return translation @ rotation

    def process_keyboard(self, direction, delta_time):
        velocity = self.movement_speed * delta_time
        if direction == 'FORWARD':
            self.position += self.front * velocity
        if direction == 'BACKWARD':
            self.position -= self.front * velocity
        if direction == 'LEFT':
            self.position -= self.right * velocity
        if direction == 'RIGHT':
            self.position += self.right * velocity

    def process_mouse_movement(self, x_offset, y_offset, constrain_pitch=True):
        self.yaw += x_offset * self.mouse_sensitivity
        self.pitch += y_offset * self.mouse_sensitivity
        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0
        self._update_camera_vectors()

    def process_mouse_scroll(self, y_offset):
        self.zoom -= y_offset
        if self.zoom < 1.0:
            self.zoom = 1.0
        if self.zoom > 60.0:
            self.zoom = 60.0

    def _update_camera_vectors(self):
        front_calc = np.zeros(3, dtype=np.float32)
        front_calc[0] = np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        front_calc[1] = np.sin(np.radians(self.pitch))
        front_calc[2] = np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        self.front = front_calc / np.linalg.norm(front_calc)
        self.right = np.cross(self.front, self.world_up) / np.linalg.norm(
            np.cross(self.front, self.world_up)
        )
        self.up = np.cross(self.right, self.front) / np.linalg.norm(
            np.cross(self.right, self.front)
        )
