#Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for point light."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.rendering.light import point_light
from tensorflow_graphics.util import test_case


def fake_brdf(incoming_light_direction, outgoing_light_direction,
              surface_point_normal):
  del incoming_light_direction, surface_point_normal
  return outgoing_light_direction


class PointLightTest(test_case.TestCase):

  # Tests the output of the received_light function with various parameters.
  # In this test the point on the surface is always (0, 0, 0) ,the surface
  # normal is (0, 0, 1) and the fake brdf function returns the (normalized)
  # direction of the outgoing light as its output.
  @parameterized.parameters(
      # Light direction is parallel to the surface normal.
      ([0, 0, 5], [7, 0, 0], [1, 0, 0]),
      # Light direction is perpendicular to the surface normal.
      ([3, 0, 0], [1, 2, 3], [0, 0, 0]),
      # Angle between surface normal and the incoming light direction is pi/3.
      ([3, 0, math.sqrt(3)], [0, 4, 0], [0, 0.5, 0]),
      # Angle between surface normal and the incoming light direction is pi/4.
      ([0, 1, 1], [5, 5, 0], [0.5, 0.5, 0]),
  )
  def test_received_light_random(self, light_pos, camera_pos, expected_result):
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    point_light_position = np.random.uniform(
        low=-1.0, high=1.0, size=tensor_shape + [3])
    observation_point = np.random.uniform(
        low=-1.0, high=1.0, size=tensor_shape + [3])
    surface_point_normal = np.array((0.0, 0.0, 1.0))
    surface_point_position = np.array((0.0, 0.0, 0.0))
    point_light_position[..., 0:3] = light_pos
    observation_point[..., 0:3] = camera_pos

    pred = point_light.received_light(point_light_position,
                                      surface_point_position,
                                      surface_point_normal, observation_point,
                                      fake_brdf)

    expected = np.random.uniform(low=-1.0, high=1.0, size=tensor_shape + [3])
    expected[..., 0:3] = expected_result
    self.assertAllClose(expected, pred)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_received_light_jacobian_random(self):
    """Tests the Jacobian of the rendering equation."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    point_light_position_init = np.random.uniform(
        -1.0, 1.0, size=tensor_shape + [3])
    surface_point_position_init = np.random.uniform(
        -1.0, 1.0, size=tensor_shape + [3])
    surface_point_normal_init = np.random.uniform(
        -1.0, 1.0, size=tensor_shape + [3])
    observation_point_init = np.random.random(tensor_shape + [3])

    point_light_position = tf.convert_to_tensor(value=point_light_position_init)
    surface_point_position = tf.convert_to_tensor(
        value=surface_point_position_init)
    surface_point_normal = tf.convert_to_tensor(value=surface_point_normal_init)
    observation_point = tf.convert_to_tensor(value=observation_point_init)

    y = point_light.received_light(point_light_position, surface_point_position,
                                   surface_point_normal, observation_point,
                                   fake_brdf)

    self.assert_jacobian_is_correct(point_light_position,
                                    point_light_position_init, y)
    self.assert_jacobian_is_correct(surface_point_position,
                                    surface_point_position_init, y)
    self.assert_jacobian_is_correct(surface_point_normal,
                                    surface_point_normal_init, y)
    self.assert_jacobian_is_correct(observation_point, observation_point_init,
                                    y)

  @parameterized.parameters(
      ((3,), (3,), (3,), (3,)),
      ((None, 3), (None, 3), (None, 3), (None, 3)),
      ((1, 3), (1, 3), (1, 3), (1, 3)),
      ((2, 3), (2, 3), (2, 3), (2, 3)),
      ((3,), (1, 3), (1, 2, 3), (1, 3)),
      ((3,), (1, 3), (1, 2, 3), (1, 2, 2, 3)),
      ((1, 2, 2, 3), (1, 2, 3), (1, 3), (3,)),
  )
  def test_received_light_shape_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        point_light.received_light, shape, brdf_func=fake_brdf)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (1,), (3,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (2,), (3,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (4,), (3,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (1,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (2,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (4,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (1,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (2,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (4,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (3,), (4,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (3,), (2,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (3,), (1,)),
      ("Not all batch dimensions are broadcast-compatible.", (2, 3), (3, 3),
       (3,), (3,)),
  )
  def test_received_light_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(
        point_light.received_light, error_msg, shape, brdf_func=fake_brdf)

  def test_received_light_exceptions_raised(self):
    """Tests that the exceptions are raised correctly."""
    point_light_position = np.random.uniform(-1.0, 1.0, size=(3,))
    surface_point_position = np.random.uniform(-1.0, 1.0, size=(3,))
    surface_point_normal = np.random.uniform(-1.0, 1.0, size=(3,))
    observation_point = np.random.uniform(0.0, 1.0, (3,))

    with self.subTest(name="assert_on_surface_point_normal_not_normalized"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            point_light.received_light(point_light_position,
                                       surface_point_position,
                                       surface_point_normal, observation_point,
                                       fake_brdf))


if __name__ == "__main__":
  test_case.main()
