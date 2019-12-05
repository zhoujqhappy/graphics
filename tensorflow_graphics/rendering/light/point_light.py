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
"""This module implements the rendering equation for a point light."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def received_light(point_light_position,
                   surface_point_position,
                   surface_point_normal,
                   observation_point,
                   brdf_func,
                   name=None):
  """Evaluates the amount of light received at the observation point.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Note:
    The function assumes a light with radiance 1.0 and a surface which doesn't
    emit light.

  Args:
    point_light_position: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the position of the point light.
    surface_point_position: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the position of the surface point.
    surface_point_normal: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the surface normal at the given surface point.
    observation_point: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the observation point.
    brdf_func: The BRDF as a function of the incoming light direction, outgoing
      light direction and the surface normal.
    name: A name for this op. Defaults to "received_light".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
      the amount of light received at the observation point after being
      reflected from the given surface point.

  Raises:
    ValueError: if the shape of `point_light_position`,
    `surface_point_position`, `surface_point_normal`, or `observation_point` is
    not supported.
    InvalidArgumentError: if 'surface_point_normal' is not normalized.
  """
  with tf.compat.v1.name_scope(name, "received_light", [
      point_light_position, surface_point_position, surface_point_normal,
      brdf_func, observation_point
  ]):
    point_light_position = tf.convert_to_tensor(value=point_light_position)
    surface_point_position = tf.convert_to_tensor(value=surface_point_position)
    surface_point_normal = tf.convert_to_tensor(value=surface_point_normal)
    observation_point = tf.convert_to_tensor(value=observation_point)

    shape.check_static(
        tensor=point_light_position,
        tensor_name="point_light_position",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=surface_point_position,
        tensor_name="surface_point_position",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=surface_point_normal,
        tensor_name="surface_point_normal",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=observation_point,
        tensor_name="observation_point",
        has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(point_light_position, surface_point_position,
                 surface_point_normal, observation_point),
        tensor_names=("point_light_position", "surface_point_position",
                      "surface_point_normal", "observation_point"),
        last_axes=-2,
        broadcast_compatible=True)
    surface_point_normal = asserts.assert_normalized(surface_point_normal)

    incoming_light_direction = tf.math.l2_normalize(surface_point_position -
                                                    point_light_position)
    outgoing_light_direction = tf.math.l2_normalize(observation_point -
                                                    surface_point_position)
    brdf_val = brdf_func(incoming_light_direction, outgoing_light_direction,
                         surface_point_normal)
    incoming_light_dot_surface_normal = vector.dot(-incoming_light_direction,
                                                   surface_point_normal)
    return brdf_val * incoming_light_dot_surface_normal


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
