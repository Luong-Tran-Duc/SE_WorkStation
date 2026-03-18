#!/usr/bin/env python3
# WARNING: SuperGlue is allowed to be used for non-commercial research purposes!!
#        : You must carefully check and follow its licensing condition!!
#        : https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE
from email.mime import image
import sys
import cv2
import math
import json
import matplotlib
import torch
import numpy
import argparse
import matplotlib
from models.matching import Matching
from models.utils import (make_matching_plot_fast, frame2tensor)

def main():
  parser = argparse.ArgumentParser(description='Initial guess estimation based on SuperGlue', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('data_path', help='Input data path')
  parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor', help='SuperGlue weights')
  parser.add_argument('--max_keypoints', type=int, default=-1, help='Maximum number of keypoints detected by Superpoint' ' (\'-1\' keeps all keypoints)')
  parser.add_argument('--keypoint_threshold', type=float, default=0.05, help='SuperPoint keypoint detector confidence threshold')
  parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
  parser.add_argument('--sinkhorn_iterations', type=int, default=20, help='Number of Sinkhorn iterations performed by SuperGlue')
  parser.add_argument('--match_threshold', type=float, default=0.01, help='SuperGlue match threshold')
  parser.add_argument('--show_keypoints', action='store_true', help='Show the detected keypoints')
  parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
  parser.add_argument('--rotate_lidar', type=int, default=0, help='Rotate LiDAR image before matching (0, 90, 180, or 270) (CW)')

  opt = parser.parse_args()
  print(opt)

  torch.set_grad_enabled(False)
  device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'

  print('Running inference on device \"{}\"'.format(device))
  config = {
    'superpoint': {
      'nms_radius': opt.nms_radius,
      'keypoint_threshold': opt.keypoint_threshold,
      'max_keypoints': opt.max_keypoints
    },
    'superglue': {
      'weights': opt.superglue,
      'sinkhorn_iterations': opt.sinkhorn_iterations,
      'match_threshold': opt.match_threshold,
    }
  }

  def angle_to_rot(angle, image_shape):
    width, height = image_shape[:2]

    if angle == 90:
      code = cv2.ROTATE_90_CLOCKWISE
      func = lambda x: numpy.stack([x[:, 1], width - x[:, 0]], axis=1)
    elif angle == 180:
      code = cv2.ROTATE_180
      func = lambda x: numpy.stack([height - x[:, 0], width - x[:, 1]], axis=1)
    elif angle == 270:
      code = cv2.ROTATE_90_COUNTERCLOCKWISE
      func = lambda x: numpy.stack([height - x[:, 1], x[:, 0]], axis=1)
    else:
      print('error: unsupported rotation angle %d' % angle)
      exit(1)

    return code, func


  data_path = opt.data_path
  with open(data_path + '/meta_data.json', 'r') as f:
    data_config = json.load(f)

  file_names = data_config['meta']['file_names']
  if data_path != data_config['meta']['data_path']:
    print('warning: data path in json is different from the input data path')
  if len(file_names) < 2:
    print('Need at least two files to run matching')
    return

  matching = Matching(config).eval().to(device)
  keys = ['keypoints', 'scores', 'descriptors']

  for i in range(len(file_names) - 1):
    target_name = file_names[i]
    source_name = file_names[i+1]
    print('processing %s vs %s' % (target_name, source_name))

    target_image = cv2.imread('%s/%s/%s_lidar_color_gray.png' % (data_path, target_name, target_name), 0)
    if target_image is None:
      print('warning: failed to load lidar image for %s' % target_name)
      continue

    target_image_tensor = frame2tensor(target_image, device)
    last_data = matching.superpoint({'image': target_image_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = target_image_tensor

    source_image = cv2.imread('%s/%s/%s_lidar_color_gray.png' % (data_path, source_name, source_name), 0)
    if source_image is None:
      print('warning: failed to load lidar image for %s' % source_name)
      continue

    if opt.rotate_lidar:
      code, lidar_R_inv = angle_to_rot(opt.rotate_lidar, source_image.shape)
      source_image = cv2.rotate(source_image, code)

    source_image_tensor = frame2tensor(source_image, device)

    pred = matching({**last_data, 'image1': source_image_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    kpts0_ = kpts0
    kpts1_ = kpts1

    if opt.rotate_lidar:
      kpts1_ = lidar_R_inv(kpts1_)

    result = { 'kpts0': kpts0_.flatten().tolist(), 'kpts1': kpts1_.flatten().tolist(), 'matches': matches.flatten().tolist(), 'confidence': confidence.flatten().tolist() }
    with open('%s/%s_vs_%s_matches.json' % (data_path, target_name, source_name), 'w') as f:
      json.dump(result, f)

    # visualization
    target_canvas = cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)
    source_canvas = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
    source_canvas = cv2.resize(source_canvas, (target_image.shape[1], target_image.shape[0]))

    sx = float(target_image.shape[1]) / float(source_image.shape[1])
    sy = float(target_image.shape[0]) / float(source_image.shape[0])

    kpts1_vis = kpts1.copy()
    kpts1_vis[:, 0] = kpts1_vis[:, 0] * sx + target_image.shape[1]
    kpts1_vis[:, 1] = kpts1_vis[:, 1] * sy

    canvas = numpy.concatenate([target_canvas, source_canvas], axis=1)
    for kp in kpts0:
      cv2.circle(canvas, (int(kp[0]), int(kp[1])), 3, (255, 255, 255))
    for kp in kpts1_vis:
      cv2.circle(canvas, (int(kp[0]), int(kp[1])), 3, (255, 255, 255))

    try:
      cmap = matplotlib.colormaps.get_cmap('turbo')
    except AttributeError:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('turbo')
    if numpy.max(confidence) > 0:
      confidence_norm = confidence / numpy.max(confidence)
    else:
      confidence_norm = confidence

    for j, match in enumerate(matches):
      if match < 0:
        continue
      kp0 = kpts0[j]
      kp1 = kpts1_vis[match]

      color = tuple((numpy.array(cmap(confidence_norm[j])) * 255).astype(int).tolist())

      cv2.line(canvas, (int(kp0[0]), int(kp0[1])), (int(kp1[0]), int(kp1[1])), color)

    cv2.imwrite('%s/%s_vs_%s_superglue.png' % (data_path, target_name, source_name), canvas)

    if opt.show_keypoints:
      cv2.imshow('canvas', canvas)
      cv2.waitKey(0)


if __name__ == '__main__':
  main()
