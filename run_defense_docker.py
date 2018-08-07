"""Tool which runs all attacks against all defenses and computes results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import os
import subprocess
import numpy as np
from PIL import Image


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description='Tool to run defense in docker.')
  parser.add_argument('--defense_dir', required=True,
                      help='Location of defense.')
  parser.add_argument('--input_dir', required=True,
                      help='Location of input.')
  parser.add_argument('--output_dir', required=True,
                      help='Location of output')
  parser.add_argument('--gpu', dest='use_gpu', action='store_true')
  parser.add_argument('--nogpu', dest='use_gpu', action='store_false')
  parser.set_defaults(use_gpu=True)
  return parser.parse_args()

def read_metadata(dirname, use_gpu):
  """Read metadata in defense_dir.

  Args:
    dirname: directory of defense.
    use_gpu: whether submissions should use GPU. This argument is
      used to pick proper Docker container for each submission and create
      instance of Attack or Defense class.

  Returns:
    (container, entry_point)
  """
  try:
    if not os.path.isdir(dirname):
      pass
    elif not os.path.exists(os.path.join(dirname, 'metadata.json')):
      pass
    else:
      with open(os.path.join(dirname, 'metadata.json')) as f:
        metadata = json.load(f)
      if use_gpu and ('container_gpu' in metadata):
        container = metadata['container_gpu']
      else:
        container = metadata['container']
      entry_point = metadata['entry_point']
  except (IOError, KeyError, ValueError):
    print('Failed to read metadata from defense directory ', dirname)
  return (container, entry_point)

def main():
  args = parse_args()
  defense_dir = args.defense_dir
  input_dir = args.input_dir
  output_dir = args.output_dir
  use_gpu = args.use_gpu
  
  container, entry_point = read_metadata(defense_dir, use_gpu)

  if use_gpu:
    docker_binary = 'nvidia-docker'
  else:
    docker_binary = 'docker'

  
  print('Running defense in docker...')
  cmd = [docker_binary, 'run',
         '-v', '{0}:/input'.format(input_dir),
         '-v', '{0}:/output'.format(output_dir),
         '-v', '{0}:/defense'.format(defense_dir),
         '-w', '/defense',
         container,
         './' + entry_point,
         '/input',
         '/output']
  print(' '.join(cmd))
  subprocess.call(cmd)


if __name__ == '__main__':
  main()
