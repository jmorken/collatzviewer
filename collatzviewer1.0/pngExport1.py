# coding: utf-8
# code from: https://gist.github.com/FlorianRhiem/dd3ae199da5ab5ff46d0
"""
Export images to PNG files.
Based on the ActiveState Recipe
    "Write a PNG image in native Python (Pythonrecipe)"
(http://code.activestate.com/recipes/577443-write-a-png-image-in-native-python/)
which Campbell Barton (http://code.activestate.com/recipes/users/4168177/)
posted in October '10 under MIT license.
Modified to use numpy and documented by Florian Rhiem.
"""

# Python 3 compatibility imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import struct
import zlib
import numpy as np


def to_png(image, flip_y=True):
    """
    Return a binary png image for a given image stored as three-dimensional
    numpy array.
    """
    height, width, _ = image.shape
    image.shape = (height, width*4)
    # Flip the image vertically
    if flip_y:
        image = image[::-1, :]
    # Add a leading 0 to each row
    padding = np.zeros((height, 1), dtype=np.uint8)
    padded_image = np.hstack([padding, image])
    raw_data = padded_image.tostring()

    def png_chunk(chunk_type, chunk_data):
        """
        This function returns a packed PNG chunk, given its type and data.
        """
        packed_data = b''
        # Add length of actual data
        packed_data += struct.pack(b'!I', len(chunk_data))
        # Add PNG tag
        packed_data += chunk_type
        # Add actual data
        packed_data += chunk_data
        # Add 32 bit cyclic redundancy check
        crc32 = 0xFFFFFFFF & zlib.crc32(packed_data[4:])
        packed_data += struct.pack('!I', crc32)
        return packed_data

    png_image = b''
    # Add the file header
    png_image += b'\x89PNG\r\n\x1a\n'
    # Add the PNG header chunk
    # the magic numbers here are:
    # 1. bit depth (bits per channel: 8)
    # 2. color type (RGBA: 6)
    # 3. compression method (DEFLATE: 0)
    # 4. filter method (adaptive filtering: 0 - no filter applied here)
    # 5. interlace method (No interlacing: 0)
    png_image += png_chunk(b'IHDR', struct.pack(b'!2I5B',
                                                width, height,
                                                8, 6, 0, 0, 0))
    # Add the DEFLATE compressed image data
    png_image += png_chunk(b'IDAT', zlib.compress(raw_data, 9))
    # Add the end chunk
    png_image += png_chunk(b'IEND', b'')
    return png_image


def write_png(image, filename):
    """
    Write an image stored as three-dimensional numpy array to a file.
    """
    with open(filename, 'wb') as out:
        out.write(to_png(image))

def main():
    """
    Create a simple image and write it to a PNG file as an example.
    """
    width = 640
    height = 480
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, width).reshape(1, width)
    image[:, :, 1] = np.linspace(0, 255, height).reshape(height, 1)
    image[:, :, 3] = 255
    write_png(image, 'example.png')

if __name__ == '__main__':
    main()