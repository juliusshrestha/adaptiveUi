#!/usr/bin/env python3
"""
Generate PNG icons for the Adaptive UI browser extension.
Run this script to create icon16.png, icon48.png, and icon128.png
"""

import struct
import zlib
from pathlib import Path


def create_png(width: int, height: int, pixels: list) -> bytes:
    """
    Create a PNG image from pixel data.

    Args:
        width: Image width
        height: Image height
        pixels: List of (r, g, b, a) tuples, row by row

    Returns:
        PNG file bytes
    """
    def write_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xffffffff
        return struct.pack('>I', len(data)) + chunk + struct.pack('>I', crc)

    # PNG signature
    png = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    png += write_chunk(b'IHDR', ihdr_data)

    # IDAT chunk (image data)
    raw_data = b''
    for y in range(height):
        raw_data += b'\x00'  # Filter type: None
        for x in range(width):
            idx = y * width + x
            if idx < len(pixels):
                r, g, b, a = pixels[idx]
                raw_data += bytes([r, g, b, a])
            else:
                raw_data += b'\x00\x00\x00\x00'

    compressed = zlib.compress(raw_data, 9)
    png += write_chunk(b'IDAT', compressed)

    # IEND chunk
    png += write_chunk(b'IEND', b'')

    return png


def draw_circle(pixels: list, width: int, cx: int, cy: int, r: int, color: tuple):
    """Draw a filled circle."""
    for y in range(max(0, cy - r), min(width, cy + r + 1)):
        for x in range(max(0, cx - r), min(width, cx + r + 1)):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist <= r:
                idx = y * width + x
                if idx < len(pixels):
                    # Anti-aliasing at edges
                    if dist > r - 1:
                        alpha = max(0, min(255, int(255 * (r - dist + 1))))
                        old = pixels[idx]
                        # Blend with existing pixel
                        blend_alpha = alpha / 255
                        new_r = max(0, min(255, int(color[0] * blend_alpha + old[0] * (1 - blend_alpha))))
                        new_g = max(0, min(255, int(color[1] * blend_alpha + old[1] * (1 - blend_alpha))))
                        new_b = max(0, min(255, int(color[2] * blend_alpha + old[2] * (1 - blend_alpha))))
                        new_a = max(0, min(255, max(old[3], alpha)))
                        pixels[idx] = (new_r, new_g, new_b, new_a)
                    else:
                        pixels[idx] = color


def create_icon(size: int) -> bytes:
    """Create an icon of the specified size."""
    # Initialize with transparent pixels
    pixels = [(0, 0, 0, 0)] * (size * size)

    center = size // 2

    # Main circle (green background)
    main_radius = int(size * 0.45)
    draw_circle(pixels, size, center, center, main_radius, (76, 175, 80, 255))

    # Eye (white circle)
    eye_y = int(center - size * 0.12)
    eye_radius = int(size * 0.18)
    draw_circle(pixels, size, center, eye_y, eye_radius, (255, 255, 255, 255))

    # Pupil (dark circle)
    pupil_radius = int(size * 0.08)
    draw_circle(pixels, size, center, eye_y, pupil_radius, (51, 51, 51, 255))

    # Smile (simplified as a small white arc at bottom)
    smile_y = int(center + size * 0.2)
    smile_radius = int(size * 0.05)
    for offset in range(-int(size * 0.2), int(size * 0.2) + 1, max(1, size // 16)):
        draw_circle(pixels, size, center + offset, smile_y, smile_radius, (255, 255, 255, 255))

    return create_png(size, size, pixels)


def main():
    script_dir = Path(__file__).parent

    sizes = [16, 48, 128]

    for size in sizes:
        png_data = create_icon(size)
        output_path = script_dir / f'icon{size}.png'
        output_path.write_bytes(png_data)
        print(f'Created {output_path}')

    print('Done! Icons created successfully.')


if __name__ == '__main__':
    main()
