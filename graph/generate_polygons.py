#!/usr/bin/env python3
"""정다각형 이미지를 생성합니다."""

import cv2
import numpy as np
import os

def generate_regular_polygon(n, size=400, thickness=3, output_path=None):
    """정n각형을 생성하여 PNG로 저장합니다.
    
    Args:
        n: 변의 개수 (3=삼각형, 4=사각형, 5=오각형, ...)
        size: 이미지 크기 (size x size 픽셀)
        thickness: 선의 두께 (-1이면 채우기)
        output_path: 저장 경로
    
    Returns:
        (이미지 numpy 배열, 저장된 파일 경로)
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    center = (size // 2, size // 2)
    radius = size // 2 - 20
    
    # 정n각형의 꼭짓점 계산
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    vertices = []
    for angle in angles:
        x = center[0] + int(radius * np.cos(angle))
        y = center[1] + int(radius * np.sin(angle))
        vertices.append([x, y])
    
    vertices = np.array(vertices, dtype=np.int32)
    
    # 다각형 그리기
    cv2.polylines(img, [vertices], True, (0, 0, 0), thickness)
    
    if output_path is None:
        output_path = f'polygon_{n}.png'
    
    cv2.imwrite(output_path, img)
    print(f'생성: {output_path}')
    return img, output_path


def generate_all_polygons(sizes=(3, 4, 5, 6, 8, 10, 12, 15, 20), output_dir='.'):
    """여러 정다각형을 생성합니다."""
    os.makedirs(output_dir, exist_ok=True)
    for n in sizes:
        path = os.path.join(output_dir, f'polygon_{n}.png')
        generate_regular_polygon(n, size=400, thickness=2, output_path=path)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='정다각형 생성')
    p.add_argument('--n', type=int, default=3, help='변의 개수')
    p.add_argument('--size', type=int, default=400, help='이미지 크기')
    p.add_argument('--thickness', type=int, default=2, help='선 두께')
    p.add_argument('--output', default=None, help='출력 경로')
    p.add_argument('--all', action='store_true', help='3, 4, 5, 6, 8, 10, 12, 15, 20각형 생성')
    p.add_argument('--range', type=str, default=None, help='범위 지정 (예: 3-20, 5-30)')
    args = p.parse_args()
    
    if args.all:
        generate_all_polygons()
    elif args.range:
        start, end = map(int, args.range.split('-'))
        sizes = list(range(start, end+1))
        generate_all_polygons(sizes=tuple(sizes))
    else:
        generate_regular_polygon(args.n, size=args.size, thickness=args.thickness, output_path=args.output)
