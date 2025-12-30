#!/usr/bin/env python3
"""정다각별(star polygon) 이미지를 생성합니다.

간단하게 각 개수만 지정하면 자동으로 가장 보기 좋은 별을 생성합니다.
"""

import cv2
import numpy as np
import os


def generate_star_polygon(n, k=None, size=400, thickness=3, output_path=None):
    """정{n/k} 다각별을 생성하여 PNG로 저장합니다.
    
    Args:
        n: 꼭짓점 수 (5, 6, 7, 8, ..., 24, ...)
        k: 간격 (None이면 자동 선택. 1 <= k <= (n-1)//2)
        size: 이미지 크기 (size x size 픽셀)
        thickness: 선의 두께
        output_path: 저장 경로
    
    Returns:
        (이미지 numpy 배열, 저장된 파일 경로)
    """
    # k 자동 결정: 가장 보기 좋은 별을 만들기 위해
    if k is None:
        # n이 홀수면 (n-1)//2, 짝수면 n//2 - 1을 기본값으로
        k = (n - 1) // 2 if n % 2 == 1 else n // 2 - 1
        # k는 최소 2 이상
        k = max(2, k)
    
    if k < 1 or k > (n - 1) // 2:
        raise ValueError(f'Invalid k={k} for n={n}. Must be 1 <= k <= {(n-1)//2}')
    
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
    
    # 간격 k로 꼭짓점 연결 (별 모양)
    for i in range(n):
        p1 = np.array(vertices[i], dtype=np.int32)
        p2 = np.array(vertices[(i + k) % n], dtype=np.int32)
        cv2.line(img, tuple(p1), tuple(p2), (0, 0, 0), thickness)
    
    if output_path is None:
        output_path = f'star_{n}.png'
    
    cv2.imwrite(output_path, img)
    print(f'생성: {output_path} (star polygon {n} vertices, k={k})')
    return img, output_path


def generate_all_star_polygons(output_dir='.'):
    """다양한 각의 정다각별들을 생성합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 3부터 30까지 생성
    for n in range(5, 31):
        path = os.path.join(output_dir, f'star_{n}.png')
        generate_star_polygon(n, k=None, size=400, thickness=2, output_path=path)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='정다각별 생성')
    p.add_argument('--n', type=int, default=5, help='꼭짓점 수')
    p.add_argument('--k', type=int, default=None, help='간격 (None이면 자동)')
    p.add_argument('--size', type=int, default=400, help='이미지 크기')
    p.add_argument('--thickness', type=int, default=2, help='선 두께')
    p.add_argument('--output', default=None, help='출력 경로')
    p.add_argument('--all', action='store_true', help='5~30각별 모두 생성')
    p.add_argument('--range', type=str, default=None, help='범위 지정 (예: 5-30, 10-50)')
    args = p.parse_args()
    
    if args.all:
        generate_all_star_polygons()
    elif args.range:
        start, end = map(int, args.range.split('-'))
        for n in range(start, end+1):
            path = f'star_{n}.png'
            generate_star_polygon(n, k=None, size=400, thickness=2, output_path=path)
    else:
        generate_star_polygon(args.n, k=args.k, size=args.size, thickness=args.thickness, output_path=args.output)

