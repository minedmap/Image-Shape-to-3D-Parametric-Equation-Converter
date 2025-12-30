#!/usr/bin/env python3
"""이미지에서 2D 곡선을 추출하고 3D 매개변수 방정식으로 변환합니다.
푸리에 급수를 사용하여 x(t), y(t), z(t) 형태의 수식을 생성합니다.

사용 예:
  python convert_image.py polygon.png --mode curve --method fourier --terms 10
  python convert_image.py polygon.png --mode parametric3d --terms 10
  python convert_image.py image.png --mode surface
"""
import argparse
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fit_curve(x, y, degree=5):
    # 데이터 정규화 - 수치 안정성 향상
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min + 1e-8) - 1
    
    # 낮은 차수로 적합
    degree = min(degree, 3)  # 최대 3차로 제한
    coeffs = np.polyfit(x_norm, y_norm, degree)
    p_norm = np.poly1d(coeffs)
    yhat_norm = p_norm(x_norm)
    rmse = np.sqrt(np.mean((y_norm - yhat_norm) ** 2))
    
    # 역정규화를 위한 변환 함수 반환
    def p_denorm(x_val):
        x_n = 2 * (x_val - x_min) / (x_max - x_min + 1e-8) - 1
        y_n = p_norm(x_n)
        return (y_n + 1) * (y_max - y_min) / 2 + y_min
    
    return p_denorm, rmse, coeffs


def fit_fourier(x, y, num_terms=10):
    """푸리에 급수로 곡선 적합 (안정성 개선)"""
    # 항 개수 제한 (오버피팅 방지)
    max_terms = min(num_terms, 100)  # 최대 100개로 제한
    if num_terms > 100:
        print(f'⚠️  경고: --terms가 너무 큽니다 (요청: {num_terms}). 100으로 자동 조정됩니다.')
        num_terms = max_terms
    
    # x를 [-π, π] 범위로 정규화
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if x_range < 1e-6:
        x_range = 1.0
    if y_range < 1e-6:
        y_range = 1.0
    
    x_norm = np.pi * (2 * (x - x_min) / x_range - 1)
    y_norm = 2 * (y - y_min) / y_range - 1
    
    # 푸리에 계수 계산 (고차 항 감쇠 줄임)
    a0 = np.mean(y_norm)
    an = np.zeros(num_terms)
    bn = np.zeros(num_terms)
    
    for n in range(1, num_terms + 1):
        # 고차 항에 대한 감쇠 줄임 (각진 부분 표현 개선)
        weight_decay = 1.0 / (1.0 + 0.05 * (n - 1))  # 0.1 -> 0.05
        an[n-1] = 2 * np.mean(y_norm * np.cos(n * x_norm)) * weight_decay
        bn[n-1] = 2 * np.mean(y_norm * np.sin(n * x_norm)) * weight_decay
    
    # 계수의 크기에 따른 적응형 clipping
    coeff_max = np.max(np.abs(np.concatenate([an, bn]))) if len(an) > 0 else 1.0
    coeff_threshold = max(4.0, coeff_max * 0.6)  # 3.0, 0.5 -> 4.0, 0.6
    an = np.clip(an, -coeff_threshold, coeff_threshold)
    bn = np.clip(bn, -coeff_threshold, coeff_threshold)
    
    # 푸리에 근사 함수
    def fourier_func(x_val):
        x_n = np.pi * (2 * (x_val - x_min) / x_range - 1)
        y_n = a0 / 2
        for n in range(1, num_terms + 1):
            weight_decay = 1.0 / (1.0 + 0.05 * (n - 1))
            y_n = y_n + (an[n-1] * np.cos(n * x_n) + bn[n-1] * np.sin(n * x_n)) * weight_decay
        
        # 범위 제한
        y_n = np.clip(y_n, -2.0, 2.0)
        result = (y_n + 1) * y_range / 2 + y_min
        result = np.clip(result, y_min * 0.3, y_max * 1.7)
        return result
    
    # RMSE 계산
    y_pred = a0 / 2
    for n in range(1, num_terms + 1):
        weight_decay = 1.0 / (1.0 + 0.05 * (n - 1))
        y_pred = y_pred + (an[n-1] * np.cos(n * x_norm) + bn[n-1] * np.sin(n * x_norm)) * weight_decay
    y_pred = np.clip(y_pred, -2.0, 2.0)
    rmse = np.sqrt(np.mean((y_norm - y_pred) ** 2))
    
    return fourier_func, rmse, (a0, an, bn)


def fit_polar_fourier(theta, r, num_terms=10, is_regular_polygon=False):
    """극좌표에서 r(θ)를 푸리에 급수로 적합 (정다각형/정다각별용)
    
    정다각형의 경우 n배 대칭성을 활용하여 더 정확한 표현을 합니다.
    """
    # 항 개수 제한 (오버피팅 방지)
    max_terms = min(num_terms, 100)  # 최대 100개로 제한
    if num_terms > 100:
        print(f'⚠️  경고: --terms가 너무 큽니다 (요청: {num_terms}). 100으로 자동 조정됩니다.')
        num_terms = max_terms
    
    # r 정규화 - 안정성 향상
    r_min, r_max = r.min(), r.max()
    r_range = r_max - r_min
    
    if r_range < 1e-6:
        r_range = 1.0
    
    r_norm = 2 * (r - r_min) / r_range - 1
    
    # 푸리에 계수 계산
    a0 = np.mean(r_norm)
    an = np.zeros(num_terms)
    bn = np.zeros(num_terms)
    
    for n in range(1, num_terms + 1):
        # 정다각형인 경우: n의 배수 주파수만 사용 (더 정확한 표현)
        if is_regular_polygon and n % 2 != 0:
            # 정다각형의 짝수 항만 중요 (홀수 항 감소)
            weight_decay = 0.3 / (1.0 + 0.1 * (n - 1))
        else:
            weight_decay = 1.0 / (1.0 + 0.05 * (n - 1))
        
        an[n-1] = 2 * np.mean(r_norm * np.cos(n * theta)) * weight_decay
        bn[n-1] = 2 * np.mean(r_norm * np.sin(n * theta)) * weight_decay
    
    # 계수의 크기에 따른 적응형 clipping
    coeff_max = np.max(np.abs(np.concatenate([an, bn]))) if len(an) > 0 else 1.0
    coeff_threshold = max(4.0, coeff_max * 0.6)
    an = np.clip(an, -coeff_threshold, coeff_threshold)
    bn = np.clip(bn, -coeff_threshold, coeff_threshold)
    
    # 푸리에 근사 함수
    def r_func(theta_val):
        r_n = a0 / 2
        for n in range(1, num_terms + 1):
            # 정다각형인 경우: n의 배수 주파수만 사용
            if is_regular_polygon and n % 2 != 0:
                weight_decay = 0.3 / (1.0 + 0.1 * (n - 1))
            else:
                weight_decay = 1.0 / (1.0 + 0.05 * (n - 1))
            r_n = r_n + (an[n-1] * np.cos(n * theta_val) + bn[n-1] * np.sin(n * theta_val)) * weight_decay
        
        # 범위 제한
        r_n = np.clip(r_n, -2.0, 2.0)
        result = (r_n + 1) * r_range / 2 + r_min
        result = np.clip(result, r_min * 0.3, r_max * 1.7)
        return result
    
    # RMSE 계산
    r_pred = a0 / 2
    for n in range(1, num_terms + 1):
        # 정다각형인 경우: n의 배수 주파수만 사용
        if is_regular_polygon and n % 2 != 0:
            weight_decay = 0.3 / (1.0 + 0.1 * (n - 1))
        else:
            weight_decay = 1.0 / (1.0 + 0.05 * (n - 1))
        r_pred = r_pred + (an[n-1] * np.cos(n * theta) + bn[n-1] * np.sin(n * theta)) * weight_decay
    r_pred = np.clip(r_pred, -2.0, 2.0)
    rmse = np.sqrt(np.mean((r_norm - r_pred) ** 2))
    
    return r_func, rmse, (a0, an, bn)


def contour_to_function(contour):
    pts = contour.reshape(-1, 2)
    pts = pts[np.argsort(pts[:, 0])]
    xs = pts[:, 0]
    ys = pts[:, 1]
    xr = np.round(xs).astype(int)
    uniq = {}
    for xi, yi in zip(xr, ys):
        uniq.setdefault(xi, []).append(yi)
    xk = np.array(sorted(uniq.keys()), dtype=float)
    yk = np.array([np.median(uniq[k]) for k in xk], dtype=float)
    return xk, yk


def detect_regular_polygon(r, theta, tolerance=0.08):
    """극좌표 곡선에서 정다각형 여부와 변의 개수를 감지합니다.
    
    정다각형은 r(theta)가 n배 대칭성을 가지므로,
    푸리에 스펙트럼에서 n의 배수 주파수만 강하게 나타납니다.
    """
    # 간단한 FFT로 주요 주파수 확인
    r_norm = (r - r.mean()) / (r.std() + 1e-6)
    fft_vals = np.abs(np.fft.fft(r_norm))
    
    # 첫 20개 주파수 확인 (3~20각형)
    for n_sides in range(20, 2, -1):
        # n_sides 주파수와 그 배수가 강해야 함
        peaks = []
        for k in range(1, 4):  # 1배, 2배, 3배
            freq_idx = k * n_sides % len(fft_vals)
            if freq_idx < len(fft_vals):
                peaks.append(fft_vals[freq_idx])
        
        # 다른 주파수는 약해야 함
        others = []
        for freq_idx in range(1, min(n_sides, len(fft_vals))):
            if freq_idx % n_sides != 0 and freq_idx < len(fft_vals):
                others.append(fft_vals[freq_idx])
        
        if len(peaks) > 0 and len(others) > 0:
            avg_peak = np.mean(peaks)
            avg_other = np.mean(others)
            
            if avg_other > 0 and avg_peak / (avg_other + 1e-6) > 2.0:  # 2배 이상 차이
                return n_sides, True
    
    return None, False


def get_regular_polygon_radius_function(n, scale=1.0):
    """Raskolnikov 공식을 사용한 정n각형의 정확한 r(θ) 함수.
    
    r(θ) = cos(π/n) / cos(2π(nθ) mod 1 / n - π/n)
    
    이는 원에 내접하는 완벽한 정다각형을 생성합니다.
    """
    def radius_func(theta_val):
        # theta를 [0, 2π) 범위로 정규화
        theta_norm = theta_val % (2 * np.pi)
        
        # 극좌표 각도를 n각형의 로컬 좌표로 변환
        # 한 변이 차지하는 각도는 2π/n
        angle_per_side = 2 * np.pi / n
        
        # 현재 각도를 한 변의 범위 내로 정규화
        local_angle = (theta_norm % angle_per_side) - np.pi / n
        
        # Raskolnikov 공식 적용
        numerator = np.cos(np.pi / n)
        denominator = np.cos(local_angle)
        
        # 분모가 0 근처인 경우 처리
        denominator = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)
        
        r_value = scale * numerator / denominator
        
        return r_value
    
    return radius_func


def polygon_r_from_vertices(vertices, center, thetas):
    """주어진 꼭짓점(vertices)과 중심(center)에 대해 각 theta에 대한 반지름(r)을 계산합니다.

    각 theta에 대해 중심에서 그 방향으로 뻗은 레이와 다각형 변(segment)의 교차를 계산하여
    해당 방향으로의 반지름(거리)을 반환합니다. 실패한 경우 NaN을 반환합니다.
    """
    cx, cy = center
    verts = np.asarray(vertices, dtype=float).reshape(-1, 2)
    m = len(verts)
    segs = [(verts[i], verts[(i + 1) % m]) for i in range(m)]

    r_vals = np.full_like(thetas, np.nan, dtype=float)
    for i, th in enumerate(thetas):
        dx = np.cos(th)
        dy = np.sin(th)
        t_candidates = []
        for p1, p2 in segs:
            x1, y1 = p1
            x2, y2 = p2
            sx = x2 - x1
            sy = y2 - y1
            # 선형 시스템: [dx, -sx; dy, -sy] [t; u] = [x1-cx; y1-cy]
            A00 = dx; A01 = -sx
            A10 = dy; A11 = -sy
            det = A00 * A11 - A01 * A10
            if abs(det) < 1e-9:
                continue
            bx = x1 - cx; by = y1 - cy
            t = (bx * A11 - A01 * by) / det
            u = (A00 * by - bx * A10) / det
            # t>0 (앞방향), u in [0,1] (세그먼트 내)
            if t > 1e-6 and u >= -1e-6 and u <= 1 + 1e-6:
                t_candidates.append(t)
        if len(t_candidates) > 0:
            r_vals[i] = min(t_candidates)
    return r_vals


def simplify_vertices(verts, min_dist=5.0):
    """연속된 꼭짓점들 중 서로 너무 가까운 항목을 병합하여 중복 꼭짓점 제거

    verts: (N,2) 배열
    min_dist: 픽셀 단위 최소 거리
    """
    if len(verts) == 0:
        return verts
    out = [verts[0].astype(float)]
    for v in verts[1:]:
        if np.linalg.norm(v - out[-1]) > min_dist:
            out.append(v.astype(float))
    # 마지막과 처음이 너무 가깝다면 병합
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) < min_dist:
        avg = (out[0] + out[-1]) / 2.0
        out[0] = avg
        out.pop()
    return np.array(out)


def contour_to_polar(contour, force_polygon=False, approx_eps=None):
    """폐곡선을 극좌표 (r, theta)로 변환하고, 정다각형이면 정확한 공식 적용"""
    pts = contour.reshape(-1, 2)
    # 무게중심 계산
    cx, cy = pts.mean(axis=0)
    # 극좌표로 변환
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    
    # theta 정렬
    sorted_idx = np.argsort(theta)
    theta_sorted = theta[sorted_idx]
    r_sorted = r[sorted_idx]
    
    # theta를 균등하게 다시 샘플링
    theta_uniform = np.linspace(theta_sorted[0], theta_sorted[-1] + 2*np.pi, len(r_sorted))
    r_interp = np.interp(theta_uniform % (2*np.pi), theta_sorted % (2*np.pi), r_sorted, period=2*np.pi)
    
    # 우선 contour 기반으로 꼭짓점 근사 시도 (더 신뢰도 높은 정다각형 감지)
    try:
        arc_len = cv2.arcLength(contour, True)
        best_candidate = None
        best_score = float('inf')
        frac_list = [0.005, 0.01, 0.02, 0.04]
        if approx_eps is not None:
            frac_list = [approx_eps]
        # 여러 eps 시도
        for frac in frac_list:
            eps = max(0.5, frac * arc_len)
            approx = cv2.approxPolyDP(contour, eps, True)
            n_approx = len(approx)
            if not (3 <= n_approx <= 40):
                continue
            area_cont = abs(cv2.contourArea(contour))
            area_approx = abs(cv2.contourArea(approx))
            if area_cont < 1e-6:
                continue
            if abs(area_cont - area_approx) / area_cont > 0.6:
                # 너무 차이나면 건너뜀
                continue
            verts = approx.reshape(-1, 2)
            # 중복/근접 꼭짓점 병합하여 잘못된 이중화(n*2) 방지
            min_dist = max(3.0, 0.01 * arc_len)
            verts_s = simplify_vertices(verts, min_dist=min_dist)
            n_simplified = len(verts_s)
            if n_simplified < 3:
                continue
            # 꼭짓점 각도 분포 검사: 각 간격이 고르게 분포하는지 확인
            # 꼭짓점 중심은 단순화된 꼭짓점의 평균으로 계산하여 각 분포 판단을 안정화
            center_vs = verts_s.mean(axis=0)
            v_angles = np.arctan2(verts_s[:, 1] - center_vs[1], verts_s[:, 0] - center_vs[0])
            v_angles = np.sort(v_angles)
            v_diffs = np.diff(np.concatenate([v_angles, [v_angles[0] + 2 * np.pi]]))
            mean_diff = np.mean(v_diffs)
            rel_std = np.std(v_diffs) / (mean_diff + 1e-9)
            expected = 2 * np.pi / float(n_simplified)
            # 각 간격 허용 오차를 약간 엄격하게 하여 이중화된 꼭짓점(n*2) 후보를 걸러냄
            if rel_std > 0.28 or abs(mean_diff - expected) / (expected + 1e-9) > 0.28:
                continue

            r_regular = polygon_r_from_vertices(verts_s, (cx, cy), theta_uniform)
            # n_approx 대신 n_simplified 사용하여 후보 평가
            n_approx = n_simplified
            if np.all(np.isnan(r_regular)):
                continue
            nan_frac = np.mean(~np.isfinite(r_regular))
            if nan_frac > 0.5:
                continue
            # r 기반 RMSE 평가: 원래 r_interp와 비교
            diff = r_interp - r_regular
            score = np.nanmean(np.abs(diff))
            if score < best_score:
                best_score = score
                best_candidate = (n_approx, r_regular, approx)
            # 강제 모드이면 첫 성공 후보 반환
            if force_polygon and best_candidate is not None:
                n_approx, r_regular, approx = best_candidate
                print(f'정{n_approx}각형 감지됨 - 윤곽선 근사 기반 적용 (forced, score={best_score:.4f})')
                return theta_uniform, r_regular, (cx, cy), int(n_approx)
        if best_candidate is not None:
            n_approx, r_regular, approx = best_candidate
            print(f'정{n_approx}각형 감지됨 - 윤곽선 근사 기반 적용 (score={best_score:.4f})')
            return theta_uniform, r_regular, (cx, cy), int(n_approx)
    except Exception:
        pass

    # 폴백: FFT 기반 또는 Raskolnikov 공식 사용
    n_sides, is_regular = detect_regular_polygon(r_interp, theta_uniform)
    if is_regular and n_sides is not None:
        # 정다각형인 경우: 후보 n 및 (짝수이면) n/2 후보를 시험
        candidate_ns = [n_sides]
        if n_sides % 2 == 0 and (n_sides // 2) >= 3:
            candidate_ns.append(n_sides // 2)

        # 입력 r를 정규화하여 스케일 편향을 최소화한 거리 기준 사용
        r_interp_mean = np.nanmean(r_interp)
        r_interp_std = np.nanstd(r_interp) + 1e-9
        r_interp_norm = (r_interp - r_interp_mean) / r_interp_std

        best_n = None
        best_score = float('inf')
        best_r = None
        for cand in candidate_ns:
            # 단위 스케일의 Raskolnikov 함수 생성 후 정규화하여 비교
            polygon_radius_func = get_regular_polygon_radius_function(cand, 1.0)
            r_cand = polygon_radius_func(theta_uniform)
            r_cand_mean = np.nanmean(r_cand)
            r_cand_std = np.nanstd(r_cand) + 1e-9
            r_cand_norm = (r_cand - r_cand_mean) / r_cand_std

            # 정규화된 RMSE 계산 (스케일 불변)
            score = np.sqrt(np.nanmean((r_interp_norm - r_cand_norm) ** 2))
            # 복잡도 패널티: 항목 수가 더 많은 다각형을 약간 벌점
            penalty = 0.005 * float(cand)
            score += penalty

            if score < best_score:
                best_score = score
                best_n = cand
                # 스케일을 원래 데이터에 맞추기 (평균에 기반)
                scale_factor = (r_interp_mean / (r_cand_mean + 1e-9))
                best_r = r_cand * scale_factor

        if best_n is not None:
            print(f'정{best_n}각형 감지됨 - Raskolnikov 공식 적용 (폴백, selected from {candidate_ns}, score={best_score:.4f})')
            return theta_uniform, best_r, (cx, cy), int(best_n)

    # 일반 곡선
    return theta_uniform, r_interp, (cx, cy), None


def process_curve(img, degree=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1)
    
    # 여러 threshold 값으로 시도
    thresholds = [(30, 100), (50, 150), (20, 80), (10, 50)]
    contours = None
    edges = None
    
    for low, high in thresholds:
        edges = cv2.Canny(blurred, low, high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours and len(contours) > 0:
            break
    
    if not contours:
        # 이진화 시도
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edges = binary
    
    if not contours:
        raise RuntimeError('이미지에서 윤곽을 찾을 수 없습니다. 더 선명한 이미지를 사용하세요.')
    
    contour = max(contours, key=lambda c: c.shape[0])
    x, y = contour_to_function(contour)
    y = img.shape[0] - y
    p, rmse, coeffs = fit_curve(x, y, degree)
    return x, y, p, rmse, coeffs, edges, contour


def process_surface(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(float) / 255.0
    return X, Y, Z


def latex_poly(x, y, degree=3):
    """간단한 다항식 수식을 LaTeX 형식으로 생성"""
    # 정규화된 좌표에서 다항식 계수 추출
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min + 1e-8) - 1
    
    coeffs = np.polyfit(x_norm, y_norm, min(degree, 3))
    
    terms = []
    for i, c in enumerate(coeffs):
        power = len(coeffs) - 1 - i
        if abs(c) < 1e-12:
            continue
        coeff_str = f'{c:.4f}'
        if power == 0:
            terms.append(f'{coeff_str}')
        elif power == 1:
            terms.append(f'{coeff_str}x')
        else:
            terms.append(f'{coeff_str}x^{power}')
    
    if not terms:
        return '$y=0$'
    body = ' + '.join(terms).replace('+ -', '- ')
    return f'$y = {body}$ (normalized coordinates)'


def desmos_poly(x, y, degree=3):
    """간단한 다항식 수식을 Desmos 형식으로 생성"""
    # 정규화된 좌표에서 다항식 계수 추출
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min + 1e-8) - 1
    
    coeffs = np.polyfit(x_norm, y_norm, min(degree, 3))
    
    terms = []
    for i, c in enumerate(coeffs):
        power = len(coeffs) - 1 - i
        if abs(c) < 1e-12:
            continue
        coeff_str = f'{c:.4f}'
        if power == 0:
            terms.append(coeff_str)
        elif power == 1:
            terms.append(f'{coeff_str}*x')
        else:
            terms.append(f'{coeff_str}*x^{power}')
    
    if not terms:
        return 'y = 0'
    body = ' + '.join(terms).replace('+ -', '- ')
    return f'y = {body}'


def latex_fourier(a0, an, bn):
    """푸리에 급수 수식을 LaTeX 형식으로 생성"""
    terms = []
    terms.append(f'{a0/2:.4f}')
    
    for n in range(len(an)):
        n_val = n + 1
        if abs(an[n]) > 1e-4:
            terms.append(f'{an[n]:.4f}\\cos({n_val}x)')
        if abs(bn[n]) > 1e-4:
            terms.append(f'{bn[n]:.4f}\\sin({n_val}x)')
    
    body = ' + '.join(terms).replace('+ -', '- ')
    return f'$y = {body}$ (normalized x ∈ [-π, π])'


def desmos_fourier(a0, an, bn):
    """푸리에 급수 수식을 Desmos 형식으로 생성"""
    terms = []
    terms.append(f'{a0/2:.4f}')
    
    for n in range(len(an)):
        n_val = n + 1
        if abs(an[n]) > 1e-4:
            sign = '+' if an[n] >= 0 else '-'
            terms.append(f'{sign} {abs(an[n]):.4f}*cos({n_val}*x)')
        if abs(bn[n]) > 1e-4:
            sign = '+' if bn[n] >= 0 else '-'
            terms.append(f'{sign} {abs(bn[n]):.4f}*sin({n_val}*x)')
    
    body = ' '.join(terms).replace('+ -', '- ').replace('- ', '- ')
    return f'y = {body}'


def latex_parametric3d(a0_r, an_r, bn_r, a0_z=None, an_z=None, bn_z=None):
    """3D 매개변수 방정식을 LaTeX 형식으로 생성"""
    # r(t) 항 생성
    r_terms = []
    r_terms.append(f'{a0_r/2:.4f}')
    
    for n in range(len(an_r)):
        n_val = n + 1
        if abs(an_r[n]) > 1e-4:
            r_terms.append(f'{an_r[n]:.4f}\\cos({n_val}t)')
        if abs(bn_r[n]) > 1e-4:
            r_terms.append(f'{bn_r[n]:.4f}\\sin({n_val}t)')
    
    r_body = ' + '.join(r_terms).replace('+ -', '- ')
    
    latex = f'''\\begin{{cases}}
x(t) = ({r_body}) \\cos(t) \\\\
y(t) = ({r_body}) \\sin(t) \\\\
z(t) = '''
    
    # z(t) 항 생성 (고도)
    if a0_z is not None:
        z_terms = []
        z_terms.append(f'{a0_z/2:.4f}')
        
        for n in range(len(an_z)):
            n_val = n + 1
            if abs(an_z[n]) > 1e-4:
                z_terms.append(f'{an_z[n]:.4f}\\cos({n_val}t)')
            if abs(bn_z[n]) > 1e-4:
                z_terms.append(f'{bn_z[n]:.4f}\\sin({n_val}t)')
        
        z_body = ' + '.join(z_terms).replace('+ -', '- ')
        latex += f'{z_body}'
    else:
        latex += '0'
    
    latex += '\\end{cases}$ (t ∈ [0, 2π])'
    return '$' + latex


def desmos_parametric3d(a0_r, an_r, bn_r, a0_z=None, an_z=None, bn_z=None):
    """3D 매개변수 방정식을 Desmos 형식으로 생성"""
    # r(t) 항 생성
    r_terms = []
    r_terms.append(f'{a0_r/2:.4f}')
    
    for n in range(len(an_r)):
        n_val = n + 1
        if abs(an_r[n]) > 1e-4:
            sign = '+' if an_r[n] >= 0 else '-'
            r_terms.append(f'{sign} {abs(an_r[n]):.4f}*cos({n_val}*t)')
        if abs(bn_r[n]) > 1e-4:
            sign = '+' if bn_r[n] >= 0 else '-'
            r_terms.append(f'{sign} {abs(bn_r[n]):.4f}*sin({n_val}*t)')
    
    r_body = ' '.join(r_terms).replace('+ -', '- ')
    
    desmos_str = f'''r(t) = {r_body}
x(t) = r(t)*cos(t)
y(t) = r(t)*sin(t)
z(t) = '''
    
    # z(t) 항 생성
    if a0_z is not None:
        z_terms = []
        z_terms.append(f'{a0_z/2:.4f}')
        
        for n in range(len(an_z)):
            n_val = n + 1
            if abs(an_z[n]) > 1e-4:
                sign = '+' if an_z[n] >= 0 else '-'
                z_terms.append(f'{sign} {abs(an_z[n]):.4f}*cos({n_val}*t)')
            if abs(bn_z[n]) > 1e-4:
                sign = '+' if bn_z[n] >= 0 else '-'
                z_terms.append(f'{sign} {abs(bn_z[n]):.4f}*sin({n_val}*t)')
        
        z_body = ' '.join(z_terms).replace('+ -', '- ')
        desmos_str += z_body
    else:
        desmos_str += '0'
    
    return desmos_str


def main():
    parser = argparse.ArgumentParser(description='이미지를 함수/그래프로 변환')
    parser.add_argument('image', help='이미지 경로')
    parser.add_argument('--mode', choices=['curve', 'surface', 'parametric3d'], default='parametric3d', 
                        help='변환 모드')
    parser.add_argument('--method', choices=['polynomial', 'fourier'], default='fourier', 
                        help='곡선 적합 방법')
    parser.add_argument('--degree', type=int, default=5, help='다항식 차수')
    parser.add_argument('--terms', type=int, default=10, help='푸리에 항의 개수')
    parser.add_argument('--prefix', default='output')
    parser.add_argument('--force-polygon', action='store_true', help='윤곽선 근사 기반으로 정다각형 강제 처리')
    parser.add_argument('--approx-eps', type=float, default=None, help='approxPolyDP epsilon 비율(예: 0.02)')
    args = parser.parse_args()
    # PNG 등에서 alpha 채널이 있을 수 있으므로 투명 배경을 흰색으로 합성
    img_raw = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise SystemExit('이미지를 열 수 없습니다: ' + args.image)
    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
        b, g, r, a = cv2.split(img_raw)
        alpha = a.astype(float) / 255.0
        # 배경을 흰색으로 합성
        background = np.ones_like(img_raw[:, :, :3], dtype=np.uint8) * 255
        fg = cv2.merge([b, g, r]).astype(float)
        bg = background.astype(float)
        comp = (fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])).astype(np.uint8)
        img = comp
    else:
        # 컬러 이미지(RGB/BGR)
        if img_raw.ndim == 2:
            img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
        else:
            img = img_raw[:, :, :3]
    
    if args.mode == 'parametric3d':
        # 정다각형/정다각별 -> 3D 매개변수 방정식
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 1)
        
        # 윤곽선 추출
        thresholds = [(30, 100), (50, 150), (20, 80), (10, 50)]
        contours = None
        
        for low, high in thresholds:
            edges = cv2.Canny(blurred, low, high)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours and len(contours) > 0:
                break
        
        if not contours:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            raise RuntimeError('윤곽선을 찾을 수 없습니다.')
        
        contour = max(contours, key=lambda c: c.shape[0])
        result = contour_to_polar(contour, force_polygon=args.force_polygon, approx_eps=args.approx_eps)
        
        # 반환값 처리 (정다각형 감지 포함)
        if len(result) == 4:
            theta, r, center, n_sides = result
        else:
            theta, r, center = result
            n_sides = None
        
        # r(θ) 푸리에 적합
        # 정다각형인 경우 더 적은 항으로도 정확하게 표현 가능
        terms_to_use = min(args.terms, 50) if n_sides else args.terms
        r_func, rmse_r, (a0_r, an_r, bn_r) = fit_polar_fourier(theta, r, num_terms=terms_to_use, is_regular_polygon=bool(n_sides))
        
        # z(t) 고도 함수 (진폭이 작은 푸리에 급수)
        z_data = np.sin(2 * theta) * 0.3  # 간단한 고도 변화
        z_func, rmse_z, (a0_z, an_z, bn_z) = fit_polar_fourier(theta, z_data, num_terms=terms_to_use, is_regular_polygon=False)
        
        # 매개변수 t ∈ [0, 2π]로 3D 곡선 계산
        t_plot = np.linspace(0, 2*np.pi, 2000)
        r_t = r_func(t_plot)
        x_t = r_t * np.cos(t_plot)
        y_t = r_t * np.sin(t_plot)
        z_t = z_func(t_plot)
        
        # 안정성 검사: NaN 또는 inf 제거
        valid_idx = np.isfinite(r_t) & np.isfinite(x_t) & np.isfinite(y_t) & np.isfinite(z_t)
        t_plot = t_plot[valid_idx]
        x_t = x_t[valid_idx]
        y_t = y_t[valid_idx]
        z_t = z_t[valid_idx]
        
        # 값의 범위 정규화
        if len(x_t) > 0:
            x_t = (x_t - x_t.mean()) / (x_t.std() + 1e-8) * 50
            y_t = (y_t - y_t.mean()) / (y_t.std() + 1e-8) * 50
            z_t = (z_t - z_t.mean()) / (z_t.std() + 1e-8) * 20
        
        # 3D 시각화 (여러 각도)
        fig = plt.figure(figsize=(16, 5))
        
        # 첫 번째 뷰: 위에서 아래로
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(x_t, y_t, z_t, 'b-', linewidth=2, label='Parametric curve')
        ax1.scatter([0], [0], [0], color='r', s=100, label='origin')
        ax1.set_xlabel('x(t)')
        ax1.set_ylabel('y(t)')
        ax1.set_zlabel('z(t)')
        ax1.set_title('3D Parametric: (x(t), y(t), z(t))')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 두 번째 뷰: 다른 각도
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot(x_t, y_t, z_t, 'g-', linewidth=2)
        ax2.scatter([0], [0], [0], color='r', s=100)
        ax2.view_init(elev=20, azim=45)
        ax2.set_xlabel('x(t)')
        ax2.set_ylabel('y(t)')
        ax2.set_zlabel('z(t)')
        ax2.set_title('View: elev=20°, azim=45°')
        ax2.grid(True, alpha=0.3)
        
        # 세 번째 뷰: 위에서 본 모양
        ax3 = fig.add_subplot(133)
        ax3.plot(x_t, y_t, 'b-', linewidth=2, label='top view')
        ax3.scatter([0], [0], color='r', s=100, label='origin')
        ax3.set_xlabel('x(t)')
        ax3.set_ylabel('y(t)')
        ax3.set_title('2D Top View (xy-plane)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        out_png = f'{args.prefix}_parametric3d.png'
        fig.savefig(out_png, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        # 추가: 각 매개변수별 함수 그래프
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(t_plot, x_t, 'r-', linewidth=2)
        axes[0].set_xlabel('t (parameter)')
        axes[0].set_ylabel('x(t)')
        axes[0].set_title('x(t) = r(t)·cos(t)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t_plot, y_t, 'g-', linewidth=2)
        axes[1].set_xlabel('t (parameter)')
        axes[1].set_ylabel('y(t)')
        axes[1].set_title('y(t) = r(t)·sin(t)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(t_plot, z_t, 'b-', linewidth=2)
        axes[2].set_xlabel('t (parameter)')
        axes[2].set_ylabel('z(t)')
        axes[2].set_title('z(t) = elevation function')
        axes[2].grid(True, alpha=0.3)
        
        out_func_png = f'{args.prefix}_parametric_functions.png'
        fig2.savefig(out_func_png, bbox_inches='tight', dpi=150)
        plt.close(fig2)
        
        # 수식 생성
        latex = latex_parametric3d(a0_r, an_r, bn_r, a0_z, an_z, bn_z)
        desmos = desmos_parametric3d(a0_r, an_r, bn_r, a0_z, an_z, bn_z)
        
        print('='*60)
        print('3D 매개변수 방정식 (Parametric 3D Curve)')
        print('='*60)
        print('\n【 LaTeX 형식 】')
        print(latex)
        print('\n【 Desmos 3D 형식 】')
        print(desmos)
        print(f'\n【 오차 지표 】')
        print(f'RMSE (r): {rmse_r:.6f}')
        print(f'RMSE (z): {rmse_z:.6f}')
        print(f'\n【 저장된 파일 】')
        print(f'3D 시각화: {out_png}')
        print(f'함수 그래프: {out_func_png}')
        print('='*60)
    
    elif args.mode == 'curve':
        x, y, p, rmse, coeffs, edges, contour = process_curve(img, degree=args.degree)
        xp = np.linspace(x.min(), x.max(), 800)
        
        # 선택한 방법으로 적합
        if args.method == 'fourier':
            fit_func, rmse, fit_coeffs = fit_fourier(x, y, num_terms=args.terms)
            yp = fit_func(xp)
            latex = latex_fourier(fit_coeffs[0], fit_coeffs[1], fit_coeffs[2])
            desmos = desmos_fourier(fit_coeffs[0], fit_coeffs[1], fit_coeffs[2])
            method_label = f'Fourier ({args.terms} terms)'
        else:
            fit_func, rmse, fit_coeffs = fit_curve(x, y, degree=args.degree)
            yp = fit_func(xp)
            latex = latex_poly(x, y, degree=min(args.degree, 3))
            desmos = desmos_poly(x, y, degree=min(args.degree, 3))
            method_label = f'Polynomial (degree {args.degree})'
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(x, y, s=10, label='extracted points')
        ax.plot(xp, yp, 'r-', label=method_label)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels, origin bottom)')
        ax.legend()
        out_png = f'{args.prefix}_curve.png'
        fig.savefig(out_png, bbox_inches='tight', dpi=150)
        print('LaTeX:', latex)
        print('Desmos:', desmos)
        print('RMSE:', rmse)
        print('그래프 저장:', out_png)
    
    else:  # surface
        X, Y, Z = process_surface(img)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax, shrink=0.6)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_zlabel('normalized intensity')
        out_png = f'{args.prefix}_surface.png'
        fig.savefig(out_png, bbox_inches='tight', dpi=150)
        print('3D surface 이미지의 밝도(intensity)를 z로 사용했습니다.')
        print('그래프 저장:', out_png)


if __name__ == '__main__':
    main()
