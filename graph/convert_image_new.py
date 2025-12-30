#!/usr/bin/env python3
"""이미지에서 윤곽을 추출하고 3D 매개함수(x(t), y(t), z(t))로 변환합니다.

개요:
- 알파 합성, 윤곽 추출
- approxPolyDP 후보들을 평가하여 꼭짓점 기반 다각형 검사
- FFT 후보를 n과 n/2로 비교(짝수 오검출 방지)
- 극좌표 r(θ)에 대해 푸리에 적합 후 LaTeX/Desmos 출력

사용 예:
  python convert_image_new.py triangle.png --mode parametric3d --force-polygon --approx-eps 0.02
"""

import argparse
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ------------------------ 유틸리티 ------------------------

def load_image(path):
    img_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(path)
    # 알파가 있으면 흰색 배경과 합성
    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
        b, g, r, a = cv2.split(img_raw)
        alpha = (a.astype('float32') / 255.0)[..., None]
        fg = cv2.merge([b, g, r]).astype('float32')
        bg = np.ones_like(fg, dtype='float32') * 255.0
        comp = (fg * alpha + bg * (1 - alpha)).astype('uint8')
        return comp
    if img_raw.ndim == 2:
        return cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
    return img_raw[:, :, :3]


def extract_largest_contour(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (7, 7), 1)
    thresholds = [(50, 150), (30, 100), (20, 80), (10, 50)]
    contours = []
    for low, high in thresholds:
        edges = cv2.Canny(blurred, low, high)
        res = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = res[0] if len(res) == 2 else res[1]
        if cnts and len(cnts) > 0:
            contours = cnts
            break
    if not contours:
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        res = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = res[0] if len(res) == 2 else res[1]
        contours = cnts
    if not contours:
        return None
    contour = max(contours, key=lambda c: cv2.contourArea(c) if c is not None else 0)
    return contour


def simplify_vertices(verts, min_dist=6.0):
    if len(verts) == 0:
        return verts
    out = [np.asarray(verts[0], dtype=float)]
    for v in verts[1:]:
        v = np.asarray(v, dtype=float)
        if np.linalg.norm(v - out[-1]) > min_dist:
            out.append(v)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) < min_dist:
        avg = (out[0] + out[-1]) / 2.0
        out[0] = avg
        out.pop()
    return np.array(out)


def polygon_r_from_vertices(vertices, center, thetas):
    cx, cy = center
    verts = np.asarray(vertices, dtype=float)
    if len(verts.shape) != 2 or verts.shape[0] < 3:
        return np.full_like(thetas, np.nan, dtype=float)
    segs = [(verts[i], verts[(i + 1) % len(verts)]) for i in range(len(verts))]
    r_vals = np.full_like(thetas, np.nan, dtype=float)
    for i, th in enumerate(thetas):
        dx = np.cos(th)
        dy = np.sin(th)
        candidates = []
        for p1, p2 in segs:
            x1, y1 = p1; x2, y2 = p2
            sx, sy = x2 - x1, y2 - y1
            A00, A01 = dx, -sx
            A10, A11 = dy, -sy
            det = A00 * A11 - A01 * A10
            if abs(det) < 1e-9:
                continue
            bx, by = x1 - cx, y1 - cy
            t = (bx * A11 - A01 * by) / det
            u = (A00 * by - bx * A10) / det
            if t > 1e-6 and u >= -1e-6 and u <= 1 + 1e-6:
                candidates.append(t)
        if candidates:
            r_vals[i] = min(candidates)
    return r_vals


def evaluate_polygon_candidate(verts, center, theta_uniform, r_interp):
    r_cand = polygon_r_from_vertices(verts, center, theta_uniform)
    if np.all(np.isnan(r_cand)):
        return np.inf, None
    mu1, s1 = np.nanmean(r_interp), np.nanstd(r_interp) + 1e-9
    mu2, s2 = np.nanmean(r_cand), np.nanstd(r_cand) + 1e-9
    r1 = (r_interp - mu1) / s1
    r2 = (r_cand - mu2) / s2
    score = np.sqrt(np.nanmean((r1 - r2) ** 2))
    r_scaled = (r_cand - mu2) / s2 * s1 + mu1
    return score, r_scaled


def contour_to_polar(contour, force_polygon=False, approx_eps=None):
    pts = contour.reshape(-1, 2)
    cx, cy = pts.mean(axis=0)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    idx = np.argsort(theta)
    theta_s = theta[idx]
    r_s = r[idx]
    theta_uniform = np.linspace(0, 2 * np.pi, len(r_s), endpoint=False)
    r_interp = np.interp(theta_uniform, np.mod(theta_s, 2*np.pi), r_s, period=2*np.pi)

    try:
        arc = cv2.arcLength(contour, True)
        eps_list = [0.005, 0.01, 0.02, 0.04]
        if approx_eps is not None:
            eps_list = [approx_eps]
        best = (np.inf, None, None)
        for frac in eps_list:
            eps = max(0.5, frac * arc)
            approx = cv2.approxPolyDP(contour, eps, True)
            verts = approx.reshape(-1, 2)
            if len(verts) < 3 or len(verts) > 60:
                continue
            A_cont = abs(cv2.contourArea(contour))
            A_approx = abs(cv2.contourArea(approx))
            if A_cont < 1e-6 or abs(A_cont - A_approx) / A_cont > 0.6:
                continue
            verts_s = simplify_vertices(verts, min_dist=max(4.0, 0.008 * arc))
            if len(verts_s) < 3:
                continue
            angs = np.arctan2(verts_s[:,1]-cy, verts_s[:,0]-cx)
            angs = np.sort(np.mod(angs, 2*np.pi))
            gaps = np.diff(np.concatenate([angs, [angs[0]+2*np.pi]]))
            rel = np.std(gaps) / (np.mean(gaps) + 1e-9)
            if rel > 0.30:
                continue
            score, r_scaled = evaluate_polygon_candidate(verts_s, (cx, cy), theta_uniform, r_interp)
            score += 0.002 * len(verts_s)
            if score < best[0]:
                best = (score, r_scaled, len(verts_s))
            if force_polygon and best[1] is not None:
                return theta_uniform, best[1], (cx, cy), int(best[2])
        if best[1] is not None and best[0] < 0.35:
            return theta_uniform, best[1], (cx, cy), int(best[2])
    except Exception:
        pass

    try:
        r_norm = (r_interp - np.nanmean(r_interp)) / (np.nanstd(r_interp) + 1e-9)
        fft = np.abs(np.fft.rfft(r_norm))
        if len(fft) > 3:
            peak = np.argmax(fft[1:]) + 1
            if peak >= 3 and peak <= 60:
                candidate_ns = [int(peak)]
                if peak % 2 == 0 and peak//2 >= 3:
                    candidate_ns.append(int(peak//2))
                best = (np.inf, None, None)
                for n in candidate_ns:
                    rf = get_regular_polygon_radius_function(n, 1.0)
                    r_cand = rf(theta_uniform)
                    mu1, s1 = np.nanmean(r_interp), np.nanstd(r_interp) + 1e-9
                    mu2, s2 = np.nanmean(r_cand), np.nanstd(r_cand) + 1e-9
                    r1 = (r_interp - mu1) / s1
                    r2 = (r_cand - mu2) / s2
                    score = np.sqrt(np.nanmean((r1 - r2)**2)) + 0.003 * n
                    if score < best[0]:
                        r_scaled = (r_cand - mu2) / s2 * s1 + mu1
                        best = (score, r_scaled, n)
                if best[1] is not None and best[0] < 0.35:
                    return theta_uniform, best[1], (cx, cy), int(best[2])
    except Exception:
        pass

    return theta_uniform, r_interp, (cx, cy), None


def fit_polar_fourier_simple(theta, r, terms=12):
    terms = max(1, min(terms, 100))
    r_mean = np.nanmean(r)
    r_std = np.nanstd(r) + 1e-9
    r_n = (r - r_mean) / r_std
    a0 = np.mean(r_n)
    an = np.zeros(terms)
    bn = np.zeros(terms)
    for n in range(1, terms+1):
        an[n-1] = 2 * np.mean(r_n * np.cos(n * theta))
        bn[n-1] = 2 * np.mean(r_n * np.sin(n * theta))
    def r_func(th):
        val = a0/2 + sum(an[k]*np.cos((k+1)*th) + bn[k]*np.sin((k+1)*th) for k in range(len(an)))
        return val * r_std + r_mean
    pred_n = a0/2 + sum(an[k]*np.cos((k+1)*theta) + bn[k]*np.sin((k+1)*theta) for k in range(len(an)))
    rmse = np.sqrt(np.nanmean((r_n - pred_n)**2))
    return r_func, rmse, (a0, an, bn)


def get_regular_polygon_radius_function(n, scale=1.0):
    def radius_func(theta_val):
        theta_norm = theta_val % (2 * np.pi)
        angle_per_side = 2 * np.pi / n
        local_angle = (theta_norm % angle_per_side) - np.pi / n
        numerator = np.cos(np.pi / n)
        denominator = np.cos(local_angle)
        denominator = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)
        return scale * numerator / denominator
    return radius_func


def fit_curve(x, y, degree=3):
    """다항식 피팅"""
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min + 1e-8) - 1
    degree = min(degree, 3)
    coeffs = np.polyfit(x_norm, y_norm, degree)
    p_norm = np.poly1d(coeffs)
    yhat_norm = p_norm(x_norm)
    rmse = np.sqrt(np.mean((y_norm - yhat_norm) ** 2))
    def p_denorm(x_val):
        x_n = 2 * (x_val - x_min) / (x_max - x_min + 1e-8) - 1
        y_n = p_norm(x_n)
        return (y_n + 1) * (y_max - y_min) / 2 + y_min
    return p_denorm, rmse, coeffs


def latex_parametric3d(a0_r, an_r, bn_r, a0_z=None, an_z=None, bn_z=None):
    parts = []
    parts.append(f'{a0_r/2:.4f}')
    for i, a in enumerate(an_r):
        n = i+1
        if abs(a) > 1e-6:
            parts.append(f'{a:.4f}\\cos({n}t)')
    for i, b in enumerate(bn_r):
        n = i+1
        if abs(b) > 1e-6:
            parts.append(f'{b:.4f}\\sin({n}t)')
    r_body = ' + '.join(parts).replace('+ -', '- ')
    z_body = '0'
    if a0_z is not None:
        zp = [f'{a0_z/2:.4f}']
        for i,a in enumerate(an_z):
            n=i+1
            if abs(a)>1e-6:
                zp.append(f'{a:.4f}\\cos({n}t)')
        for i,b in enumerate(bn_z):
            n=i+1
            if abs(b)>1e-6:
                zp.append(f'{b:.4f}\\sin({n}t)')
        z_body = ' + '.join(zp).replace('+ -', '- ')
    latex = f'''$\\begin{{cases}} x(t)=({r_body})\\cos t\\\\ y(t)=({r_body})\\sin t\\\\ z(t)={z_body} \\end{{cases}}$'''
    return latex


def desmos_parametric3d(a0_r, an_r, bn_r, a0_z=None, an_z=None, bn_z=None):
    terms = [f'{a0_r/2:.4f}']
    for i,a in enumerate(an_r):
        n=i+1
        if abs(a)>1e-6:
            sign = '+' if a>=0 else '-'
            terms.append(f'{sign} {abs(a):.4f}*cos({n}*t)')
    for i,b in enumerate(bn_r):
        n=i+1
        if abs(b)>1e-6:
            sign = '+' if b>=0 else '-'
            terms.append(f'{sign} {abs(b):.4f}*sin({n}*t)')
    r_body = ' '.join(terms).replace('+ -', '- ')
    z_body = '0'
    if a0_z is not None:
        zp = [f'{a0_z/2:.4f}']
        for i,a in enumerate(an_z):
            n=i+1
            if abs(a)>1e-6:
                sign='+' if a>=0 else '-'
                zp.append(f'{sign} {abs(a):.4f}*cos({n}*t)')
        for i,b in enumerate(bn_z):
            n=i+1
            if abs(b)>1e-6:
                sign='+' if b>=0 else '-'
                zp.append(f'{sign} {abs(b):.4f}*sin({n}*t)')
        z_body = ' '.join(zp).replace('+ -', '- ')
    return f'r(t) = {r_body}\nx(t) = r(t)*cos(t)\ny(t) = r(t)*sin(t)\nz(t) = {z_body}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('image')
    p.add_argument('--mode', choices=['parametric3d','curve','surface'], default='parametric3d')
    p.add_argument('--terms', type=int, default=12)
    p.add_argument('--prefix', default='output')
    p.add_argument('--force-polygon', action='store_true')
    p.add_argument('--approx-eps', type=float, default=None)
    args = p.parse_args()

    img = load_image(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.mode == 'parametric3d':
        contour = extract_largest_contour(gray)
        if contour is None:
            print('윤곽을 찾을 수 없습니다.')
            sys.exit(1)
        theta, r, center, n = contour_to_polar(contour, force_polygon=args.force_polygon, approx_eps=args.approx_eps)
        
        # 작은 다각형(3,4,5)은 더 많은 항을 사용하여 직선 표현 향상
        terms_to_use = args.terms
        if n is not None and n <= 5:
            # 작은 다각형: 항 개수를 3배 증가
            terms_to_use = max(args.terms, int(args.terms * 3 / max(1, n)))
        
        r_func, rmse_r, (a0r, anr, bnr) = fit_polar_fourier_simple(theta, r, terms=terms_to_use)
        z_data = 0.2 * np.sin(2*theta)
        z_func, rmse_z, (a0z, anz, bnz) = fit_polar_fourier_simple(theta, z_data, terms=max(3, terms_to_use//4))

        t = np.linspace(0, 2*np.pi, 1500)
        r_t = r_func(t); x_t = r_t * np.cos(t); y_t = r_t * np.sin(t); z_t = z_func(t)
        if len(x_t)>0:
            x_t = (x_t - x_t.mean()) / (x_t.std() + 1e-9) * 50
            y_t = (y_t - y_t.mean()) / (y_t.std() + 1e-9) * 50
            z_t = (z_t - z_t.mean()) / (z_t.std() + 1e-9) * 20
        fig = plt.figure(figsize=(12,4))
        ax = fig.add_subplot(131, projection='3d')
        ax.plot(x_t, y_t, z_t, 'b-')
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot(x_t, y_t, z_t, 'g-')
        ax3 = fig.add_subplot(133)
        ax3.plot(x_t, y_t, 'b-')
        ax3.axis('equal')
        out = f'{args.prefix}_parametric3d.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 극좌표 r(theta)과 매개변수 함수 그래프
        fig_funcs = plt.figure(figsize=(14, 10))
        
        # r(theta) 그래프
        ax_r = fig_funcs.add_subplot(2, 2, 1)
        ax_r.plot(theta, r, 'b.', markersize=2, alpha=0.5, label='extracted')
        ax_r.plot(theta, r, 'b-', alpha=0.3, linewidth=0.5)
        r_plot = r_func(theta)
        ax_r.plot(theta, r_plot, 'r-', linewidth=2, label='fitted r(theta)')
        ax_r.set_xlabel('theta (rad)')
        ax_r.set_ylabel('r')
        ax_r.set_title(f'Polar: r(theta) [RMSE={rmse_r:.6f}]')
        ax_r.grid(True, alpha=0.3)
        ax_r.legend()
        
        # x(t) 그래프
        ax_x = fig_funcs.add_subplot(2, 2, 2)
        ax_x.plot(t, x_t, 'r-', linewidth=2)
        ax_x.set_xlabel('t (parameter)')
        ax_x.set_ylabel('x(t)')
        ax_x.set_title('x(t) = r(t)*cos(t)')
        ax_x.grid(True, alpha=0.3)
        
        # y(t) 그래프
        ax_y = fig_funcs.add_subplot(2, 2, 3)
        ax_y.plot(t, y_t, 'g-', linewidth=2)
        ax_y.set_xlabel('t (parameter)')
        ax_y.set_ylabel('y(t)')
        ax_y.set_title('y(t) = r(t)*sin(t)')
        ax_y.grid(True, alpha=0.3)
        
        # z(t) 그래프
        ax_z = fig_funcs.add_subplot(2, 2, 4)
        ax_z.plot(t, z_t, 'b-', linewidth=2)
        ax_z.set_xlabel('t (parameter)')
        ax_z.set_ylabel('z(t)')
        ax_z.set_title('z(t) = elevation')
        ax_z.grid(True, alpha=0.3)
        
        out_funcs = f'{args.prefix}_parametric_functions.png'
        fig_funcs.savefig(out_funcs, dpi=150, bbox_inches='tight')
        plt.close(fig_funcs)

        latex = latex_parametric3d(a0r, anr, bnr, a0z, anz, bnz)
        desmos = desmos_parametric3d(a0r, anr, bnr, a0z, anz, bnz)

        print('--- Parametric 3D ---')
        print(latex)
        print('--- Desmos ---')
        print(desmos)
        print('RMSE r (normalized):', f'{rmse_r:.6f}')
        print('output 3D:', out)
        print('output functions:', out_funcs)

    elif args.mode == 'curve':
        contour = extract_largest_contour(gray)
        if contour is None:
            print('윤곽을 찾을 수 없습니다.')
            sys.exit(1)
        x, y = contour.reshape(-1,2)[:,0], contour.reshape(-1,2)[:,1]
        y = gray.shape[0] - y
        func, rmse, coeffs = fit_curve(x, y, degree=3)
        xp = np.linspace(x.min(), x.max(), 800)
        yp = func(xp)
        fig, ax = plt.subplots()
        ax.plot(x, y, '.', markersize=2)
        ax.plot(xp, yp, 'r-')
        out = f'{args.prefix}_curve.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('curve output:', out)

    else:
        h,w = gray.shape
        X,Y = np.meshgrid(np.arange(w), np.arange(h))
        Z = gray.astype(float)/255.0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        out = f'{args.prefix}_surface.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('surface output:', out)


if __name__ == '__main__':
    main()
