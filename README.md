================================================================================
  이미지/도형을 3D 매개변수 방정식으로 변환하기
  Image/Shape to 3D Parametric Equation Converter
================================================================================

【 프로젝트 개요 】

이 프로젝트는 이미지나 도형(정다각형, 정다각별)을 분석하여 3D 매개변수 방정식
(x(t), y(t), z(t))으로 변환하고, Desmos/LaTeX 형식의 수식 및 시각화 그래프를
생성하는 도구입니다.

주요 기능:
  - 정다각형 자동 생성 (3각형 ~ 30각형 이상)
  - 정다각별 자동 생성 (5각별 ~ 30각별 이상, 간격 자동 설정)
  - 이미지 윤곽 추출
  - 극좌표 변환 및 푸리에 급수 적합
  - 3D 매개변수 함수 생성
  - 라이브 그래프 출력 (r(θ), x(t), y(t), z(t))
  - Desmos/LaTeX 형식 수식 출력


【 설치 및 환경 설정 】

1. Python 3.7 이상 필요

2. 필요한 패키지 설치:
   pip install opencv-python numpy matplotlib

   또는 requirements.txt 사용:
   pip install -r requirements.txt

3. 가상 환경 사용 (권장):
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt


【 파일 설명 】

1. generate_polygons.py
   정다각형(삼각형, 사각형, 오각형 등) 이미지 생성
   
2. generate_star_polygons.py
   정다각별(오각별, 7각별 등) 이미지 생성
   
3. convert_image_new.py
   이미지를 3D 매개변수 방정식으로 변환 (권장)
   
4. convert_shape.py
   정다각형/정다각별을 3D 매개변수 방정식으로 변환
   
5. convert_image.py
   기존 버전 (convert_image_new.py 사용 권장)


【 사용법 】

========== 1단계: 도형 생성 ==========

【 정다각형 생성 】

  # 삼각형 생성
  python generate_polygons.py --n 3 --output triangle.png

  # 사각형 생성
  python generate_polygons.py --n 4 --output square.png

  # 오각형 생성
  python generate_polygons.py --n 5 --output pentagon.png

  # 여러 개 생성 (3~20각형)
  python generate_polygons.py --range 3-20

  # 기본 세트 생성 (3, 4, 5, 6, 8, 10, 12, 15, 20각형)
  python generate_polygons.py --all


【 정다각별 생성 】

  # 24각별 생성 (간격 자동 설정)
  python generate_star_polygons.py --n 24 --output star_24.png

  # 5각별(오각별) 생성
  python generate_star_polygons.py --n 5 --output pentagram.png

  # 12각별 생성
  python generate_star_polygons.py --n 12 --output star_12.png

  # 범위 생성 (5~30각별)
  python generate_star_polygons.py --range 5-30

  # 모든 각별 생성 (5~30)
  python generate_star_polygons.py --all


========== 2단계: 도형 변환 ==========

【 기본 사용법 】

  python convert_shape.py <이미지경로> --mode parametric3d --terms <항의개수> --prefix <출력파일명>


【 정다각형 변환 】

  # 삼각형 변환 (--terms 24 사용하면 자동으로 72항 사용)
  python convert_shape.py triangle.png --mode parametric3d --terms 24 --prefix result_tri

  # 오각형 변환
  python convert_shape.py pentagon.png --mode parametric3d --terms 24 --prefix result_pent

  # 20각형 변환
  python convert_shape.py polygon_20.png --mode parametric3d --terms 12 --prefix result_20


【 정다각별 변환 】

  # 24각별 변환
  python convert_shape.py star_24.png --mode parametric3d --terms 24 --prefix result_star_24

  # 5각별 변환
  python convert_shape.py pentagram.png --mode parametric3d --terms 20 --prefix result_penta


【 옵션 설명 】

  --mode
    parametric3d  : 3D 매개변수 함수 생성 (기본값)
    curve         : 2D 곡선 (개발 중)
    surface       : 3D 표면 (개발 중)

  --terms <숫자>
    푸리에 급수의 항 개수 (기본값: 12)
    - 작은 다각형(3,4,5)은 자동으로 항 증가
    - 값이 클수록 더 정확하지만 계산 시간 증가

  --prefix <파일명>
    출력 파일 이름 접두사
    기본값: "output"
    출력: {prefix}_parametric3d.png, {prefix}_parametric_functions.png

  --force-polygon
    윤곽선 근사 기반으로 정다각형 강제 인식

  --approx-eps <숫자>
    approxPolyDP epsilon 비율 (예: 0.02)


【 출력 파일 】

각 변환 후 다음 두 개의 파일이 생성됩니다:

1. {prefix}_parametric3d.png
   3D 곡선 시각화 (3개 각도)
   - 첫 번째 서브플롯: 정면 뷰
   - 두 번째 서브플롯: 사각 뷰 (elev=20°, azim=45°)
   - 세 번째 서브플롯: 위에서 본 2D 뷰

2. {prefix}_parametric_functions.png
   함수 그래프 (2×2 레이아웃)
   - 좌상: r(θ) 극좌표 함수 (원래 데이터 + 피팅 곡선)
   - 우상: x(t) = r(t)·cos(t)
   - 좌하: y(t) = r(t)·sin(t)
   - 우하: z(t) 고도 함수

콘솔 출력:
   - LaTeX 형식 수식 (Overleaf 등에서 사용 가능)
   - Desmos 형식 수식 (https://www.desmos.com 에서 사용 가능)
   - RMSE 오차 지표


【 사용 예제 】

========== 예제 1: 삼각형 분석 ==========

  # 삼각형 생성
  python generate_polygons.py --n 3 --output triangle.png

  # 삼각형을 3D로 변환
  python convert_shape.py triangle.png --mode parametric3d --terms 24 --prefix tri

  # 출력 파일
  # - tri_parametric3d.png
  # - tri_parametric_functions.png
  # - 콘솔에 LaTeX/Desmos 수식 출력


========== 예제 2: 24각별 분석 ==========

  # 24각별 생성
  python generate_star_polygons.py --n 24 --output star_24.png

  # 24각별을 3D로 변환
  python convert_shape.py star_24.png --mode parametric3d --terms 30 --prefix star24

  # 출력 파일
  # - star24_parametric3d.png
  # - star24_parametric_functions.png
  # - 콘솔에 LaTeX/Desmos 수식 출력


========== 예제 3: 자신의 이미지 사용 ==========

  # 자신의 PNG/JPG 이미지를 convert_image_new.py 로 처리
  python convert_image_new.py my_shape.png --mode parametric3d --terms 20 --prefix my_result

  # 또는 convert_shape.py 사용
  python convert_shape.py my_shape.png --mode parametric3d --terms 20 --prefix my_result


========== 예제 4: 배치 처리 ==========

  # 3~10각형 모두 생성 및 변환
  python generate_polygons.py --range 3-10
  for /L %n in (3,1,10) do (
    python convert_shape.py polygon_%n.png --mode parametric3d --terms 12 --prefix polygon_%n
  )

  # 5~20각별 모두 생성 및 변환
  python generate_star_polygons.py --range 5-20
  for /L %n in (5,1,20) do (
    python convert_shape.py star_%n.png --mode parametric3d --terms 24 --prefix star_%n
  )


【 출력 수식 해석 】

생성된 수식은 다음과 같은 형태입니다:

  LaTeX:
  x(t) = (a0 + a1·cos(t) + b1·sin(t) + ...) · cos(t)
  y(t) = (a0 + a1·cos(t) + b1·sin(t) + ...) · sin(t)
  z(t) = c0 + c1·cos(t) + c1·sin(t) + ...

  여기서:
  - r(t) = a0 + a1·cos(t) + b1·sin(t) + ... (극좌표 반지름)
  - (x(t), y(t)) = r(t)·(cos(t), sin(t)) (극좌표 변환)
  - z(t) = 고도 함수 (시각화 효과)

  t는 0 ~ 2π의 매개변수입니다.


【 Desmos에서 보기 】

1. https://www.desmos.com/3d 접속
2. 콘솔 출력의 Desmos 형식 수식 복사
3. 그래프에 붙여넣기
4. 실시간으로 3D 곡선 표시


【 LaTeX에서 보기 】

1. Overleaf.com 또는 로컬 LaTeX 편집기 열기
2. 콘솔 출력의 LaTeX 형식 수식을 문서에 삽입
3. 컴파일하면 수식 표시


【 문제 해결 】

Q: "윤곽을 찾을 수 없습니다" 오류 발생
A: 이미지가 너무 밝거나 어두울 수 있습니다.
   - generate_polygons.py 또는 generate_star_polygons.py로 샘플 이미지 생성 후 사용

Q: 변환 결과가 원본 도형과 다름
A: --terms 값을 증가시켜보세요.
   - 예: --terms 24 또는 --terms 30
   - 값이 클수록 더 정확하지만 곡선이 많아집니다.

Q: 작은 다각형(3, 4, 5각형)이 곡선이 많이 보임
A: 자동으로 항이 증가하도록 설정되었습니다.
   - 더 많은 항을 사용하여 직선을 정확히 표현합니다.

Q: 수식이 너무 길어서 복잡함
A: 낮은 RMSE 값을 가진 항들만 선택하여 사용하세요.
   - --terms 값을 낮추어 항 개수를 줄일 수 있습니다.


【 권장 설정 】

도형 종류별 권장 --terms 값:

  정다각형:
    - 3각형:  24 이상 (3배 자동 증가 → 72항)
    - 4각형:  20 이상 (2.25배 자동 증가 → 45항)
    - 5각형:  20 이상 (2.4배 자동 증가 → 48항)
    - 6각형 이상: 12 이상

  정다각별:
    - 5각별: 15 이상
    - 7각별: 18 이상
    - 12각별: 20 이상
    - 24각별: 24 이상


【 성능 팁 】

1. 정다각형/다각별은 generate_polygons.py와 generate_star_polygons.py로 생성된
   이미지를 사용하는 것이 가장 정확합니다.

2. 자신의 이미지를 사용할 때는 명확한 검은색 윤곽선과 흰색 배경을 가진
   이미지를 사용하세요.

3. 투명 PNG (alpha 채널)는 자동으로 흰색 배경으로 합성됩니다.

4. 더 빠른 처리를 원하면 --terms 값을 낮추세요.
   더 정확한 표현을 원하면 --terms 값을 높이세요.


【 라이선스 및 저작권 】

이 프로젝트는 교육 및 개인 사용 목적으로 제공됩니다.
자유롭게 수정하여 사용할 수 있습니다.


【 문의 및 피드백 】

이 도구를 사용하며 문제가 발생하거나 개선 제안이 있으면 알려주세요.


================================================================================
마지막 업데이트: 2025년 12월 31일
================================================================================
