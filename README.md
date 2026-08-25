# Kimodo on macOS

Apple Silicon Mac에서 Kimodo SOMA77 text-to-motion을 MPS 또는 CPU로
실행하고, 생성한 모션을 RIM v6에서 바로 불러오기 위한 재현 가능한
wrapper 저장소다. 모션과 full/summary prompt, 생성 설정, runtime
provenance를 하나의 pickle-free NPZ에 함께 저장한다.

이 저장소의 표준 환경 관리자는 **Conda**다. `uv`, `venv`, Poetry는
사용하지 않는다.

## 빠른 설치

준비물은 Apple Silicon Mac, Git, Conda, 약 25GB 이상의 여유 공간이다.
Conda가 없다면 Apple Silicon용
[Miniforge](https://github.com/conda-forge/miniforge)를 먼저 설치한다.
Kimodo checkpoint와 Llama 3 text encoder를 받으려면 Hugging Face 계정과
각 모델의 사용 조건 동의가 필요할 수 있다.

```bash
git clone --recurse-submodules https://github.com/sjchoi86/kimodo_macos.git
cd kimodo_macos
./setup_macos.sh
conda run -n kimodo-macos hf auth login
```

`setup_macos.sh`는 다음 작업을 수행한다.

1. 고정된 MPS 지원 Kimodo submodule을 내려받는다.
2. `environment.yml`로 `kimodo-macos` Conda 환경을 만든다.
3. 검증된 PyTorch와 Kimodo를 설치한다.
4. PyTorch의 MPS 사용 가능 여부를 출력한다.

선택적인 native motion correction은 Apple Silicon 기본 설치에서 제외한다.
이 저장소의 생성 명령도 `--no-postprocess`를 사용한다. Kimodo text
encoder와 diffusion motion generation 자체는 MPS와 CPU에서 모두 동작한다.

## 첫 모션 만들기

최초 실행은 모델을 내려받아야 하므로 online mode를 명시한다.

```bash
HF_HUB_OFFLINE=0 ./run_motion.sh \
    mps \
    "A person walks forward, turns to the right, and stops upright" \
    5.0 \
    100 \
    walk_turn_stop_5s \
    "Walk, turn, and stop" \
    7
```

인자는 다음 순서다.

```text
backend full_prompt duration_sec diffusion_steps output_name summary_prompt seed
```

backend는 `mps` 또는 `cpu`다. 인자를 생략하면 MPS, 3초, 100 steps,
seed 7의 기본 예제를 생성한다.

```bash
./run_motion.sh mps
./run_motion.sh cpu
```

모델을 한 번 모두 받은 다음 네트워크 없이 실행하려면 다음처럼 한다.

```bash
HF_HUB_OFFLINE=1 ./run_motion.sh mps \
    "A person takes two side steps while opening both arms" \
    5.0 100 my_side_steps "Side steps with open arms" 7
```

공식 Meta Llama 저장소 대신 호환 mirror를 사용하는 환경이라면 실제 base
model 식별자를 metadata에 남기도록 `TEXT_ENCODER_BASE`도 함께 지정한다.

## 파일이 저장되는 위치

생성 결과는 항상 이 저장소의 다음 위치에 저장된다.

```text
outputs/<output_name>.npz
```

예를 들어 `output_name`이 `walk_turn_stop_5s`이면 실제 경로는
`outputs/walk_turn_stop_5s.npz`다. 같은 이름의 파일이 이미 있으면
스크립트가 중단하며 기존 결과를 덮어쓰지 않는다.

Kimodo가 numeric motion arrays를 쓴 직후 `embed_motion_metadata.py`가 같은
NPZ를 원자적으로 다시 작성한다. 따라서 sidecar JSON 없이 파일 하나만
옮겨도 다음 내용이 모두 유지된다.

- `posed_joints (T,77,3)`와 global/local rotations
- `foot_contacts`, scalar `fps`, frame count와 duration
- exact full prompt와 짧은 summary prompt
- model, MPS/CPU backend, diffusion steps, seed와 sampling 설정
- generator repository/revision, 생성 UTC 시간, Python/PyTorch/platform 정보
- text encoder base와 adapter 식별자

모든 문자열은 NumPy Unicode dtype이다. archive는 항상
`np.load(path,allow_pickle=False)`로 읽을 수 있다. 생성이 끝나면
`validate_motion.py`가 필수 key, SOMA77 형상, 유한값, prompt metadata와
시간 일관성을 자동 검사한다.

Hugging Face cache와 checkpoint는 각각 `hf-cache/`, `checkpoints/` 아래에
두며 Git에는 포함하지 않는다. 새로 생성한 NPZ도 기본적으로 무시된다.
다만 아래의 검증된 5초 예제 3개는 clone 직후 확인할 수 있도록 저장소에
포함한다.

- `outputs/side_steps_arms_open_5s.npz`
- `outputs/march_arm_swing_5s.npz`
- `outputs/forward_back_arm_reach_5s.npz`

## RIM v6에서 사용하기

`kimodo_macos`와 `rim_v6`가 같은 상위 폴더에 있다면 다음 명령으로 모든
NPZ를 RIM v6의 motion source 폴더에 복사할 수 있다.

```bash
./copy_to_rim_v6.sh ../rim_v6
```

목적지는 다음과 같다.

```text
rim_v6/notebooks/15_motion_retargeting/motion_data/
```

파일 이름 앞에는 source와 model을 구분하는
`kimodo_soma_rp_v1_` prefix가 자동으로 붙는다. 목적지에 내용이 다른
동명 파일이 있으면 덮어쓰지 않고 중단한다.

RIM v6 Python API에서 원본 `outputs/` 경로를 직접 읽는 것도 가능하다.

```python
from pathlib import Path

from rim_v6.motion import Soma77Motion,Soma77Skeleton
from rim_v6.utility import rpy2r

motion_path = Path("../kimodo_macos/outputs/side_steps_arms_open_5s.npz")
motion = Soma77Motion.from_kimodo_npz(
    motion_path,
    skeleton=Soma77Skeleton(),
    R_source_to_target=rpy2r([90.0,0.0,0.0],unit="deg"),
    fps_default=30.0,
)

print(motion.generation.prompt_full)
print(motion.frame_count,motion.fps,motion.duration_sec)
```

15번 notebook에서 복사한 모션을 선택하려면 `MOTION_SPECS`에 다음처럼
등록한다.

```python
"walk_turn_stop_5s":{
    "display_name":"Walk, turn, and stop",
    "path":"motion_data/kimodo_soma_rp_v1_walk_turn_stop_5s.npz",
    "fps":30.0,
},
```

RIM v6의 strict loader는 prompt-rich metadata schema를 기본으로 요구한다.
이 저장소의 `run_motion.sh`로 만든 파일은 그 계약을 그대로 만족한다.

## 검증된 환경과 성능

- Apple Silicon M4 Max, macOS 26.5
- Conda Python 3.10.20
- PyTorch 2.13.0
- Kimodo MPS fork commit `598fee96ca39bff9403db652d756d9046f089fc3`
- Kimodo checkpoint `Kimodo-SOMA-RP-v1`

5초, 150-frame, 100-step 비교에서는 전체 실행 시간이 MPS 14.48초,
CPU 33.41초였고 diffusion 구간은 MPS가 약 4.2배 빨랐다. 세부 측정과
한계는 [RESULTS.md](RESULTS.md)에 기록했다.

MPS와 CPU는 같은 seed에서도 bitwise-identical motion을 보장하지 않는다.
생성 결과를 재현·비교할 때는 prompt와 seed뿐 아니라 backend도 함께
확인해야 한다.

## 라이선스와 원본

`kimodo/`는 Apache-2.0 Kimodo MPS fork를 고정한 Git submodule이다. 모델과
text encoder에는 각 배포처의 별도 사용 조건이 적용된다. 공개 결과를
재배포하거나 상업적으로 사용할 때는 checkpoint 및 Llama 라이선스를
직접 확인해야 한다.

- [Official Kimodo repository](https://github.com/nv-tlabs/kimodo)
- [Pinned MPS fork](https://github.com/atticus-lv/kimodo/commit/598fee96ca39bff9403db652d756d9046f089fc3)
- [Official Kimodo installation](https://github.com/nv-tlabs/kimodo/blob/main/docs/source/getting_started/installation.md)
- [Official NPZ format](https://github.com/nv-tlabs/kimodo/blob/main/docs/source/user_guide/output_formats.md)
