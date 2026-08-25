# Kimodo macOS validation results

검증일은 2026-08-24이며, Apple Silicon M4 Max(16 CPU cores, 128 GB unified
memory), macOS 26.5에서 실행했다. 각 측정은 독립적인 CLI 프로세스이므로
Llama 3 text encoder와 Kimodo checkpoint를 새로 올리는 시간을 포함한다.

| Backend | Duration | Diffusion steps | Frames | Total time | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| MPS | 1.0 s | 5 | 30 | 15.53 s | pass |
| CPU | 1.0 s | 5 | 30 | 12.74 s | pass |
| MPS | 3.0 s | 100 | 90 | 19.08 s | pass |
| MPS | 5.0 s | 100 | 150 | 14.48 s | pass |
| CPU | 5.0 s | 100 | 150 | 33.41 s | pass |

같은 5초 prompt와 seed를 사용한 비교에서 MPS는 전체 실행 시간이 CPU보다
약 2.3배 빨랐다. 100-step diffusion 구간만 비교하면 MPS 약 5.54초,
CPU 약 23.47초로 MPS가 약 4.2배 빨랐다. 짧은 5-step 실험은 모델 로딩
비중이 커서 backend 계산 성능 비교에는 적합하지 않다.

모든 당시 비교 결과는 pickle 없이 열렸고, `posed_joints (T,77,3)`,
`global_rot_mats (T,77,3,3)`, `foot_contacts (T,6)` 형상과 유한한 수치
배열을 가졌다. RIM v6의 설치된 `Soma77Motion.from_kimodo_npz(...)`로도
세 파일을 검증했다.

같은 seed라도 MPS와 CPU 결과는 bitwise deterministic하지 않다. 1초
스모크 결과의 최대 joint-position 절대 차이는 약 0.7001이었고 contact
mask도 서로 달랐다. Backend를 바꿔가며 결과 동일성을 기대하지 말고,
생성 실험에서는 backend와 seed를 함께 기록해야 한다.

현재 검증은 `--no-postprocess`를 사용한다. Kimodo의 핵심 text encoder와
diffusion generation은 MPS/CPU에서 모두 동작하지만, 선택적인 native
motion-correction 빌드와 시각적 품질 평가는 별도 단계다.

현재 Transformers/PEFT 조합은 두 번째 LLM2Vec adapter를 올릴 때
`Already found a peft_config` 경고를 출력한다. 프로세스는 정상 종료하고
유효한 NPZ를 만들지만, 이 검증은 실행 호환성과 RIM 입력 계약까지만
확인한 것이다. prompt 의미 일치도와 adapter 품질은 렌더링 비교를 거쳐
별도로 판단해야 한다.

## 2026-08-25 prompt-rich source set

기존 성능 비교 NPZ와 초기 motion 후보는 prompt가 archive 안에 없었기
때문에 모두 삭제했다. 현재 `outputs/`에는 다음 세 개의 self-contained
schema-v1 파일만 남아 있다. 각 생성은 MPS, 5초, 150 frames, 30 FPS,
100 diffusion steps, seed 7, postprocess off 구성을 사용했다.

| Output | Summary prompt | Total time | SHA-256 |
| --- | --- | ---: | --- |
| `side_steps_arms_open_5s.npz` | Side steps with arms opening | 17.61 s | `ef203d00eb7f1468272dbc57d40bb66ef1961d7f97045aeaf1b2415cb62f91f0` |
| `march_arm_swing_5s.npz` | March in place with arm swings | 14.64 s | `b015c13f11d5017148cb2d73548535fc99534fb12469b7586e770bc04dbd866b` |
| `forward_back_arm_reach_5s.npz` | Forward and backward steps with arm reaches | 13.92 s | `3b4f5c273b56f3f256e1984cfbb618268b58972fdc9034666239e7a94f3e2e7f` |

세 파일 모두 required metadata, no-object-array, SOMA77 shape, finiteness,
frame/FPS/duration consistency 검사를 통과했다. exact full prompt와 summary
prompt는 각 NPZ 내부에 있고, generator revision, model, seed, sampling
configuration, text encoder, device flags, UTC timestamp, Python, PyTorch,
platform 정보도 같은 archive에서 `allow_pickle=False`로 읽힌다.

RIM v6에는 세 파일을 그대로 복사했고, strict
`Soma77Motion.from_kimodo_npz(...)`로 다시 검증했다. 그중
`side_steps_arms_open_5s`는 K1 notebook `03`–`05`의 ordered numerical
pipeline까지 완료했다. Qt viewer의 prompt 배치, keyboard input, camera,
skin synchronization은 desktop `Restart Kernel and Run All` 확인이 남아
있다.
