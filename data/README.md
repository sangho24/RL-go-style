# data 디렉토리 안내

이 디렉토리에는 바둑 포지션을 전처리한 `.npz` 파일들이 들어 있습니다.  
모든 `.npz` 파일은 공통적으로 다음 키를 가집니다.

- `states`: float32, shape `(N, 3, 19, 19)`
  - 채널 0: 현재 플레이어의 돌 (0/1)
  - 채널 1: 상대 플레이어의 돌 (0/1)
  - 채널 2: 보조 정보 (예: 턴/플레이어 등, 프로젝트 코드와 동일한 방식)
- `actions`: int64, shape `(N,)`
  - 각 수는 `0 ~ 360` 사이의 인덱스로 인코딩됨  
    (row * 19 + col 형태, `src` 코드와 동일)

raw data는 명시된 출처에서 한 번에 다운받아서 unzip할 수 있습니다. 
---

## 디렉토리 구조

```text
data/
  processed/
    base_policy_100k.npz
    shin_policy_50k.npz

