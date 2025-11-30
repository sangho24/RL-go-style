import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sgfmill import sgf, boards

from shin_utils import is_shin_jinseo

BOARD_SIZE = 19


def encode_board(board: boards.Board, next_color: str) -> np.ndarray:
    C, H, W = 3, BOARD_SIZE, BOARD_SIZE
    x = np.zeros((C, H, W), dtype=np.float32)

    for r in range(H):
        for c in range(W):
            stone = board.get(r, c)  # 'b', 'w', or None
            if stone == 'b':
                x[0, r, c] = 1.0
            elif stone == 'w':
                x[1, r, c] = 1.0

    if next_color == 'b':
        x[2, :, :] = 1.0
    else:
        x[2, :, :] = 0.0

    return x


def get_players_and_color(game: sgf.Sgf_game) -> Tuple[str, str, str]:
    """
    SGF 게임에서 (PB, PW, shin_color) 반환.
    shin_color: 'b' / 'w' / '' (못 찾으면 빈 문자열)
    """
    root = game.get_root()

    # sgfmill Node.get은 default 인자를 안 받으므로 KeyError 처리
    try:
        pb = root.get("PB")
    except KeyError:
        pb = ""
    try:
        pw = root.get("PW")
    except KeyError:
        pw = ""

    if isinstance(pb, bytes):
        pb = pb.decode("utf-8", errors="ignore")
    if isinstance(pw, bytes):
        pw = pw.decode("utf-8", errors="ignore")

    shin_is_black = is_shin_jinseo(pb)
    shin_is_white = is_shin_jinseo(pw)

    shin_color = ""
    if shin_is_black and not shin_is_white:
        shin_color = 'b'
    elif shin_is_white and not shin_is_black:
        shin_color = 'w'
    elif shin_is_black and shin_is_white:
        shin_color = 'b'  # 양쪽 다 매칭되면 일단 흑으로

    return pb, pw, shin_color


def sgf_to_shin_positions(
    sgf_path: Path,
    skip_handicap: bool = True,
) -> List[Tuple[np.ndarray, int]]:
    """
    하나의 SGF에서 '신진서가 둔 수'만 (state, action) 포지션으로 추출.
    """
    try:
        with sgf_path.open("rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
    except Exception as e:
        print(f"[WARN] Failed to parse {sgf_path}: {e}")
        return []

    if game.get_size() != BOARD_SIZE:
        return []

    pb, pw, shin_color = get_players_and_color(game)

    if shin_color == "":
        # RL/shin 밑에는 신진서 기보만 들어갈 예정이라면 로그만 찍고 넘어가도 됨
        print(f"[WARN] Could not detect Shin color in {sgf_path} (PB='{pb}', PW='{pw}')")
        return []

    root = game.get_root()

    # 🔴 여기 수정 포인트: HA가 0보다 클 때만 진짜 핸디캡 게임으로 보고 스킵
    if skip_handicap and root.has_property("HA"):
        try:
            ha_val = root.get("HA")
            # ha_val은 보통 문자열이므로 int 변환 시도
            if isinstance(ha_val, bytes):
                ha_val = ha_val.decode("utf-8", errors="ignore")
            ha_int = int(str(ha_val))
        except Exception:
            ha_int = 0  # 파싱 실패 시 그냥 0으로 간주

        if ha_int > 0:
            # 진짜 핸디캡 게임이면 제외
            return []

    board = boards.Board(BOARD_SIZE)
    positions: List[Tuple[np.ndarray, int]] = []

    for node in game.get_main_sequence():
        color, move = node.get_move()
        if color is None:
            continue

        # 신진서 차례가 아닌 수: 보드만 업데이트하고 학습 포지션은 만들지 않음
        if color != shin_color:
            if move is None:
                try:
                    board.play_pass(color)
                except Exception:
                    pass
            else:
                row, col = move
                try:
                    board.play(row, col, color)
                except Exception:
                    pass
            continue

        # 여기부터는 신진서가 둔 수
        if move is None:
            # pass 수는 일단 제외 (원하면 포함하도록 바꿀 수 있음)
            try:
                board.play_pass(color)
            except Exception:
                pass
            continue

        row, col = move

        state = encode_board(board, next_color=color)
        action = row * BOARD_SIZE + col

        try:
            board.play(row, col, color)
        except Exception:
            continue

        positions.append((state, action))

    return positions


def build_shin_dataset_from_dir(
    root_dir: str,
    out_path: str,
    max_games: int,
    max_positions: int,
) -> None:
    root = Path(root_dir)
    sgf_files = list(root.rglob("*.sgf")) + list(root.rglob("*.SGF"))
    total_files = len(sgf_files)
    print(f"[INFO] root_dir = {root_dir}")
    print(f"[INFO] Found {total_files} SGF files")

    states = []
    actions = []
    games_used = 0
    games_with_moves = 0

    for i, sgf_path in enumerate(sgf_files, start=1):
        if max_games is not None and games_used >= max_games:
            print(f"[INFO] Reached max_games={max_games}, stopping.")
            break

        if i % 5 == 0 or i == 1:
            print(f"[INFO] Processing {i}/{total_files}: {sgf_path}")

        positions = sgf_to_shin_positions(sgf_path)
        games_used += 1

        if not positions:
            continue

        games_with_moves += 1

        for s, a in positions:
            states.append(s)
            actions.append(a)
            if max_positions is not None and len(states) >= max_positions:
                print(f"[INFO] Reached max_positions={max_positions}, stopping early.")
                break

        if max_positions is not None and len(states) >= max_positions:
            break

    print("========== Shin Dataset Summary ==========")
    print(f"Total SGF files found     : {total_files}")
    print(f"Games scanned (limited)   : {games_used}")
    print(f"Games with Shin moves     : {games_with_moves}")
    print(f"Total Shin positions      : {len(states)}")
    print("==========================================")

    if len(states) == 0:
        print("[WARN] No positions collected. Skip saving.")
        return

    states_arr = np.stack(states, axis=0)
    actions_arr = np.array(actions, np.int64)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving dataset to {out_path}")
    np.savez_compressed(out_path, states=states_arr, actions=actions_arr)
    print("[INFO] Done.")


if __name__ == "__main__":
    print("[DEBUG] sgf_shin_moves running as __main__")
    print("[DEBUG] sys.argv:", sys.argv)

    if len(sys.argv) != 5:
        print(
            "Usage:\n"
            "  python src/sgf_shin_moves.py <root_dir> <out_path> <max_games> <max_positions>\n"
            "Example:\n"
            "  python src/sgf_shin_moves.py /content/drive/MyDrive/RL/shin data/processed/shin_policy_50k.npz 2000 50000"
        )
        sys.exit(1)

    root_dir = sys.argv[1]
    out_path = sys.argv[2]
    max_games = int(sys.argv[3])
    max_positions = int(sys.argv[4])

    build_shin_dataset_from_dir(
        root_dir=root_dir,
        out_path=out_path,
        max_games=max_games,
        max_positions=max_positions,
    )
