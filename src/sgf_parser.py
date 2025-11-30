import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sgfmill import sgf, boards

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

    # 턴 정보 채널
    if next_color == 'b':
        x[2, :, :] = 1.0
    else:
        x[2, :, :] = 0.0

    return x


def sgf_game_to_positions(sgf_path: Path) -> List[Tuple[np.ndarray, int]]:
    try:
        with sgf_path.open("rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
    except Exception as e:
        print(f"[WARN] Failed to parse {sgf_path}: {e}")
        return []

    size = game.get_size()
    if size != BOARD_SIZE:
        return []

    root = game.get_root()
    if root.has_property("HA"):
        # handicap 게임은 일단 제외
        return []

    board = boards.Board(BOARD_SIZE)
    positions: List[Tuple[np.ndarray, int]] = []

    for node in game.get_main_sequence():
        color, move = node.get_move()
        if color is None:
            continue

        if move is None:
            # pass 수
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
        except ValueError:
            # 이상한 수는 버리고 넘어감
            continue

        positions.append((state, action))

    return positions


def build_dataset_from_dir(
    root_dir: str,
    out_path: str,
    max_games: int,
    max_positions: int,
) -> None:
    root = Path(root_dir)
    sgf_files = list(root.rglob("*.sgf")) + list(root.rglob("*.SGF"))

    total_files_found = len(sgf_files)
    print(f"[INFO] root_dir = {root_dir}")
    print(f"[INFO] Found {total_files_found} SGF files under {root_dir}")

    if total_files_found == 0:
        print("[WARN] No SGF files found. Abort.")
        return

    if max_games is not None and max_games > 0:
        sgf_files = sgf_files[:max_games]
        print(f"[INFO] Limiting to first {len(sgf_files)} games (max_games={max_games})")

    states = []
    actions = []

    games_processed = 0
    games_with_positions = 0

    for i, sgf_path in enumerate(sgf_files, start=1):
        if i % 50 == 0 or i == 1:
            print(f"[INFO] Processing game {i}/{len(sgf_files)}: {sgf_path}")

        games_processed += 1

        positions = sgf_game_to_positions(sgf_path)

        if not positions:
            continue

        games_with_positions += 1

        for s, a in positions:
            states.append(s)
            actions.append(a)

            if max_positions is not None and len(states) >= max_positions:
                print(f"[INFO] Reached max_positions={max_positions}, stopping early.")
                break

        if max_positions is not None and len(states) >= max_positions:
            break

    print("========== Parsing Summary ==========")
    print(f"Total SGF files found      : {total_files_found}")
    print(f"Games processed            : {games_processed}")
    print(f"Games with positions       : {games_with_positions}")
    print(f"Total positions collected  : {len(states)}")
    print("=====================================")

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
    print("[DEBUG] sgf_parser running as __main__")
    print("[DEBUG] sys.argv:", sys.argv)

    if len(sys.argv) != 5:
        print(
            "Usage:\n"
            "  python3 src/sgf_parser.py <root_dir> <out_path> <max_games> <max_positions>\n"
            "Example:\n"
            "  python3 src/sgf_parser.py data/raw/cwi/games data/processed/base_policy_100k.npz 500 100000"
        )
        sys.exit(1)

    root_dir = sys.argv[1]
    out_path = sys.argv[2]
    max_games = int(sys.argv[3])
    max_positions = int(sys.argv[4])

    build_dataset_from_dir(
        root_dir=root_dir,
        out_path=out_path,
        max_games=max_games,
        max_positions=max_positions,
    )
