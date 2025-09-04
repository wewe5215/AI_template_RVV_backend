import re
import pathlib
def fetch_lmul_and_tile(summary_file: str = "profile_summary") -> tuple[dict[str, int], dict[str, int]]:
    md_path = pathlib.Path(f"{summary_file}.md")
    weight_to_lmul,   weight_to_tile   = {}, {}
    with md_path.open(encoding="utf-8") as fp:
        for raw in fp:
            raw = raw.rstrip()
            if not raw.startswith("|") or raw.startswith("|---"):
                continue
            cols = [c.strip() for c in raw.split("|")[1:-1]]
            if len(cols) != 5:
                continue

            parameter, operator, *_dimension, lmul_str, tile_str = cols

            try:
                lmul       = int(lmul_str)
                tile_size  = int(tile_str)
            except ValueError:        # non-numeric columns – ignore row
                continue

            weight_to_lmul[parameter]  = lmul
            weight_to_tile[parameter]  = tile_size

    return weight_to_lmul, weight_to_tile
if __name__ == "__main__":
    weight_to_lmul, weight_to_tile = fetch_lmul_and_tile()
    for key in weight_to_lmul:
        print(f'{key}, using lmul = {weight_to_lmul[key]}')
    for key in weight_to_tile:
        print(f'{key}, using tile size = {weight_to_tile[key]}')