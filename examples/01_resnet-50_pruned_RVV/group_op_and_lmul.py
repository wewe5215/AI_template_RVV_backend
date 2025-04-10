import re
def fetch_lmul_for_op(batch_size, read_file="chosen_lmul_bs"):
    operator_to_lmul = {}
    weight_to_lmul = {}
    read_file = f'{read_file}{batch_size}'
    print(f"readfile = {read_file}")
    with open(f"{read_file}.md", "r") as f:
        for line in f:
            line = line.strip()
            # Check that the line is formatted with pipe delimiters.
            if line.startswith("|") and line.endswith("|"):
                parts = [part.strip() for part in line.split("|") if part.strip()]
                if len(parts) < 4:
                    print("len(parts) < 4")
                    continue
                # parts[0]: dimensions, parts[1]: operator field, parts[2]: LMUL value
                parameter_field, operators_field, dimention_field, lmul_field = parts[:4]
                # Extract operator names (split on commas)
                # operators = [op.strip() for op in operators_field.split(",") if op.strip()]
                try:
                    lmul = lmul_field
                except ValueError:
                    lmul = lmul_field  # fallback if conversion fails
                # Record each operator with its LMUL
                if 'weight' in parameter_field:
                    operator_to_lmul[operators_field] = lmul
                    weight_to_lmul[parameter_field] = lmul

    return weight_to_lmul
if __name__ == "__main__":
    weight_to_lmul = fetch_lmul_for_op(1)
    for key in weight_to_lmul:
        print(f'{key}, using {weight_to_lmul[key]}')