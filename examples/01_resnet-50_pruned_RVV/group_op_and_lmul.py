import re
def fetch_lmul_for_op():
    operator_to_lmul = {}
    with open("chosen_lmul_bs1.txt", "r") as f:
        for line in f:
            line = line.strip()
            # Check that the line is formatted with pipe delimiters.
            if line.startswith("|") and line.endswith("|"):
                parts = [part.strip() for part in line.split("|") if part.strip()]
                if len(parts) < 3:
                    continue
                # parts[0]: dimensions, parts[1]: operator field, parts[2]: LMUL value
                dims_field, operators_field, lmul_field = parts[:3]
                # Extract operator names (split on commas)
                operators = [op.strip() for op in operators_field.split(",") if op.strip()]
                try:
                    lmul = int(lmul_field)
                except ValueError:
                    lmul = lmul_field  # fallback if conversion fails
                # Record each operator with its LMUL
                for op in operators:
                    operator_to_lmul[op] = lmul

    # print("Operator to LMUL mapping from chosen_lmul_bs1.txt:")
    # print(operator_to_lmul)

    # Step 2: Parse the header file ("model-generated.h") to extract weight usage for the target operator.
    weight_to_lmul = {}

    with open("model-generated.h", "r") as f:
        header_content = f.read()

    # For each operator extracted from chosen_lmul_bs1.txt, find its function calls.
    for op, lmul in operator_to_lmul.items():
        # Use regex to match: operator_name ( ... );
        pattern = re.compile(rf"{op}\s*\((.*?)\);", re.DOTALL)
        matches = pattern.findall(header_content)
        for match in matches:
            # Split the function call arguments by comma.
            # Assumes that none of the arguments contain commas (or are split across lines in a simple way).
            args = [arg.strip() for arg in match.split(",")]
            if len(args) < 2 or "_weight" not in args[1]:
                continue
            # In the expected pattern, the weight variable is the second argument.
            weight = args[1]
            weight_to_lmul[weight] = int(lmul)

    # print("\nWeight to LMUL mapping for operators found in model-generated.h:")
    # print(weight_to_lmul)
    # for weight in weight_to_lmul:
    #     print(f"{weight}, {weight_to_lmul[weight]}")
    return weight_to_lmul