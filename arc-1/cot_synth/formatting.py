def format_board(
    board: list[list[int]],
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
    array_brackets: bool = True,
    with_board_dim: bool = True,
) -> str:
    h, w = len(board), len(board[0])
    buffer = []

    if with_board_dim:
        buffer.append(f"{h}x{w}\n")

    if array_brackets:
        buffer.append(f"[")
        for row in range(h):
            if row > 0 and row_delimiter:
                buffer.append(row_delimiter)
            buffer.append("[")
            for col in range(w):
                if col > 0 and col_delimiter:
                    buffer.append(col_delimiter)
                value = board[row][col]
                buffer.append(alphabet[value])
            buffer.append("]")
        buffer.append("]")
    else:
        for row in range(h):
            if row > 0 and row_delimiter:
                buffer.append(row_delimiter)
            for col in range(w):
                if col > 0 and col_delimiter:
                    buffer.append(col_delimiter)
                value = board[row][col]
                buffer.append(alphabet[value])

    return "".join(buffer)


def format_board_pair(
    index: int,
    pair: dict,
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
    array_brackets: bool = True,
) -> str:
    input = format_board(
        pair["input"],
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
        array_brackets=array_brackets,
    )
    output = format_board(
        pair["output"],
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
        array_brackets=array_brackets,
    )
    return f"input{index}: {input}\n\noutput{index}: {output}\n\n"


def format_training_examples(
    train_pairs: list[dict],
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
    array_brackets: bool = True,
) -> str:
    buffer = []
    for i, board_pair in enumerate(train_pairs):
        s = format_board_pair(
            index=i,
            pair=board_pair,
            alphabet=alphabet,
            col_delimiter=col_delimiter,
            row_delimiter=row_delimiter,
            array_brackets=array_brackets,
        )
        buffer.append(s)

    return "".join(buffer)
