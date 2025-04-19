from collect_training_data_from_fen import split_pgn_file

if __name__ == "__main__":
    split_pgn_file(
        r"sample_data\ficsgamesdb_2024_chess2000.pgn",
        r"sample_data\ficsgame_2024_chess2000"
    )