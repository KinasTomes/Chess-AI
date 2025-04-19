import os
import bz2

def extract_bz_file_to_pgn(input_file_path: str, output_file_path: str, output_file_name: str) -> str:
    try:
        if not output_file_name.endswith(".pgn"):
            output_file_name += ".pgn"

        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        output_full_path = os.path.join(output_file_path, output_file_name)

        with bz2.open(input_file_path, 'rb') as f_in, open(output_full_path, 'wb') as f_out:
            f_out.write(f_in.read())

        print(f"✅ Extracted to {output_full_path}")
        return output_full_path

    except Exception as e:
        print(f"❌ Failed to extract: {e}")
        return ""
