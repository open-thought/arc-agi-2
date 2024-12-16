#/bin/bash
python -u tgi_inference.py --format chatml --cutoff_length 8096 --max_new_tokens 2048 --temperature 0 --jsonl_out t0_digits.jsonl --trials 1 --alphabet "[\"0\",\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\",\"9\"]"
#python -u tgi_inference.py --format chatml --cutoff_length 8096 --max_new_tokens 2048 --temperature 0 --jsonl_out t0_letters.jsonl --trials 1 --alphabet "[\"A\",\"B\",\"C\",\"D\",\"E\",\"F\",\"G\",\"H\",\"I\",\"J\"]"
#python -u tgi_inference.py --format chatml --cutoff_length 48576 --max_new_tokens 2048 --temperature 0 --jsonl_out t0_colors.jsonl --trials 1 --alphabet "[\"black\",\"red\",\"green\",\"blue\",\"yellow\",\"orange\",\"brown\",\"pink\",\"purple\",\"white\"]"
#python -u tgi_inference.py --format chatml --cutoff_length 48576 --max_new_tokens 2048 --temperature 0 --jsonl_out t0_icao.jsonl --trials 1 --alphabet "[\"Alfa\",\"Bravo\",\"Charlie\",\"Delta\",\"Echo\",\"Foxtrot\",\"Golf\",\"Hotel\",\"India\",\"Juliett\"]"
python -u tgi_inference.py --format chatml --cutoff_length 8096 --max_new_tokens 2048 --temperature 0.3 --jsonl_out t3_digits.jsonl --trials 3 --alphabet "[\"0\",\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\",\"9\"]"
#python -u tgi_inference.py --format chatml --cutoff_length 8096 --max_new_tokens 2048 --temperature 0.3 --jsonl_out t3_letters.jsonl --trials 3 --alphabet "[\"A\",\"B\",\"C\",\"D\",\"E\",\"F\",\"G\",\"H\",\"I\",\"J\"]"
#python -u tgi_inference.py --format chatml --cutoff_length 48576 --max_new_tokens 2048 --temperature 0.3 --jsonl_out t3_colors.jsonl --trials 3 --alphabet "[\"black\",\"red\",\"green\",\"blue\",\"yellow\",\"orange\",\"brown\",\"pink\",\"purple\",\"white\"]"
#python -u tgi_inference.py --format chatml --cutoff_length 48576 --max_new_tokens 2048 --temperature 0.3 --jsonl_out t3_icao.jsonl --trials 3 --alphabet "[\"Alfa\",\"Bravo\",\"Charlie\",\"Delta\",\"Echo\",\"Foxtrot\",\"Golf\",\"Hotel\",\"India\",\"Juliett\"]"
