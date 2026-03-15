import json

nb_path = r'c:\Users\soume\Dev\Predicting_Movie_Success\Movie_Predictor.ipynb'
py_path = r'c:\Users\soume\Dev\Predicting_Movie_Success\movie_success_predictor.py'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(py_path, 'w', encoding='utf-8') as out:
    out.write("# Auto-generated from Movie_Predictor.ipynb\n\n")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            out.write(source)
            out.write("\n\n")

print("Conversion complete.")
