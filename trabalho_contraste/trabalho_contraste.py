from PIL import Image
from pathlib import Path
import numpy as np
import polars as pl
from functools import reduce
from operator import mul
import plotly.express as px

# Load images
IMG_FOLDER = Path("data")
image_paths = [
    IMG_FOLDER / "figuraEscura.jpg",
    IMG_FOLDER / "figuraClara.jpg",
    IMG_FOLDER / "lena_B.png",
]

images = [Image.open(image_path).convert("L") for image_path in image_paths]
print([image.mode for image in images])

# Images as dataframe
images_matrix = [np.array(image) for image in images]
images_dfs = [
    pl.DataFrame({"Pixels": matrix.reshape(reduce(mul, matrix.shape))})
    for matrix in images_matrix
]

# Avg and Std Dev can be derive from dataframe
# And it is easier to display plots :)
# for df in images_dfs:
#     print(df.describe())

# Avg
avgs = [df["Pixels"].mean() for df in images_dfs]
print("Médias: ", avgs)

# Std Dev
stds = [df["Pixels"].std() for df in images_dfs]
print("Desvios Padrão: ", stds)

# Entropy
entropies = [image.entropy() for image in images]
print("Entropias: ", entropies)

# Histogram
for df in images_dfs:
    px.bar(
        data_frame=df.groupby("Pixels")
        .agg(pl.col("Pixels").count().alias("Qtd Pixels"))
        .to_pandas(),
        x="Pixels",
        y="Qtd Pixels",
    ).show()
