import streamlit as st

from functools import reduce
from operator import mul
from pathlib import Path

import numpy as np
import plotly.express as px
import polars as pl
from PIL import Image

st.title("Contraste - Operação sobre Pixel")

st.subheader("Etapa 1")

# Load images
IMG_FOLDER = Path("data")
image_paths = [
    IMG_FOLDER / "figuraEscura.jpg",
    IMG_FOLDER / "figuraClara.jpg",
    IMG_FOLDER / "lena_B.png",
]


@st.cache_data
def load_images():
    return {
        image_path.name: Image.open(image_path).convert("L")
        for image_path in image_paths
    }


images = load_images()

# st.write([image.mode for image in images])
cols = st.columns(len(images))

for index, col in enumerate(cols):
    with col:
        st.image(list(images.values())[index])

# Images as dataframe
images_matrix = {name: np.array(image) for name, image in images.items()}
images_dfs = {
    name: pl.DataFrame(
        {"Pixels": matrix.reshape(reduce(mul, matrix.shape))},
        schema={"Pixels": pl.UInt8},
    )
    for name, matrix in images_matrix.items()
}

# Avg and Std Dev can be derive from dataframe
# And it is easier to display plots :)
# for df in images_dfs:
#     print(df.describe())

st.markdown("### Métricas")

# Avg
avgs = {name: df["Pixels"].mean() for name, df in images_dfs.items()}
st.dataframe(pl.DataFrame({"Imagem": avgs.keys(), "Média": avgs.values()}).to_pandas())
# Std Dev
stds = {name: df["Pixels"].std() for name, df in images_dfs.items()}
st.dataframe(
    pl.DataFrame({"Imagem": stds.keys(), "Desvio Padrão": stds.values()}).to_pandas()
)

# Entropy
entropies = {name: image.entropy() for name, image in images.items()}
st.dataframe(
    pl.DataFrame(
        {"Imagem": entropies.keys(), "Entropia": entropies.values()}
    ).to_pandas()
)

st.markdown("### Histogramas")

# Histogram
for name, df in images_dfs.items():
    grouped_df = df.groupby("Pixels").agg(
        pl.col("Pixels").count().alias("Qtd Pixels").cast(pl.UInt64)
    )
    non_existent_gray_values = np.array(
        [grey for grey in range(256) if grey not in grouped_df["Pixels"].unique()],
        dtype=np.uint8,
    )
    missing_df = pl.DataFrame(
        {
            "Pixels": non_existent_gray_values,
            "Qtd Pixels": np.zeros(non_existent_gray_values.shape),
        },
        schema={"Pixels": pl.UInt8, "Qtd Pixels": pl.UInt64},
    )
    histogram_df = pl.concat([grouped_df, missing_df], how="vertical")
    st.plotly_chart(
        px.bar(
            title=f"Histograma {name}",
            data_frame=histogram_df.to_pandas(),
            x="Pixels",
            y="Qtd Pixels",
        )
    )

# -------------------------------------- Parte 2 --------------------------------------
st.subheader("Etapa 2")

# Load Images
image_paths = [
    IMG_FOLDER / "figuraEscura.jpg",
    IMG_FOLDER / "figuraClara.jpg",
    IMG_FOLDER / "xadrez_lowCont.png",
    IMG_FOLDER / "marilyn.jpg",
]

images = {
    image_path.name: Image.open(image_path).convert("L") for image_path in image_paths
}

cols = st.columns(len(images))

for index, col in enumerate(cols):
    with col:
        st.image(list(images.values())[index])

# Images as dataframe
images_matrix = {name: np.array(image) for name, image in images.items()}
images_dfs = {
    name: pl.DataFrame(
        {"Pixels": matrix.reshape(reduce(mul, matrix.shape))},
        schema={"Pixels": pl.UInt8},
    )
    for name, matrix in images_matrix.items()
}

st.markdown("### Métricas")

# Avg
avgs = {name: df["Pixels"].mean() for name, df in images_dfs.items()}
st.dataframe(pl.DataFrame({"Imagem": avgs.keys(), "Média": avgs.values()}).to_pandas())
# Std Dev
stds = {name: df["Pixels"].std() for name, df in images_dfs.items()}
st.dataframe(
    pl.DataFrame({"Imagem": stds.keys(), "Desvio Padrão": stds.values()}).to_pandas()
)

# Entropy
entropies = {name: image.entropy() for name, image in images.items()}
st.dataframe(
    pl.DataFrame(
        {"Imagem": entropies.keys(), "Entropia": entropies.values()}
    ).to_pandas()
)


@st.cache_data
def norm_hist(histograma: np.ndarray):
    probabilities = histograma / histograma.sum()
    summed_probs = np.array(
        reduce(lambda old, new: [*old, old[-1] + new], probabilities, [0])
    )

    summed_probs_rounded = np.rint(255 * summed_probs)

    probs_df = pl.DataFrame(
        {
            "IndexProb": [
                *list(range(summed_probs.shape[0])),
                *list(range(summed_probs_rounded.shape[0])),
            ],
            "Probability": [*(255 * summed_probs), *summed_probs_rounded],
            "Type": [
                *["Sum" for _ in range(summed_probs.shape[0])],
                *["Rounded Sum" for _ in range(summed_probs_rounded.shape[0])],
            ],
        }
    )

    px.line(
        data_frame=probs_df.to_pandas(),
        x="IndexProb",
        y="Probability",
        color="Type",
    )

    return probs_df


# Histogram
for name, df in images_dfs.items():
    grouped_df = df.groupby("Pixels").agg(
        pl.col("Pixels").count().alias("Qtd Pixels").cast(pl.UInt64)
    )

    non_existent_gray_values = np.array(
        [grey for grey in range(256) if grey not in grouped_df["Pixels"].unique()],
        dtype=np.uint8,
    )

    missing_df = pl.DataFrame(
        {
            "Pixels": non_existent_gray_values,
            "Qtd Pixels": np.zeros(non_existent_gray_values.shape),
        },
        schema={"Pixels": pl.UInt8, "Qtd Pixels": pl.UInt64},
    )

    histogram_df = pl.concat([grouped_df, missing_df], how="vertical")

    px.bar(
        data_frame=histogram_df.to_pandas(),
        x="Pixels",
        y="Qtd Pixels",
    )

    probs_df = norm_hist(
        histogram_df.sort("Pixels", descending=False)["Qtd Pixels"].to_numpy()
    )
    probs_rounded_df = probs_df.filter(pl.col("Type") == "Rounded Sum")
    probs_rounded = probs_rounded_df.to_pandas()

    def convert_pixel(value):
        return probs_rounded["Probability"][value]

    # print(df.with_columns([pl.col("Pixels").apply(convert_pixel).alias("NormPixels")]))
    vec_conv_pixel = np.vectorize(convert_pixel)
    pixels = df["Pixels"].to_numpy()
    print(pixels.shape)
    pixels_norm: np.ndarray = vec_conv_pixel(pixels)
    pixels_norm = pixels_norm.reshape(images_matrix[name].shape)
    img_norm = Image.fromarray(pixels_norm)
    # img_norm.show()
    st.image(img_norm.convert("RGB"))

    # TODO(Otavio): Still need to reapply metrics, and redo histograms
