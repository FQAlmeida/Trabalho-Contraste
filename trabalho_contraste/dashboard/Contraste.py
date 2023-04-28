from functools import partial, reduce
from multiprocessing.pool import ThreadPool
from operator import mul
from pathlib import Path
from typing import Dict, List
import numpy as np
import plotly.express as px
import polars as pl
from typing import Callable
import streamlit as st
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
def load_images(image_paths: List[Path]) -> Dict[str, Image.Image]:
    return {
        image_path.name: Image.open(image_path).convert("L")
        for image_path in image_paths
    }


images = load_images(image_paths=image_paths)
# st.write([image.mode for image in images])

cols = st.columns(len(images))
for index, col in enumerate(cols):
    with col:
        st.image(list(images.values())[index])


# Images as dataframe
@st.cache_data
def get_img_matrixes(images: Dict[str, Image.Image]):
    return {name: np.array(image) for name, image in images.items()}


images_matrix = get_img_matrixes(images)


def get_img_dfs(images_matrix: Dict[str, np.ndarray]):
    return {
        name: pl.DataFrame(
            data={"Pixels": matrix.reshape(reduce(mul, matrix.shape))},
            schema={"Pixels": pl.UInt8},
        )
        for name, matrix in images_matrix.items()
    }


images_dfs = get_img_dfs(images_matrix)

# Avg and Std Dev can be derive from dataframe
# And it is easier to display plots :)
# for df in images_dfs:
#     print(df.describe())

st.markdown("### Métricas")


def get_metrics(images_dfs: Dict[str, pl.DataFrame], images: Dict[str, Image.Image]):
    return pl.DataFrame(
        data={
            "Nomes": list(images_dfs.keys()),
            "Médias": [df["Pixels"].mean() for df in images_dfs.values()],
            "Desvios": [df["Pixels"].std() for df in images_dfs.values()],
            "Entropias": [image.entropy() for image in images.values()],
        }
    )


st.dataframe(get_metrics(images_dfs, images).to_pandas())

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

images = load_images(image_paths)

cols = st.columns(len(images))

for index, col in enumerate(cols):
    with col:
        st.image(list(images.values())[index])

# Images as dataframe
images_matrix = get_img_matrixes(images)
images_dfs = get_img_dfs(images_matrix)

st.markdown("### Métricas")

st.dataframe(get_metrics(images_dfs, images).to_pandas())


@st.cache_data
def norm_hist(histograma: np.ndarray):
    probabilities = histograma / histograma.sum()
    summed_probs = np.array(
        list(reduce(lambda old, new: [*old, old[-1] + new], probabilities, [0]))[1:]
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

    return probs_df


histogramas: Dict[str, pl.DataFrame] = dict()


def get_histograma(df: pl.DataFrame):
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
    return histogram_df


# Histogram
for name, df in images_dfs.items():
    histogram_df = get_histograma(df)
    histogramas[name] = histogram_df

    # st.plotly_chart(
    #     px.bar(
    #         title=f"Histograma {name}",
    #         data_frame=histogram_df.to_pandas(),
    #         x="Pixels",
    #         y="Qtd Pixels",
    #     )
    # )

counter = 0
conversion_funcs: Dict[str, np.ndarray] = dict()
probs_dfs: Dict[str, pl.DataFrame] = dict()
for name, histogram_df in histogramas.items():
    probs_df = norm_hist(
        histogram_df.sort("Pixels", descending=False)["Qtd Pixels"].to_numpy()
    )
    probs_dfs[name] = probs_df

    probs_rounded_df = probs_df.filter(pl.col("Type") == "Rounded Sum")
    probs_rounded = probs_rounded_df.to_pandas()

    conversion_funcs[name] = probs_rounded["Probability"].to_numpy()


def _apply_filter(pixels: np.ndarray, name: str):
    vec_conv_pixel = conversion_funcs[name]
    result = np.array(list(map(lambda x: vec_conv_pixel[x], pixels)))
    return result


@st.cache_data
def apply_filter(name: str, pixels: np.ndarray):
    with ThreadPool() as pool:
        result = pool.map(partial(_apply_filter, name=name), pixels)
    return np.array(result)


norm_images: Dict[str, Image.Image] = dict()
for name, image in images.items():
    pixels = np.array(image)
    pixels_norm = apply_filter(name, pixels)
    counter += 1
    # pixels_norm = pixels_norm.reshape(images_matrix[name].shape)
    img_norm = Image.fromarray(pixels_norm)
    norm_images[name] = img_norm
    # # img_norm.show()
    # st.image(img_norm.convert("RGB"))

    # TODO(Otavio): Still need to reapply metrics, and redo histograms

norm_images_matrix = get_img_matrixes(norm_images)
norm_images_dfs = get_img_dfs(norm_images_matrix)

histogramas_norm: Dict[str, pl.DataFrame] = dict()
for name, df in norm_images_dfs.items():
    histogram_df_norm = get_histograma(df)
    histogram_df = histogramas[name]
    histogramas_norm[name] = histogram_df_norm

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.bar(
                title=f"Histograma {name}",
                data_frame=histogram_df.to_pandas(),
                x="Pixels",
                y="Qtd Pixels",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            px.bar(
                title=f"Histograma {name} Normalizado",
                data_frame=histogram_df_norm.to_pandas(),
                x="Pixels",
                y="Qtd Pixels",
            ),
            use_container_width=True,
        )
    st.plotly_chart(
        px.line(
            data_frame=probs_dfs[name].to_pandas(),
            x="IndexProb",
            y="Probability",
            color="Type",
        )
    )

cols = st.columns(len(norm_images))

for index, col in enumerate(cols):
    with col:
        st.image(list(norm_images.values())[index].convert("RGB"))
