from functools import partial, reduce
from multiprocessing.pool import ThreadPool
from operator import mul
from pathlib import Path
from typing import Dict, List
import numpy as np
import plotly.express as px
import polars as pl
from typing import Union, Literal
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
def load_images(
    image_paths: List[Path], color_scheme: Literal["RGB", "L"] = "L"
) -> Dict[str, Image.Image]:
    return {
        image_path.name: Image.open(image_path).convert(color_scheme)
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


def get_histograma(
    df: pl.DataFrame, pixel_col: Union[Literal["Pixels"], str] = "Pixels"
):
    grouped_df = (
        df.groupby(pixel_col)
        .agg(pl.col(pixel_col).count().alias("Qtd Pixels").cast(pl.UInt64))
        .with_columns([pl.col(pixel_col).cast(pl.UInt8)])
    )

    non_existent_gray_values = np.array(
        [grey for grey in range(256) if grey not in grouped_df[pixel_col].unique()],
        dtype=np.uint8,
    )

    missing_df = pl.DataFrame(
        {
            pixel_col: non_existent_gray_values,
            "Qtd Pixels": np.zeros(non_existent_gray_values.shape),
        },
        schema={pixel_col: pl.UInt8, "Qtd Pixels": pl.UInt64},
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


def _apply_filter(pixels: np.ndarray, name: str, func: Dict[str, np.ndarray]):
    st.write(pixels)
    result = np.array(list(map(lambda x: func[name][int(x)], pixels)))
    return result


@st.cache_data
def apply_filter(
    name: str, pixels: np.ndarray, conversion_rgb_funcs: Dict[str, np.ndarray]
):
    with ThreadPool() as pool:
        result = pool.map(
            partial(_apply_filter, name=name, func=conversion_rgb_funcs), pixels
        )
    return np.array(result)


norm_images: Dict[str, Image.Image] = dict()
for name, image in images.items():
    pixels = np.array(image)
    pixels_norm = apply_filter(name, pixels, conversion_funcs)
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

st.subheader("Etapa 3")
image_paths = [
    IMG_FOLDER / "outono_LC.png",
    IMG_FOLDER / "predios.jpeg",
]

images = load_images(image_paths, "RGB")

cols = st.columns(len(images))

for index, col in enumerate(cols):
    with col:
        st.image(list(images.values())[index])

image_matrixes = get_img_matrixes(images)


def get_img_dfs_rgb(images_matrix: Dict[str, np.ndarray]):
    return {
        name: pl.DataFrame(
            data={
                "PixelsR": matrix[:, :, 0].reshape(reduce(mul, matrix.shape[:-1])),
                "PixelsG": matrix[:, :, 1].reshape(reduce(mul, matrix.shape[:-1])),
                "PixelsB": matrix[:, :, 2].reshape(reduce(mul, matrix.shape[:-1])),
            },
            schema={"PixelsR": pl.UInt8, "PixelsG": pl.UInt8, "PixelsB": pl.UInt8},
        )
        for name, matrix in images_matrix.items()
    }


images_dfs = get_img_dfs_rgb(image_matrixes)
histogramas_rgb: Dict[str, Dict[str, pl.DataFrame]] = dict()

for name, df in images_dfs.items():
    histogram_df_red = get_histograma(df, "PixelsR")
    histogram_df_blue = get_histograma(df, "PixelsB")
    histogram_df_green = get_histograma(df, "PixelsG")
    histogramas_rgb[f"{name}"] = {
        "red": histogram_df_red,
        "green": histogram_df_green,
        "blue": histogram_df_blue,
    }

conversion_rgb_funcs: Dict[str, np.ndarray] = dict()
probs_rgb_dfs: Dict[str, Dict[str, pl.DataFrame]] = dict()
for name, histogram_df in histogramas_rgb.items():
    probs_df_red = norm_hist(
        histogram_df["red"].sort("PixelsR", descending=False)["Qtd Pixels"].to_numpy()
    )
    probs_df_green = norm_hist(
        histogram_df["green"].sort("PixelsG", descending=False)["Qtd Pixels"].to_numpy()
    )
    probs_df_blue = norm_hist(
        histogram_df["blue"].sort("PixelsB", descending=False)["Qtd Pixels"].to_numpy()
    )

    probs_rgb_dfs[name] = {
        "red": probs_df_red,
        "green": probs_df_green,
        "blue": probs_df_blue,
    }

    probs_rounded_red_df = probs_df_red.filter(pl.col("Type") == "Rounded Sum")
    probs_rounded_green_df = probs_df_green.filter(pl.col("Type") == "Rounded Sum")
    probs_rounded_blue_df = probs_df_blue.filter(pl.col("Type") == "Rounded Sum")

    conversion_rgb_funcs[f"{name} red"] = probs_rounded_red_df["Probability"].to_numpy()
    conversion_rgb_funcs[f"{name} green"] = probs_rounded_green_df[
        "Probability"
    ].to_numpy()
    conversion_rgb_funcs[f"{name} blue"] = probs_rounded_blue_df[
        "Probability"
    ].to_numpy()

norm_rgb_images: Dict[str, Image.Image] = dict()
for name, image in images.items():
    pixels = np.array(image)
    pixels_norm_red = apply_filter(f"{name} red", pixels[:, :, 0], conversion_rgb_funcs)
    pixels_norm_green = apply_filter(
        f"{name} green", pixels[:, :, 1], conversion_rgb_funcs
    )
    pixels_norm_blue = apply_filter(
        f"{name} blue", pixels[:, :, 2], conversion_rgb_funcs
    )
    image_arr = np.zeros((*pixels_norm_blue.shape, 3), dtype=np.uint8)
    # pixels_norm = pixels_norm.reshape(images_matrix[name].shape)
    image_arr[:, :, 0] = pixels_norm_red
    image_arr[:, :, 1] = pixels_norm_green
    image_arr[:, :, 2] = pixels_norm_blue

    img_norm = Image.fromarray(image_arr)
    norm_rgb_images[name] = img_norm

cols = st.columns(len(norm_rgb_images))

for index, col in enumerate(cols):
    with col:
        st.image(list(norm_rgb_images.values())[index].convert("RGB"))

from skimage.color import rgb2yiq, yiq2rgb


def convert_to_yiq():
    ...


def get_img_dfs_yiq(images_matrix: Dict[str, np.ndarray]):
    return {
        name: pl.DataFrame(
            data={
                "PixelsY": matrix[:, :, 0].reshape(reduce(mul, matrix.shape[:-1])),
                "PixelsI": matrix[:, :, 1].reshape(reduce(mul, matrix.shape[:-1])),
                "PixelsQ": matrix[:, :, 2].reshape(reduce(mul, matrix.shape[:-1])),
            },
            schema={"PixelsY": pl.UInt8, "PixelsI": pl.UInt8, "PixelsQ": pl.UInt8},
        )
        for name, matrix in [
            (
                name,
                (
                    255
                    * (
                        rgb2yiq(np.array(image) / 255)
                        - np.min(rgb2yiq(np.array(image) / 255))
                        / np.max(
                            rgb2yiq(np.array(image) / 255)
                            - np.min(rgb2yiq(np.array(image) / 255))
                        )
                    )
                ).astype(np.uint8),
            )
            for name, image in images_matrix.items()
        ]
    }


images_dfs = get_img_dfs_yiq(image_matrixes)

histogramas_yiq: Dict[str, pl.DataFrame] = dict()
for name, df in images_dfs.items():
    histogram_df = get_histograma(df, "PixelsY")
    histogramas_yiq[name] = histogram_df

conversion_yiq_funcs: Dict[str, np.ndarray] = dict()
probs_yiq_dfs: Dict[str, pl.DataFrame] = dict()
for name, histogram_df in histogramas_yiq.items():
    probs_df_y = norm_hist(
        histogram_df.sort("PixelsY", descending=False)["Qtd Pixels"].to_numpy()
    )

    probs_yiq_dfs[name] = probs_df_y

    probs_rounded_y_df = probs_df_y.filter(pl.col("Type") == "Rounded Sum")

    conversion_yiq_funcs[name] = probs_rounded_y_df["Probability"].to_numpy()


histogramas_norm_yiq: Dict[str, pl.DataFrame] = dict()
norm_yiq_images: Dict[str, Image.Image] = dict()
for name, image in images.items():
    pixels = rgb2yiq(np.array(image) / 255)
    pixels_m = pixels[:, :, 0] - np.min(pixels[:, :, 0])
    pixels_n = 255 * (pixels_m / np.max(pixels_m))
    # st.write(np.rint(pixels_n))
    pixels_norm_y = apply_filter(name, np.rint(pixels_n), conversion_yiq_funcs)

    image_arr = pixels.copy()
    # pixels_norm = pixels_norm.reshape(images_matrix[name].shape)
    image_arr[:, :, 0] = pixels_norm_y
    image_arr = yiq2rgb(image_arr / 255)

    # image_arr_m = image_arr[:, :, 0] - np.min(image_arr[:, :, 0])
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr[:, :, 0] = image_arr_n
    # image_arr_m = image_arr[:, :, 1] - np.min(image_arr[:, :, 1])
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr[:, :, 1] = image_arr_n
    # image_arr_m = image_arr[:, :, 2] - np.min(image_arr[:, :, 2])
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr[:, :, 2] = image_arr_n

    image_arr_m = image_arr - np.min(image_arr)
    image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))

    st.write(image_arr_n.shape)
    img_norm = Image.fromarray(image_arr_n.astype(np.uint8), mode="RGB")

    histogram_df = get_histograma(
        pl.DataFrame(
            data={
                "PixelsY": pixels_norm_y.reshape(reduce(mul, image_arr_n.shape[:-1])),
            },
            schema={"PixelsY": pl.UInt8},
        ),
        "PixelsY",
    )

    histogramas_norm_yiq[name] = histogram_df
    norm_yiq_images[name] = img_norm

for name, df in histogramas_yiq.items():
    histogram_norm_yiq = histogramas_norm_yiq[name]
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.bar(
                title=f"Histograma {name}",
                data_frame=df.to_pandas(),
                x="PixelsY",
                y="Qtd Pixels",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            px.bar(
                title=f"Histograma {name} Normalizado",
                data_frame=histogram_norm_yiq.to_pandas(),
                x="PixelsY",
                y="Qtd Pixels",
            ),
            use_container_width=True,
        )
    st.plotly_chart(
        px.line(
            data_frame=probs_yiq_dfs[name].to_pandas(),
            x="IndexProb",
            y="Probability",
            color="Type",
        )
    )

cols = st.columns(len(norm_yiq_images))

for index, col in enumerate(cols):
    with col:
        st.image(list(norm_yiq_images.values())[index].convert("RGB"))
