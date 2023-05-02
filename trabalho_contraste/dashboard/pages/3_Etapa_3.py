from functools import reduce
from operator import mul
from typing import Dict
import numpy as np
from numpy import linalg
import plotly.express as px
import polars as pl
from skimage.color import rgb2yiq, yiq2rgb
import streamlit as st
from PIL import Image
from trabalho_contraste.dashboard.commom import (
    IMG_FOLDER,
    apply_filter,
    norm_hist,
    get_histograma,
    load_images,
    get_img_matrixes,
)

st.subheader("Etapa 3")
image_paths = [
    IMG_FOLDER / "outono_LC.png",
    IMG_FOLDER / "predios.jpeg",
]

images = load_images(image_paths, "RGB")

st.markdown("#### Imagens Originais")

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

st.markdown("#### Imagens Normalizadas sobre canais RGB")
cols = st.columns(len(norm_rgb_images))
for index, col in enumerate(cols):
    with col:
        st.image(list(norm_rgb_images.values())[index].convert("RGB"))


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
for name, df in images_dfs.items():
    image = image_matrixes[name]
    pixels = df["PixelsY"].to_numpy().reshape(image.shape[:-1])

    pixels_norm_y = apply_filter(name, pixels, conversion_yiq_funcs)

    image_arr = image.copy()
    # pixels_norm = pixels_norm.reshape(images_matrix[name].shape)
    image_arr[:, :, 0] = pixels_norm_y
    image_arr[:, :, 1] = df["PixelsI"].to_numpy().reshape(image.shape[:-1])
    image_arr[:, :, 2] = df["PixelsQ"].to_numpy().reshape(image.shape[:-1])
    image_arr = yiq2rgb(image_arr / 255)

    image_arr_m = image_arr[:, :, 0] - np.min(image_arr[:, :, 0])
    image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    image_arr[:, :, 0] = image_arr_n
    image_arr_m = image_arr[:, :, 1] - np.min(image_arr[:, :, 1])
    image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    image_arr[:, :, 1] = image_arr_n
    image_arr_m = image_arr[:, :, 2] - np.min(image_arr[:, :, 2])
    image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    image_arr[:, :, 2] = image_arr_n

    # image_arr_m = image_arr - np.min(image_arr)
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr = image_arr_n

    img_norm = Image.fromarray(image_arr.astype(np.uint8), mode="RGB")

    histogram_df = get_histograma(
        pl.DataFrame(
            data={
                "PixelsY": pixels_norm_y.reshape(reduce(mul, image_arr.shape[:-1])),
            },
            schema={"PixelsY": pl.UInt8},
        ),
        "PixelsY",
    )

    histogramas_norm_yiq[name] = histogram_df
    norm_yiq_images[name] = img_norm


st.markdown("#### Imagens Normalizadas sobre canal Y (YIQ)")
cols = st.columns(len(norm_yiq_images))
for index, col in enumerate(cols):
    with col:
        st.image(list(norm_yiq_images.values())[index].convert("RGB"))

st.markdown("#### Histogramas Y")
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

st.subheader("Implementação Própria de Conversão YIQ")


def convert_rgb_to_yiq(matriz: np.ndarray) -> np.ndarray:
    yiq_from_rgb = np.array(
        [
            [0.299, 0.587, 0.114],
            [0.59590059, -0.27455667, -0.32134392],
            [0.21153661, -0.52273617, 0.31119955],
        ]
    )
    image_arr = (matriz @ yiq_from_rgb.T.copy()).astype(np.float64)
    # image_arr_m = image_arr[:, :, 0] - np.min(image_arr[:, :, 0])
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr[:, :, 0] = image_arr_n
    # image_arr_m = image_arr[:, :, 1] - np.min(image_arr[:, :, 1])
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr[:, :, 1] = image_arr_n
    # image_arr_m = image_arr[:, :, 2] - np.min(image_arr[:, :, 2])
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr[:, :, 2] = image_arr_n
    image_arr_n = image_arr - np.min(image_arr)
    image_arr = image_arr_n / np.max(image_arr_n)
    return (255 * image_arr).astype(np.uint8)

def convert_yiq_to_rgb(matriz: np.ndarray) -> np.ndarray:
    yiq_from_rgb = np.array(
        [
            [0.299, 0.587, 0.114],
            [0.59590059, -0.27455667, -0.32134392],
            [0.21153661, -0.52273617, 0.31119955],
        ]
    )
    image_arr = (matriz @ linalg.inv(yiq_from_rgb).T.copy()).astype(np.float64)
    
    # image_arr_n = image_arr - np.min(image_arr)
    # image_arr = image_arr_n / np.max(image_arr_n)
    
    image_arr_m = image_arr[:, :, 0] - np.min(image_arr[:, :, 0])
    image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    image_arr[:, :, 0] = image_arr_n
    image_arr_m = image_arr[:, :, 1] - np.min(image_arr[:, :, 1])
    image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    image_arr[:, :, 1] = image_arr_n
    image_arr_m = image_arr[:, :, 2] - np.min(image_arr[:, :, 2])
    image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    image_arr[:, :, 2] = image_arr_n

    return (image_arr).astype(np.uint8)

def get_img_dfs_own_yiq(images_matrix: Dict[str, np.ndarray]):
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
                convert_rgb_to_yiq(image),
            )
            for name, image in images_matrix.items()
        ]
    }


images_dfs = get_img_dfs_own_yiq(images_matrix=image_matrixes)

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
for name, df in images_dfs.items():
    image = image_matrixes[name]
    pixels = df["PixelsY"].to_numpy().reshape(image.shape[:-1])

    pixels_norm_y = apply_filter(name, pixels, conversion_yiq_funcs)

    image_arr = image.copy()
    # pixels_norm = pixels_norm.reshape(images_matrix[name].shape)
    image_arr[:, :, 0] = pixels_norm_y
    image_arr[:, :, 1] = df["PixelsI"].to_numpy().reshape(image.shape[:-1])
    image_arr[:, :, 2] = df["PixelsQ"].to_numpy().reshape(image.shape[:-1])
    image_arr = convert_yiq_to_rgb(image_arr)

    # image_arr_m = image_arr[:, :, 0] - np.min(image_arr[:, :, 0])
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr[:, :, 0] = image_arr_n
    # image_arr_m = image_arr[:, :, 1] - np.min(image_arr[:, :, 1])
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr[:, :, 1] = image_arr_n
    # image_arr_m = image_arr[:, :, 2] - np.min(image_arr[:, :, 2])
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr[:, :, 2] = image_arr_n

    # image_arr_m = image_arr - np.min(image_arr)
    # image_arr_n = np.rint(255 * (image_arr_m / np.max(image_arr_m)))
    # image_arr = image_arr_n

    img_norm = Image.fromarray(image_arr.astype(np.uint8), mode="RGB")

    histogram_df = get_histograma(
        pl.DataFrame(
            data={
                "PixelsY": pixels_norm_y.reshape(reduce(mul, image_arr.shape[:-1])),
            },
            schema={"PixelsY": pl.UInt8},
        ),
        "PixelsY",
    )

    histogramas_norm_yiq[name] = histogram_df
    norm_yiq_images[name] = img_norm

st.markdown("#### Imagens Normalizadas sobre canal Y (YIQ)")
cols = st.columns(len(norm_yiq_images))
for index, col in enumerate(cols):
    with col:
        st.image(list(norm_yiq_images.values())[index].convert("RGB"))

st.markdown("#### Histogramas Y")
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
