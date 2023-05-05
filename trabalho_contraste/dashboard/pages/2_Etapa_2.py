from typing import Dict
import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st
from PIL import Image
from trabalho_contraste.commom import (
    IMG_FOLDER,
    apply_filter,
    get_histograma,
    load_images,
    norm_hist,
    get_img_dfs,
    get_img_matrixes,
    get_metrics,
)

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
st.markdown("#### Imagens Originais")
for index, col in enumerate(cols):
    with col:
        st.image(list(images.values())[index])

# Images as dataframe
images_matrix = get_img_matrixes(images)
images_dfs = get_img_dfs(images_matrix)

st.markdown("#### Métricas")

st.dataframe(get_metrics(images_dfs, images).to_pandas())


histogramas: Dict[str, pl.DataFrame] = dict()


# Histogram
for name, df in images_dfs.items():
    histogram_df = get_histograma(df)
    histogramas[name] = histogram_df

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


norm_images: Dict[str, Image.Image] = dict()
for name, image in images.items():
    pixels = np.array(image)
    pixels_norm = apply_filter(name, pixels, conversion_funcs)
    # pixels_norm = pixels_norm.reshape(images_matrix[name].shape)
    img_norm = Image.fromarray(pixels_norm)
    norm_images[name] = img_norm

norm_images_matrix = get_img_matrixes(norm_images)
norm_images_dfs = get_img_dfs(norm_images_matrix)

histogramas_norm: Dict[str, pl.DataFrame] = dict()
st.markdown("#### Histogramas")

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
            title=f"Probabilidade Somada {name}",
            data_frame=probs_dfs[name].to_pandas(),
            x="IndexProb",
            y="Probability",
            color="Type",
        )
    )


st.markdown("#### Imagens Normalizadas")

cols = st.columns(len(norm_images))
for index, col in enumerate(cols):
    with col:
        st.image(list(norm_images.values())[index].convert("RGB"))
