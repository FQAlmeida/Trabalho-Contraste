import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st
from trabalho_contraste.dashboard.commom import (
    load_images,
    get_img_matrixes,
    IMG_FOLDER,
    get_img_dfs,
    get_metrics,
)

st.title("Contraste - Operação sobre Pixel")

st.subheader("Etapa 1")

# Load images
image_paths = [
    IMG_FOLDER / "figuraEscura.jpg",
    IMG_FOLDER / "figuraClara.jpg",
    IMG_FOLDER / "lena_B.png",
]


images = load_images(image_paths=image_paths)
# st.write([image.mode for image in images])

st.markdown("#### Imagens Originais")

cols = st.columns(len(images))
for index, col in enumerate(cols):
    with col:
        st.image(list(images.values())[index])


# Images as dataframe


images_matrix = get_img_matrixes(images)


images_dfs = get_img_dfs(images_matrix)

# Avg and Std Dev can be derive from dataframe
# And it is easier to display plots :)
# for df in images_dfs:
#     print(df.describe())

st.markdown("#### Métricas")


st.dataframe(get_metrics(images_dfs, images).to_pandas())

st.markdown("#### Histogramas")

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
