from multiprocessing.pool import ThreadPool
from pathlib import Path
from PIL import Image
from typing import List, Literal, Dict, Union
import numpy as np
import polars as pl
from functools import partial, reduce
from operator import mul

IMG_FOLDER = Path("data")


def load_images(
    image_paths: List[Path], color_scheme: Literal["RGB", "L"] = "L"
) -> Dict[str, Image.Image]:
    return {
        image_path.name: Image.open(image_path).convert(color_scheme)
        for image_path in image_paths
    }


def get_img_matrixes(images: Dict[str, Image.Image]):
    return {name: np.array(image) for name, image in images.items()}


def get_img_dfs(images_matrix: Dict[str, np.ndarray]):
    return {
        name: pl.DataFrame(
            data={"Pixels": matrix.reshape(reduce(mul, matrix.shape))},
            schema={"Pixels": pl.UInt8},
        )
        for name, matrix in images_matrix.items()
    }


def get_metrics(images_dfs: Dict[str, pl.DataFrame], images: Dict[str, Image.Image]):
    return pl.DataFrame(
        data={
            "Nomes": list(images_dfs.keys()),
            "Médias": [df["Pixels"].mean() for df in images_dfs.values()],
            "Desvios": [df["Pixels"].std() for df in images_dfs.values()],
            "Entropias": [image.entropy() for image in images.values()],
        }
    )


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


def _apply_filter(pixels: np.ndarray, name: str, func: Dict[str, np.ndarray]):
    result = np.array(list(map(lambda x: func[name][int(x)], pixels)))
    return result


def apply_filter(
    name: str, pixels: np.ndarray, conversion_rgb_funcs: Dict[str, np.ndarray]
):
    with ThreadPool() as pool:
        result = pool.map(
            partial(_apply_filter, name=name, func=conversion_rgb_funcs), pixels
        )
    return np.array(result)
