"""Retreino reproduzível do Skin Analyser no HAM10000.

Reimplementa, num arquivo só, o protocolo do system.py: backbones do ImageNet
congelados em transfer learning, mesma entrada 224x224 para todos, augmentation,
dropout, L2, early stopping e redução do learning rate em platô. Duas diferenças
em relação ao notebook original, ambas para a métrica sair honesta:

- A separação treino/validação/teste é por lesão (lesion_id), estratificada por
  classe. O HAM10000 tem várias imagens da mesma lesão e não publica identificador
  de paciente; o system.py separava imagem a imagem.
- Cada backbone recebe o pré-processamento do próprio Keras (preprocess_input),
  em vez de dividir tudo por 255.

No Kaggle: crie um notebook, adicione o dataset "kmader/skin-cancer-mnist-ham10000",
ligue a GPU e rode
    !python treino_ham10000.py
O dataset é encontrado sozinho em /kaggle/input. Cada execução grava numa pasta
própria, ./runs/<data e hora>: metrics.csv (uma linha por arquitetura), relatório e
matriz de confusão por arquitetura, split.csv com a partição de cada imagem e
ambiente.json com as versões usadas. A pasta results/ do repositório guarda só os
resultados publicados e nunca é sobrescrita.

Fora do Kaggle: pip install -r kaggle/requirements.txt e aponte --data-dir para a
pasta com o HAM10000_metadata.csv e as imagens .jpg (em qualquer subpasta).
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

apps = tf.keras.applications

# Ordem fixa das 7 classes do HAM10000 (coluna dx do metadata)
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# Classe de maior risco clínico, cujo recall é priorizado
CLASSE_RISCO = "mel"

BACKBONES = {
    "mobilenetv2": (apps.MobileNetV2, apps.mobilenet_v2.preprocess_input),
    "efficientnetb0": (apps.EfficientNetB0, apps.efficientnet.preprocess_input),
    "resnet50": (apps.ResNet50, apps.resnet50.preprocess_input),
    "inceptionv3": (apps.InceptionV3, apps.inception_v3.preprocess_input),
    "densenet121": (apps.DenseNet121, apps.densenet.preprocess_input),
    "xception": (apps.Xception, apps.xception.preprocess_input),
    "vgg19": (apps.VGG19, apps.vgg19.preprocess_input),
    "nasnetmobile": (apps.NASNetMobile, apps.nasnet.preprocess_input),
}


def arquivos(raiz: Path):
    """Todos os arquivos sob raiz, seguindo links simbólicos (o Path.rglob não segue)."""
    for pasta, _, nomes in os.walk(raiz, followlinks=True):
        for nome in sorted(nomes):
            yield Path(pasta) / nome


def achar_dataset() -> Path:
    """No Kaggle, o caminho de montagem do dataset muda entre versões da interface."""
    achado = next((p for p in arquivos(Path("/kaggle/input")) if p.name == "HAM10000_metadata.csv"), None)
    return achado.parent if achado else Path("data")


def args_da_linha_de_comando():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=None,
                   help="pasta com o HAM10000_metadata.csv (padrão: procura em /kaggle/input, depois ./data)")
    p.add_argument("--archs", nargs="+", default=["efficientnetb0", "densenet121", "resnet50"],
                   choices=sorted(BACKBONES))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out", type=Path, default=None,
                   help="pasta de saída (padrão: runs/<data e hora>)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="usa só as N primeiras imagens (teste rápido do script)")
    return p.parse_args()


def carregar_metadados(data_dir: Path, limit: int | None) -> pd.DataFrame:
    meta = pd.read_csv(data_dir / "HAM10000_metadata.csv")
    # O dataset do Kaggle traz as imagens duplicadas em pastas com e sem maiúsculas;
    # o mapa por nome de arquivo fica com uma cópia de cada.
    caminhos = {}
    for p in arquivos(data_dir):
        if p.suffix == ".jpg":
            caminhos.setdefault(p.stem, str(p))
    meta["path"] = meta["image_id"].map(caminhos)
    faltando = meta["path"].isna().sum()
    if faltando:
        raise FileNotFoundError(f"{faltando} imagens do metadata não foram encontradas em {data_dir}")
    meta["label"] = meta["dx"].map({c: i for i, c in enumerate(CLASSES)})
    if limit:
        meta = meta.head(limit)
    return meta.reset_index(drop=True)


def separar_por_lesao(meta: pd.DataFrame, seed: int) -> pd.DataFrame:
    """64% treino, 16% validação e 20% teste, sem lesão repetida entre partições."""
    meta = meta.copy()
    meta["split"] = "train"
    externo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    resto_idx, teste_idx = next(externo.split(meta, meta["label"], groups=meta["lesion_id"]))
    meta.loc[teste_idx, "split"] = "test"

    resto = meta.iloc[resto_idx]
    interno = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, val_rel = next(interno.split(resto, resto["label"], groups=resto["lesion_id"]))
    meta.loc[resto.index[val_rel], "split"] = "val"

    por_lesao = meta.groupby("lesion_id")["split"].nunique()
    assert (por_lesao == 1).all(), "lesão presente em mais de uma partição"
    return meta


def dataset(df: pd.DataFrame, img_size: int, batch: int, treino: bool, seed: int) -> tf.data.Dataset:
    def ler(caminho, rotulo):
        img = tf.io.decode_jpeg(tf.io.read_file(caminho), channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        return tf.cast(img, tf.uint8), rotulo

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    # Cache em uint8 (o dataset inteiro cabe em ~1,5 GB); o float32 só existe por lote
    ds = ds.map(ler, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if treino:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda img, rotulo: (tf.cast(img, tf.float32), rotulo))
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def montar_modelo(arch: str, img_size: int) -> tf.keras.Model:
    backbone_cls, preprocess = BACKBONES[arch]
    backbone = backbone_cls(include_top=False, weights="imagenet",
                            input_shape=(img_size, img_size, 3), pooling="avg")
    backbone.trainable = False

    entrada = tf.keras.Input((img_size, img_size, 3))
    # Augmentation só age no treino; na inferência essas camadas passam a imagem direto
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(entrada)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.Lambda(preprocess, name="preprocess")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    saida = tf.keras.layers.Dense(len(CLASSES), activation="softmax",
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    modelo = tf.keras.Model(entrada, saida, name=arch)
    modelo.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def avaliar(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = probs.argmax(axis=1)
    rotulos = list(range(len(CLASSES)))
    risco = CLASSES.index(CLASSE_RISCO)
    presentes = np.unique(y_true)
    # AUC só é definida para classes presentes no teste (sempre todas, fora do --limit)
    auc_macro = (roc_auc_score(y_true, probs[:, presentes] / probs[:, presentes].sum(1, keepdims=True),
                               multi_class="ovr", labels=presentes)
                 if len(presentes) > 2 else float("nan"))
    auc_risco = (roc_auc_score(y_true == risco, probs[:, risco])
                 if 0 < (y_true == risco).sum() < len(y_true) else float("nan"))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=rotulos, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=rotulos, average="weighted", zero_division=0),
        f"recall_{CLASSE_RISCO}": recall_score(y_true, y_pred, labels=[risco], average="macro", zero_division=0),
        "auc_macro_ovr": auc_macro,
        f"auc_{CLASSE_RISCO}": auc_risco,
    }


def main():
    args = args_da_linha_de_comando()
    args.data_dir = args.data_dir or achar_dataset()
    args.out = args.out or Path("runs") / time.strftime("%Y-%m-%d_%H%M%S")
    print("dataset:", args.data_dir, "| saída:", args.out)
    tf.keras.utils.set_random_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    ambiente = {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "gpus": [g.name for g in tf.config.list_physical_devices("GPU")],
    }
    (args.out / "ambiente.json").write_text(json.dumps(ambiente, indent=2) + "\n")

    meta = separar_por_lesao(carregar_metadados(args.data_dir, args.limit), args.seed)
    meta[["image_id", "lesion_id", "dx", "split"]].to_csv(args.out / "split.csv", index=False)
    partes = {s: meta[meta["split"] == s] for s in ("train", "val", "test")}
    print({s: len(df) for s, df in partes.items()}, "imagens;",
          meta["lesion_id"].nunique(), "lesões;", "GPUs:", tf.config.list_physical_devices("GPU"))

    ds_treino = dataset(partes["train"], args.img_size, args.batch_size, True, args.seed)
    ds_val = dataset(partes["val"], args.img_size, args.batch_size, False, args.seed)
    ds_teste = dataset(partes["test"], args.img_size, args.batch_size, False, args.seed)

    # Pesos por classe: sem eles o modelo aprende a dizer "nv" (67% do dataset)
    presentes = np.unique(partes["train"]["label"])
    pesos = compute_class_weight("balanced", classes=presentes, y=partes["train"]["label"])
    class_weight = dict(zip(presentes.tolist(), pesos))

    linhas = []
    for arch in args.archs:
        print(f"\n=== {arch} ===")
        tf.keras.backend.clear_session()
        modelo = montar_modelo(arch, args.img_size)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
            tf.keras.callbacks.CSVLogger(str(args.out / f"{arch}_history.csv")),
        ]
        inicio = time.time()
        hist = modelo.fit(ds_treino, validation_data=ds_val, epochs=args.epochs,
                          class_weight=class_weight, callbacks=callbacks, verbose=2)
        segundos = time.time() - inicio

        probs = modelo.predict(ds_teste, verbose=0)
        y_true = partes["test"]["label"].values
        y_pred = probs.argmax(axis=1)
        rotulos = list(range(len(CLASSES)))
        pd.DataFrame(classification_report(y_true, y_pred, labels=rotulos, target_names=CLASSES,
                                           output_dict=True, zero_division=0)).T.to_csv(
            args.out / f"{arch}_classification_report.csv")
        pd.DataFrame(confusion_matrix(y_true, y_pred, labels=rotulos),
                     index=[f"real_{c}" for c in CLASSES], columns=[f"prev_{c}" for c in CLASSES]).to_csv(
            args.out / f"{arch}_confusion_matrix.csv")

        linha = {"arch": arch, **avaliar(y_true, probs),
                 "epochs_run": len(hist.history["loss"]), "train_seconds": round(segundos),
                 "n_train": len(partes["train"]), "n_val": len(partes["val"]), "n_test": len(partes["test"])}
        print(linha)
        linhas.append(linha)
        # Grava a cada arquitetura para não perder tudo se a sessão do Kaggle cair
        pd.DataFrame(linhas).round(4).to_csv(args.out / "metrics.csv", index=False)

    print("\n" + pd.DataFrame(linhas).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
