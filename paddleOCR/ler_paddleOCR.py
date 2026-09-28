import csv
import re
import time
import unicodedata
from pathlib import Path
from importlib.metadata import version

from paddleocr import PaddleOCR


PASTA = Path(__file__).resolve().parent
IMAGENS = PASTA.parent / "recortes_margem10_v2" / "com_margem"
CSV = PASTA / f"resultados_{time.time_ns()}.csv"


def normalizar(texto):
    texto = unicodedata.normalize("NFKC", texto).upper()
    return re.sub(r"[^A-Z0-9]", "", texto)


def main():
    if version("paddleocr").split(".")[0] != "2":
        raise SystemExit("Este script requer PaddleOCR 2.x.")

    imagens = sorted(
        p for p in IMAGENS.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not imagens:
        raise SystemExit(f"Nenhuma imagem encontrada: {IMAGENS}")

    modelo = PaddleOCR(
        lang="en",
        use_angle_cls=True,
        use_gpu=False,
        show_log=False,
    )

    with CSV.open("x", newline="", encoding="utf-8-sig") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow([
            "File_Name", "Extracted_Characters", "Raw_Response",
            "Status", "Error", "Seconds", "Truncated"
        ])

        for i, imagem in enumerate(imagens, 1):
            inicio = time.perf_counter()
            bruto, texto, erro = "", "", ""
            status = "OK"

            try:
                resultado = modelo.ocr(str(imagem), cls=True)
                linhas = resultado[0] if resultado else None
                bruto = "\n".join(
                    linha[1][0] for linha in (linhas or [])
                )
                texto = normalizar(bruto)

                if not texto:
                    status = "NOT_FOUND"

            except Exception as exc:
                status = "MODEL_ERROR"
                erro = f"{type(exc).__name__}: {exc}"

            writer.writerow([
                imagem.relative_to(IMAGENS).as_posix(),
                texto, bruto, status, erro,
                round(time.perf_counter() - inicio, 3), False
            ])
            arquivo.flush()

            print(f"[{i}/{len(imagens)}] {status}: {texto or '(vazio)'}")
            if erro:
                print(erro)

    print(f"\nCSV salvo: {CSV}")


if __name__ == "__main__":
    main()