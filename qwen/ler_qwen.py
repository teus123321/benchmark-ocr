import base64
import csv
import re
import time
import unicodedata
from pathlib import Path

import requests


PASTA = Path(__file__).resolve().parent
IMAGENS = PASTA.parent / "recortes_margem10_v2" / "com_margem"
CSV = PASTA / f"resultados_{time.time_ns()}.csv"

MODEL = "qwen2.5vl:latest"
URL = "http://127.0.0.1:11434/api/chat"

PROMPT = """
Read only the chassis/VIN or engine identification number visible
in the main stamped or engraved region of this image.
Ignore decorative stars, asterisks, separators and punctuation.
Ignore unrelated labels, model names, addresses, GPS coordinates,
dates, timestamps, watermarks and camera overlays.
Preserve all letters and digits belonging to the identifier,
including leading zeros and suffixes.
If the identifier spans multiple lines, read left to right,
then top to bottom, and concatenate the lines.
Do not guess, complete, correct or repeat characters.
Do not assume a fixed identifier length.
Return only uppercase letters A-Z and digits 0-9.
Do not include explanations, labels, spaces or markdown.
If no identifier character is readable, return exactly NOT_FOUND.
""".strip()


def normalizar(texto):
    texto = unicodedata.normalize("NFKC", texto).upper()
    return re.sub(r"[^A-Z0-9]", "", texto)


def main():
    imagens = sorted(
        p for p in IMAGENS.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not imagens:
        raise SystemExit(f"Nenhuma imagem encontrada: {IMAGENS}")

    with requests.Session() as sessao, CSV.open(
        "x", newline="", encoding="utf-8-sig"
    ) as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow([
            "File_Name", "Extracted_Characters", "Raw_Response",
            "Status", "Error", "Seconds", "Truncated"
        ])

        try:
            for i, imagem in enumerate(imagens, 1):
                inicio = time.perf_counter()
                bruto, texto, erro = "", "", ""
                status, truncado = "OK", False

                try:
                    imagem_b64 = base64.b64encode(
                        imagem.read_bytes()
                    ).decode("ascii")

                    payload = {
                        "model": MODEL,
                        "messages": [{
                            "role": "user",
                            "content": PROMPT,
                            "images": [imagem_b64],
                        }],
                        "stream": False,
                        "keep_alive": "2m",
                        "options": {
                            "temperature": 0,
                            "seed": 0,
                            "num_predict": 128,
                            "num_ctx": 4096,
                        },
                    }

                    with sessao.post(
                        URL, json=payload, timeout=(30, 300)
                    ) as resposta:
                        resposta.raise_for_status()
                        dados = resposta.json()

                    if dados.get("done") is not True:
                        raise RuntimeError("Resposta não concluída.")

                    bruto = dados["message"]["content"]
                    if not isinstance(bruto, str):
                        raise RuntimeError("Resposta sem texto.")

                    truncado = dados.get("done_reason") == "length"

                    if bruto.strip().upper() != "NOT_FOUND":
                        texto = normalizar(bruto)

                    if not texto:
                        status = "NOT_FOUND"
                    if truncado:
                        status = "TRUNCATED"

                except Exception as exc:
                    status = "API_ERROR"
                    erro = f"{type(exc).__name__}: {exc}"

                writer.writerow([
                    imagem.relative_to(IMAGENS).as_posix(),
                    texto, bruto, status, erro,
                    round(time.perf_counter() - inicio, 3), truncado
                ])
                arquivo.flush()

                print(f"[{i}/{len(imagens)}] {status}: {texto or '(vazio)'}")
                if erro:
                    print(erro)

        finally:
            # Solicita a liberação do modelo da memória ao encerrar.
            try:
                sessao.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": MODEL, "keep_alive": 0},
                    timeout=15,
                ).close()
            except requests.RequestException:
                pass

    print(f"\nCSV salvo: {CSV}")


if __name__ == "__main__":
    main()