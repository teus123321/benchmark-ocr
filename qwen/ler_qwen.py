import base64
import csv
import json
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import requests


# ================= CONFIGURAÇÕES =================

PASTA = Path(__file__).resolve().parent
IMAGENS = PASTA.parent / "recortes_margem10_v2" / "com_margem"

EXECUCAO = time.time_ns()
CSV = PASTA / f"resultados_{EXECUCAO}.csv"
LOG = PASTA / f"erros_{EXECUCAO}.jsonl"

MODEL = "qwen2.5vl:latest"
URL = "http://127.0.0.1:11434/api/chat"

MAX_TENTATIVAS = 5
ESPERA_SEGUNDOS = 10
TIMEOUT = (30, 300)

HTTP_TEMPORARIOS = {408, 429, 500, 502, 503, 504}

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


# ================= FUNÇÕES =================

def normalizar(texto):
    texto = unicodedata.normalize("NFKC", texto).upper()
    return re.sub(r"[^A-Z0-9]", "", texto)


def registrar_erro(nome, tentativa, erro, codigo, corpo):
    """Preserva a resposta do servidor sem salvar a imagem em base64."""
    registro = {
        "data": datetime.now().astimezone().isoformat(),
        "arquivo": nome,
        "modelo": MODEL,
        "tentativa": tentativa,
        "http_status": codigo,
        "erro": erro,
        "resposta_servidor": corpo,
    }

    with LOG.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
        arquivo.flush()
        os.fsync(arquivo.fileno())


def consultar(sessao, payload, nome):
    """Repete falhas técnicas; aceita a primeira resposta concluída."""
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        codigo = None
        corpo = ""
        erro = ""
        repetir = True

        try:
            with sessao.post(
                URL,
                json=payload,
                timeout=TIMEOUT,
            ) as resposta:
                codigo = resposta.status_code
                corpo = resposta.text

            if codigo != 200:
                erro = f"HTTP {codigo}: {corpo}"
                repetir = codigo in HTTP_TEMPORARIOS

            else:
                dados = json.loads(corpo)

                if not isinstance(dados, dict):
                    erro = "O servidor retornou um JSON inesperado."

                elif dados.get("error"):
                    erro = f"Ollama: {dados['error']}"

                elif dados.get("done") is not True:
                    erro = (
                        "Resposta não concluída: "
                        f"done={dados.get('done')!r}; "
                        f"done_reason={dados.get('done_reason')!r}"
                    )

                else:
                    mensagem = dados.get("message")

                    if (
                        not isinstance(mensagem, dict)
                        or not isinstance(mensagem.get("content"), str)
                    ):
                        erro = "Resposta concluída sem conteúdo textual."
                    else:
                        return dados, tentativa, ""

        except (requests.Timeout, requests.ConnectionError) as exc:
            erro = f"{type(exc).__name__}: {exc}"

        except json.JSONDecodeError as exc:
            erro = f"JSON inválido retornado pelo servidor: {exc}"

        except requests.RequestException as exc:
            erro = f"{type(exc).__name__}: {exc}"
            repetir = False

        registrar_erro(nome, tentativa, erro, codigo, corpo)

        print(
            f"    Tentativa {tentativa}/{MAX_TENTATIVAS}: "
            f"{erro[:800]}",
            flush=True,
        )

        if not repetir or tentativa == MAX_TENTATIVAS:
            return None, tentativa, erro

        print(
            f"    Aguardando {ESPERA_SEGUNDOS}s para reenviar...",
            flush=True,
        )
        time.sleep(ESPERA_SEGUNDOS)


def main():
    if not IMAGENS.is_dir():
        raise SystemExit(f"Pasta não encontrada: {IMAGENS}")

    imagens = sorted(
        p for p in IMAGENS.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not imagens:
        raise SystemExit(f"Nenhuma imagem encontrada: {IMAGENS}")

    print(f"Modelo: {MODEL}")
    print(f"Imagens: {len(imagens)}")
    print(f"CSV: {CSV}")
    print(f"Diagnóstico de falhas: {LOG}\n", flush=True)

    total_salvo = 0
    interrompido = False

    with requests.Session() as sessao, CSV.open(
        "x", newline="", encoding="utf-8-sig"
    ) as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow([
            "File_Name",
            "Extracted_Characters",
            "Raw_Response",
            "Status",
            "Error",
            "Seconds",
            "Truncated",
            "Attempts",
            "Finish_Reason",
        ])
        arquivo.flush()

        try:
            for i, imagem in enumerate(imagens, 1):
                nome = imagem.relative_to(IMAGENS).as_posix()
                print(f"[{i}/{len(imagens)}] Lendo: {nome}", flush=True)

                inicio = time.perf_counter()
                bruto = ""
                texto = ""
                erro = ""
                status = "OK"
                truncado = False
                tentativas = 0
                motivo = ""

                try:
                    conteudo = imagem.read_bytes()

                except OSError as exc:
                    status = "INPUT_ERROR"
                    erro = f"{type(exc).__name__}: {exc}"
                    registrar_erro(nome, 0, erro, None, "")

                else:
                    imagem_b64 = base64.b64encode(
                        conteudo
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

                    dados, tentativas, erro = consultar(
                        sessao, payload, nome
                    )

                    if dados is None:
                        status = "API_ERROR"

                    else:
                        bruto = dados["message"]["content"]
                        motivo = dados.get("done_reason") or ""
                        truncado = motivo == "length"

                        if bruto.strip().upper() != "NOT_FOUND":
                            texto = normalizar(bruto)

                        if truncado:
                            status = "TRUNCATED"
                        elif not texto:
                            status = "NOT_FOUND"

                writer.writerow([
                    nome,
                    texto,
                    bruto,
                    status,
                    erro,
                    round(time.perf_counter() - inicio, 3),
                    truncado,
                    tentativas,
                    motivo,
                ])

                # Grava cada imagem assim que o processamento termina.
                arquivo.flush()
                os.fsync(arquivo.fileno())
                total_salvo += 1

                print(
                    f"    {status}: {texto or '(vazio)'}\n",
                    flush=True,
                )

        except KeyboardInterrupt:
            interrompido = True
            print(
                "\nExecução interrompida. "
                "Os registros já gravados foram preservados.",
                flush=True,
            )

        finally:
            # Solicita a liberação do modelo da memória.
            try:
                with sessao.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": MODEL, "keep_alive": 0},
                    timeout=(5, 15),
                ):
                    pass
            except requests.RequestException:
                pass

    situacao = "Interrompido" if interrompido else "Finalizado"
    print(f"\n{situacao}: {total_salvo}/{len(imagens)} registros salvos.")
    print(f"CSV: {CSV}")

    if LOG.exists():
        print(f"Detalhes dos erros: {LOG}")


if __name__ == "__main__":
    main()