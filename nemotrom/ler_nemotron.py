import os
import base64
import csv
import re
import time
import unicodedata
from pathlib import Path

import requests


# ================= CONFIGURAÇÕES =================

PASTA = Path(__file__).resolve().parent
IMAGENS = PASTA.parent / "recortes_margem10_v2" / "com_margem"
CSV = PASTA / f"resultados_{time.time_ns()}.csv"

MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MAX_TENTATIVAS = 5
ESPERA_ENTRE_TENTATIVAS = 10
INTERVALO_ENTRE_IMAGENS = 5

# Mesmos parâmetros da última versão separada.
TEMPERATURA = 0
SEED = 0
MAX_TOKENS = 4096
REASONING_BUDGET = 1024

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


def enviar_com_tentativas(sessao, payload):
    """Repete somente falhas técnicas temporárias."""
    temporarios = {408, 429, 500, 502, 503, 504}

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            with sessao.post(
                URL,
                json=payload,
                timeout=(30, 300),
            ) as resposta:

                codigo = resposta.status_code

                if codigo in temporarios:
                    erro = f"HTTP {codigo}"

                elif codigo != 200:
                    return {
                        "dados": None,
                        "tentativas": tentativa,
                        "erro": f"HTTP {codigo}",
                        # Evita repetir configuração inválida no dataset.
                        "parar": codigo in {400, 401, 403, 404, 422},
                    }

                else:
                    try:
                        dados = resposta.json()
                    except ValueError:
                        erro = "A API retornou JSON inválido."
                    else:
                        return {
                            "dados": dados,
                            "tentativas": tentativa,
                            "erro": "",
                            "parar": False,
                        }

        except (requests.Timeout, requests.ConnectionError) as exc:
            erro = type(exc).__name__

        except requests.RequestException as exc:
            return {
                "dados": None,
                "tentativas": tentativa,
                "erro": type(exc).__name__,
                "parar": False,
            }

        if tentativa < MAX_TENTATIVAS:
            print(
                f"    {erro}. Tentativa "
                f"{tentativa}/{MAX_TENTATIVAS} falhou."
            )
            print(
                f"    Aguardando {ESPERA_ENTRE_TENTATIVAS}s "
                "para reenviar a mesma imagem..."
            )
            time.sleep(ESPERA_ENTRE_TENTATIVAS)

    return {
        "dados": None,
        "tentativas": MAX_TENTATIVAS,
        "erro": (
            f"Falha após {MAX_TENTATIVAS} tentativas. "
            f"Último erro: {erro}"
        ),
        "parar": False,
    }


def main():
    chave = os.getenv("NVIDIA_API_KEY", "").strip()

    if not chave:
        raise SystemExit(
            "Configure NVIDIA_API_KEY no PowerShell antes de executar."
        )

    if not IMAGENS.is_dir():
        raise SystemExit(f"Pasta não encontrada:\n{IMAGENS}")

    imagens = sorted(
        imagem
        for imagem in IMAGENS.rglob("*")
        if imagem.is_file()
        and imagem.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not imagens:
        raise SystemExit(f"Nenhuma imagem encontrada:\n{IMAGENS}")

    print(f"Modelo: {MODEL}")
    print(f"Imagens encontradas: {len(imagens)}")
    print(f"CSV: {CSV}\n")

    total_salvo = 0
    interrompido = False

    with requests.Session() as sessao, CSV.open(
        "x", newline="", encoding="utf-8-sig"
    ) as arquivo:

        sessao.headers.update({
            "Authorization": f"Bearer {chave}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

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
            for indice, imagem in enumerate(imagens, start=1):
                nome = imagem.relative_to(IMAGENS).as_posix()
                print(f"[{indice}/{len(imagens)}] {nome}")

                inicio = time.perf_counter()
                bruto = ""
                texto = ""
                erro = ""
                status = "OK"
                truncado = False
                tentativas = 0
                motivo = ""
                parar = False

                try:
                    conteudo = imagem.read_bytes()
                except OSError as exc:
                    status = "INPUT_ERROR"
                    erro = f"{type(exc).__name__}: {exc}"
                else:
                    mime = (
                        "image/png"
                        if imagem.suffix.lower() == ".png"
                        else "image/jpeg"
                    )
                    imagem_b64 = base64.b64encode(
                        conteudo
                    ).decode("ascii")

                    payload = {
                        "model": MODEL,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": PROMPT,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            f"data:{mime};base64,"
                                            f"{imagem_b64}"
                                        )
                                    },
                                },
                            ],
                        }],
                        "temperature": TEMPERATURA,
                        "seed": SEED,
                        "max_tokens": MAX_TOKENS,
                        "reasoning_budget": REASONING_BUDGET,
                        "stream": False,
                    }

                    resultado = enviar_com_tentativas(sessao, payload)
                    tentativas = resultado["tentativas"]
                    parar = resultado["parar"]

                    if resultado["dados"] is None:
                        status = "API_ERROR"
                        erro = resultado["erro"]

                    else:
                        try:
                            escolha = resultado["dados"]["choices"][0]
                            resposta_texto = escolha["message"].get("content")
                            motivo = escolha.get("finish_reason", "")
                            truncado = motivo == "length"

                            if resposta_texto is None and truncado:
                                resposta_texto = ""

                            if not isinstance(resposta_texto, str):
                                raise ValueError(
                                    "Resposta sem conteúdo textual final."
                                )

                            bruto = resposta_texto

                            if motivo not in {"stop", "length"}:
                                raise ValueError(
                                    f"Finalização inesperada: {motivo!r}"
                                )

                            if bruto.strip().upper() != "NOT_FOUND":
                                texto = normalizar(bruto)

                            if truncado:
                                status = "TRUNCATED"
                            elif not texto:
                                status = "NOT_FOUND"

                        except (
                            KeyError, IndexError, TypeError, ValueError
                        ) as exc:
                            status = "API_ERROR"
                            erro = f"Resposta inválida: {exc}"

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

                # Grava cada resultado antes de passar à próxima imagem.
                arquivo.flush()
                os.fsync(arquivo.fileno())
                total_salvo += 1

                print(
                    f"    {status}: {texto or '(sem transcrição)'} "
                    f"| Tentativas: {tentativas}"
                )

                if erro:
                    print(f"    Detalhe: {erro}")

                if parar:
                    print(
                        "\nExecução interrompida. Confira a chave, "
                        "o acesso ao modelo e a configuração da API."
                    )
                    interrompido = True
                    break

                if indice < len(imagens):
                    time.sleep(INTERVALO_ENTRE_IMAGENS)

        except KeyboardInterrupt:
            interrompido = True
            print("\nInterrompido pelo usuário.")

    situacao = "Interrompido" if interrompido else "Finalizado"
    print(f"\n{situacao}: {total_salvo}/{len(imagens)} imagens registradas.")
    print(f"CSV salvo em: {CSV}")


if __name__ == "__main__":
    main()