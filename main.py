from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import faiss
import json
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Carrega os dados dos três livros
livros = [
    {
        "nome_arquivo": "referencias.json",
        "nome_faiss": "indice_capitulos.faiss",
        "referencia": "Sabiston Textbook of Surgery 21st Ed",
    },
    {
        "nome_arquivo": "referencias_anatomia.json",
        "nome_faiss": "indice_anatomia.faiss",
        "referencia": "Surgical Anatomy and Technique – Skandalakis 5th Ed",
    },
    {
        "nome_arquivo": "referencias_mattox.json",
        "nome_faiss": "indice_mattox.faiss",
        "referencia": "Mattox Trauma 9th Ed",
    }
]

# Carrega dados e índices
for livro in livros:
    with open(livro["nome_arquivo"], "r", encoding="utf-8") as f:
        livro["referencias"] = json.load(f)
    livro["index"] = faiss.read_index(livro["nome_faiss"])

# Função para gerar embedding
def gerar_embedding(texto):
    response = client.embeddings.create(
        input=texto,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

@app.route("/perguntar", methods=["POST"])
def perguntar():
    dados = request.get_json()
    pergunta = dados.get("pergunta")

    if not pergunta:
        return jsonify({"erro": "Pergunta não fornecida."}), 400

    # Traduz a pergunta para inglês médico
    traducao = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "Traduza para inglês médico sem explicações."},
            {"role": "user", "content": pergunta}
        ]
    ).choices[0].message.content.strip()

    embedding = gerar_embedding(traducao).reshape(1, -1)

    contexto = ""
    capitulos_usados = []

    for livro in livros:
        _, indices = livro["index"].search(embedding, 3)
        for i in indices[0]:
            trecho = livro["referencias"][i]["texto"]
            capitulo = livro["referencias"][i]["capitulo"]
            contexto += f"\n[{livro['referencia']} – {capitulo}]\n{trecho}\n"
            capitulos_usados.append((livro["referencia"], capitulo))

    # Prompt final
    prompt = f"""
Você é um assistente médico especializado em Cirurgia Geral e altamente científico.
Responda à pergunta abaixo usando exclusivamente as informações contidas no material fornecido.
Não use conhecimento próprio e não adicione dados externos, mesmo que saiba a resposta.

Apresente a resposta da maneira mais completa possível, com todo conteúdo relevante extraído das fontes, sem omitir informações.
Organize a resposta em HTML, com títulos, listas e parágrafos para facilitar a leitura.

Caso a informação não esteja no contexto, responda exatamente:
<b>Informação não disponível. Dica: tente alterar as palavras, ou reformular a pergunta!</b>

Se houver divergências entre os livros, explique de forma objetiva e cite os livros logo abaixo de cada ponto de diferença.

---

<h3>Contexto:</h3>
<pre>{contexto}</pre>

<h3>Pergunta:</h3>
<p>{pergunta}</p>
"""

    resposta = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )

    resposta_texto = resposta.choices[0].message.content.strip()

    # Contagem dos capítulos mais usados (evita listar todos)
    from collections import Counter
    contagem = Counter(capitulos_usados)
    capitulos_relevantes = contagem.most_common(4)  # Pega no máximo 4 mais citados

    referencias_formatadas = "**Referências (em ordem de relevância):**\n" + "\n".join(
    f"{i+1}. {livro} – {capitulo}" for i, (livro, capitulo) in enumerate(capitulos_relevantes)
)


    return jsonify({
        "resposta": resposta_texto,
        "referencia": referencias_formatadas
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
