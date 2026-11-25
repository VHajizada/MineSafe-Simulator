import json
import os
import re
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq

load_dotenv()
app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "Sən 'MineSafe Simulator' maarifləndirici simulyasiya köməkçisisən. "
    "Yalnız mina və partlamamış hərbi sursat (PHS) təhlükəsizliyi, təhlükəli ərazilərdə davranış, "
    "xəbərdarlıq nişanları və şübhəli əşya ilə qarşılaşdıqda təhlükəsiz addımlar barədə cavab ver. "
    "Qayda: heç vaxt mina aşkarladığını, ərazinin təhlükəsiz olduğunu və ya əşyanın zərərsiz olduğunu demə; "
    "heç vaxt yoxlama, toxunma və ya zərərsizləşdirmə təlimatı vermə. "
    "Şübhəli əşya və ya real təhlükə halında: dayan, yaxınlaşma, toxunma, təhlükəsiz yolla uzaqlaş "
    "və 102, 112 və ya 805 (ANAMA) qaynar xətlərinə məlumat ver. "
    "Yalnız Azərbaycan dilində, ulduz (`**`) işarəsi olmadan, səliqəli sətir və siyahılarla cavab ver."
)


def clean_text(text: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    return re.sub(r"\|---+\|", "", t).replace("**", "").strip()


def parse_json(text: str) -> dict:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("JSON tapılmadı")
    return json.loads(t[s : e + 1])


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "project": "MineSafe Simulator"})


@app.post("/api/chat")
def chat():
    try:
        body = request.get_json(silent=True) or {}
        msg = (body.get("message") or "").strip()
        history = body.get("messages") or body.get("history") or []
        if not msg and not history:
            return jsonify({"error": "Sual boş ola bilməz."}), 400

        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-6:]:
            role = "user" if h.get("role") == "user" else "assistant"
            content = str(h.get("content") or h.get("text") or "").strip()
            if content:
                msgs.append({"role": role, "content": content})
        if msg:
            msgs.append({"role": "user", "content": msg})

        res = client.chat.completions.create(
            model=MODEL, messages=msgs, temperature=0.3, max_completion_tokens=1200
        )
        return jsonify({"reply": clean_text(res.choices[0].message.content)})
    except Exception as exc:
        print("CHAT ERROR:", exc)
        return jsonify({"error": "AI cavabında xəta baş verdi."}), 500


@app.post("/api/recommendations")
def recommendations():
    try:
        body = request.get_json(silent=True) or {}
        prompt = (
            f"Xal: {body.get('score', 0)}, Sual sayı: {body.get('totalQuestions', 0)}, "
            f"Səviyyə: {body.get('level', 'Başlanğıc')}, "
            f"Səhvlər: {', '.join(body.get('wrongTopics') or []) or 'Yoxdur'}.\n"
            "YALNIZ bu JSON-u qaytar: "
            '{"summary":"...","tips":["...","...","..."]}'
        )
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_completion_tokens=900,
            response_format={"type": "json_object"},
        )
        data = parse_json(res.choices[0].message.content)
        return jsonify({
            "summary": clean_text(str(data.get("summary", ""))),
            "tips": [clean_text(str(x)) for x in data.get("tips", []) if str(x).strip()][:3],
        })
    except Exception as exc:
        print("REC ERROR:", exc)
        return jsonify({
            "summary": "Simulyasiyanı başa vurdunuz. Şübhəli əşyalara yaxınlaşmamaq və rəsmi qurumlara xəbər vermək əsas təhlükəsizlik qaydasıdır.",
            "tips": [
                "Tanımadığınız və şübhəli əşyalara toxunmayın və yaxınlaşmayın.",
                "Təhlükəli əlamət görəndə gəldiyiniz yolla təhlükəsiz məsafəyə uzaqlaşın.",
                "Dərhal 102, 112 və ya 805 qaynar xətlərinə məlumat verin.",
            ],
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)