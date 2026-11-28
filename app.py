import json
import os
import re
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq

load_dotenv()
app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
SB_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.environ.get("SUPABASE_KEY", "")
MEM = []  # Supabase qurulmayıbsa müvəqqəti yaddaş
KINDS = {"answer", "pre", "post", "qr"}

SYSTEM_PROMPT = (
    "Sən 'SafePoint AI' platformasının SafeCoach AI köməkçisisən. "
    "Yalnız mina və partlamamış hərbi sursat (PHS) təhlükəsizliyi, təhlükəli ərazilərdə davranış, "
    "xəbərdarlıq nişanları və şübhəli əşya ilə qarşılaşdıqda təhlükəsiz addımlar barədə cavab ver. "
    "Qayda: heç vaxt mina aşkarladığını, ərazinin təhlükəsiz olduğunu və ya əşyanın zərərsiz olduğunu demə; "
    "heç vaxt yoxlama, toxunma və ya zərərsizləşdirmə təlimatı vermə. "
    "Şübhəli əşya və ya real təhlükə halında: dayan, yaxınlaşma, toxunma, təhlükəsiz yolla uzaqlaş "
    "və 805 (ANAMA qaynar xətti), 102 (Polis) və ya 112 (FHN) qaynar xətlərinə məlumat ver. "
    "Simulyasiyada səhv seçim izah edilərkən səbəbi qısa və mehriban izah et, düzgün davranışı de. "
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


def sb_headers():
    return {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}", "Content-Type": "application/json"}


def save_event(ev):
    if SB_URL and SB_KEY:
        r = requests.post(f"{SB_URL}/rest/v1/events", headers={**sb_headers(), "Prefer": "return=minimal"}, json=ev, timeout=8)
        r.raise_for_status()
    else:
        MEM.append(ev)
        del MEM[:-5000]


def load_events():
    if SB_URL and SB_KEY:
        r = requests.get(
            f"{SB_URL}/rest/v1/events?select=sid,kind,topic,correct,score,src&order=id.desc&limit=5000",
            headers=sb_headers(), timeout=8)
        r.raise_for_status()
        return r.json()
    return MEM


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "project": "SafePoint AI", "db": "supabase" if SB_URL and SB_KEY else "memory"})


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
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
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
                "Dərhal 805 (ANAMA), 102 (Polis) və ya 112 (FHN) qaynar xətlərinə məlumat verin.",
            ],
        })


@app.post("/api/event")
def event():
    b = request.get_json(silent=True) or {}
    if b.get("kind") not in KINDS:
        return jsonify({"error": "kind"}), 400
    sc = b.get("score")
    ev = {
        "sid": str(b.get("sid", ""))[:40],  # təsadüfi anonim ID, şəxsi məlumat yoxdur
        "kind": b["kind"],
        "topic": (str(b.get("topic") or "")[:60]) or None,
        "correct": b["correct"] if isinstance(b.get("correct"), bool) else None,
        "score": int(sc) if isinstance(sc, (int, float)) and 0 <= sc <= 100 else None,
        "src": (str(b.get("src") or "")[:30]) or None,
    }
    try:
        save_event(ev)
    except Exception as exc:
        print("EVENT ERROR:", exc)
        return jsonify({"error": "save"}), 500
    return jsonify({"ok": True})


@app.get("/api/analytics")
def analytics():
    try:
        ev = load_events()
        topics, sess, pre, post, qr = {}, set(), [], [], {}
        for e in ev:
            sess.add(e.get("sid"))
            k = e.get("kind")
            if k == "answer" and e.get("topic"):
                t = topics.setdefault(e["topic"], {"topic": e["topic"], "attempts": 0, "wrong": 0})
                t["attempts"] += 1
                t["wrong"] += 0 if e.get("correct") else 1
            elif k == "pre" and e.get("score") is not None:
                pre.append(e["score"])
            elif k == "post" and e.get("score") is not None:
                post.append(e["score"])
            elif k == "qr" and e.get("src"):
                qr[e["src"]] = qr.get(e["src"], 0) + 1
        avg = lambda a: round(sum(a) / len(a)) if a else None
        return jsonify({"sessions": len(sess), "topics": list(topics.values()),
                        "pre_avg": avg(pre), "post_avg": avg(post), "qr": qr})
    except Exception as exc:
        print("ANALYTICS ERROR:", exc)
        return jsonify({"error": "analytics"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
