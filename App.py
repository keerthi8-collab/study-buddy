# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from agent import StudyBuddyAgent
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

agent = StudyBuddyAgent(model="llama3", max_token_limit=1000)

BASE_DIR = os.path.dirname(__file__)

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

# ── Core explain ────────────────────────────────────────────────────────────
@app.route("/explain", methods=["POST"])
def explain():
    data = request.json
    concept = data.get("concept", "").strip()
    level = data.get("level", "intermediate")
    user_message = data.get("user_message", "")
    if not concept:
        return jsonify({"error": "No concept provided"}), 400
    result = agent.explain(concept, level, user_message)
    return jsonify(result)

# ── Quiz ────────────────────────────────────────────────────────────────────
@app.route("/quiz", methods=["POST"])
def quiz():
    data = request.json
    concept = data.get("concept", "").strip()
    if not concept:
        return jsonify({"error": "No concept provided"}), 400
    questions = agent.generate_quiz(concept)
    return jsonify({"questions": questions, "concept": concept})

# ── Knowledge graph ─────────────────────────────────────────────────────────
@app.route("/graph", methods=["GET"])
def graph():
    return jsonify({"graph": agent.get_knowledge_graph()})

# ── Whiteboard ──────────────────────────────────────────────────────────────
@app.route("/whiteboard", methods=["GET"])
def get_whiteboard():
    return jsonify({"notes": agent.get_whiteboard()})

@app.route("/whiteboard", methods=["POST"])
def add_whiteboard():
    data = request.json
    agent.add_whiteboard_note(data.get("topic", ""), data.get("note", ""))
    return jsonify({"notes": agent.get_whiteboard()})

# ── Gaps ────────────────────────────────────────────────────────────────────
@app.route("/gaps", methods=["GET"])
def gaps():
    return jsonify({"gaps": agent.analyze_gaps()})

# ── Stats ───────────────────────────────────────────────────────────────────
@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "streak": agent.streak_data,
        "style_scores": agent.style_scores,
        "dominant_style": agent.get_dominant_style(),
        "confusion_tracker": agent.confusion_tracker,
        "covered_topics": list(agent.covered_topics),
        "gaps": agent.analyze_gaps(),
        "quiz_count": len(agent.quiz_history),
    })

# ── Reset ───────────────────────────────────────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset():
    global agent
    agent = StudyBuddyAgent(model="llama3", max_token_limit=1000)
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)