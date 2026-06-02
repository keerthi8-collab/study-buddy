# agent.py
import json
import re
from datetime import date
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from prompts import build_prompt


class StudyBuddyAgent:
    def __init__(self, model: str = "llama3", max_token_limit: int = 1000):
        self.llm = ChatOllama(model=model, temperature=0.7)

        # Core memory
        self.covered_topics: set[str] = set()
        self.chat_history: list[dict] = []
        self.summary: str = ""
        self.turn_count: int = 0
        self.summarise_every: int = 4

        # Knowledge Graph: {topic: [related_topics]}
        self.knowledge_graph: dict[str, list[str]] = {}

        # Confusion tracker: {topic: confusion_count}
        self.confusion_tracker: dict[str, int] = {}

        # Learning style scores
        self.style_scores: dict[str, int] = {
            "visual": 0, "analytical": 0, "example_based": 0, "storytelling": 0
        }

        # Study streak
        self.streak_data: dict = {
            "dates": [],
            "current_streak": 0,
            "longest_streak": 0
        }
        self._update_streak()

        # Knowledge gap log: list of topics flagged as weak
        self.knowledge_gaps: list[str] = []

        # Quiz history
        self.quiz_history: list[dict] = []

        # Whiteboard notes: list of {topic, note}
        self.whiteboard: list[dict] = []

    # ── Streak ──────────────────────────────────────────────────────────────
    def _update_streak(self):
        today = str(date.today())
        dates = self.streak_data["dates"]
        if today not in dates:
            dates.append(today)
            dates.sort()
            self.streak_data["dates"] = dates
            # Calculate current streak
            streak = 1
            for i in range(len(dates) - 1, 0, -1):
                d1 = date.fromisoformat(dates[i])
                d2 = date.fromisoformat(dates[i - 1])
                if (d1 - d2).days == 1:
                    streak += 1
                else:
                    break
            self.streak_data["current_streak"] = streak
            self.streak_data["longest_streak"] = max(
                streak, self.streak_data["longest_streak"]
            )

    # ── Memory ──────────────────────────────────────────────────────────────
    def _summarise_history(self):
        if not self.chat_history:
            return
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in self.chat_history
        )
        prompt = (
            "Summarise this tutoring session concisely, "
            "listing topics covered and key points:\n\n" + history_text
        )
        result = self.llm.invoke([HumanMessage(content=prompt)])
        self.summary = result.content
        self.chat_history = []

    def _get_history_text(self) -> str:
        recent = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in self.chat_history
        )
        if self.summary and recent:
            return f"[Summary]\n{self.summary}\n\n[Recent]\n{recent}"
        return self.summary or recent or "None yet."

    # ── Knowledge Graph ──────────────────────────────────────────────────────
    def _update_knowledge_graph(self, concept: str, answer: str):
        key = concept.lower().strip()
        prompt = (
            f"List 3-5 topics closely related to '{concept}' based on this explanation. "
            f"Return ONLY a JSON array of strings, nothing else.\n\nExplanation:\n{answer[:500]}"
        )
        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            text = result.content.strip()
            text = re.sub(r"```json|```", "", text).strip()
            related = json.loads(text)
            if isinstance(related, list):
                self.knowledge_graph[key] = [r.lower() for r in related[:5]]
        except Exception:
            self.knowledge_graph[key] = []

    def get_knowledge_graph(self) -> dict:
        return self.knowledge_graph

    # ── Confusion Detector ───────────────────────────────────────────────────
    def detect_confusion(self, user_message: str, concept: str) -> bool:
        confusion_keywords = [
            "confused", "don't understand", "not clear", "what do you mean",
            "can you explain again", "i'm lost", "unclear", "huh", "what?",
            "don't get it", "explain again", "still confused", "too complex"
        ]
        msg_lower = user_message.lower()
        is_confused = any(kw in msg_lower for kw in confusion_keywords)
        if is_confused:
            key = concept.lower().strip()
            self.confusion_tracker[key] = self.confusion_tracker.get(key, 0) + 1
            if key not in self.knowledge_gaps:
                self.knowledge_gaps.append(key)
        return is_confused

    # ── Learning Style Detection ─────────────────────────────────────────────
    def _detect_learning_style(self, user_message: str):
        msg = user_message.lower()
        if any(w in msg for w in ["diagram", "chart", "visual", "draw", "show me", "picture"]):
            self.style_scores["visual"] += 1
        if any(w in msg for w in ["example", "instance", "like what", "for example", "such as"]):
            self.style_scores["example_based"] += 1
        if any(w in msg for w in ["why", "how does", "formula", "proof", "logic", "technically"]):
            self.style_scores["analytical"] += 1
        if any(w in msg for w in ["story", "history", "origin", "who invented", "real world"]):
            self.style_scores["storytelling"] += 1

    def get_dominant_style(self) -> str:
        return max(self.style_scores, key=self.style_scores.get)

    # ── Knowledge Gap Analyzer ───────────────────────────────────────────────
    def analyze_gaps(self) -> list[str]:
        gaps = list(self.knowledge_gaps)
        # Also add topics in knowledge graph that haven't been studied
        for topic, related in self.knowledge_graph.items():
            for r in related:
                if r not in self.covered_topics and r not in gaps:
                    gaps.append(r)
        return gaps[:10]

    # ── Quiz Generation ──────────────────────────────────────────────────────
    def generate_quiz(self, concept: str) -> list[dict]:
        prompt = (
            f"Generate 3 multiple choice questions about '{concept}'. "
            f"Return ONLY a JSON array with this exact format, nothing else:\n"
            f'[{{"question":"...","options":["A)...","B)...","C)...","D)..."],"answer":"A"}}]'
        )
        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            text = result.content.strip()
            text = re.sub(r"```json|```", "", text).strip()
            questions = json.loads(text)
            if isinstance(questions, list):
                self.quiz_history.append({"concept": concept, "questions": questions})
                return questions
        except Exception:
            pass
        return []

    # ── Whiteboard ───────────────────────────────────────────────────────────
    def add_whiteboard_note(self, topic: str, note: str):
        self.whiteboard.append({"topic": topic, "note": note})

    def get_whiteboard(self) -> list[dict]:
        return self.whiteboard

    # ── Core Explain ─────────────────────────────────────────────────────────
    def explain(self, concept: str, level: str = "intermediate",
                user_message: str = "") -> dict:
        level = level.lower()
        if level not in ("beginner", "intermediate", "expert"):
            level = "intermediate"

        key = concept.lower().strip()
        repeated = key in self.covered_topics
        is_confused = self.detect_confusion(user_message or concept, concept)
        self._detect_learning_style(user_message or concept)

        history_text = self._get_history_text()
        prompt = build_prompt(concept, level, history_text)

        if repeated:
            prompt += "\n\n[Note: already covered — give a new angle or deeper dive.]"
        if is_confused:
            prompt += "\n\n[Note: student is confused — use simpler language and a concrete analogy.]"

        dominant_style = self.get_dominant_style()
        style_hints = {
            "visual": "\n\n[Hint: student prefers visual explanations — use spatial descriptions and step-by-step breakdowns.]",
            "example_based": "\n\n[Hint: student learns by examples — lead with a concrete real-world example.]",
            "analytical": "\n\n[Hint: student is analytical — include the underlying logic and reasoning.]",
            "storytelling": "\n\n[Hint: student likes stories — frame the explanation with context and history.]"
        }
        prompt += style_hints.get(dominant_style, "")

        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content

        # Update all systems
        self._update_knowledge_graph(concept, answer)
        self.chat_history.append({"role": "student", "content": f"Explain '{concept}' at {level} level"})
        self.chat_history.append({"role": "tutor", "content": answer})
        self.covered_topics.add(key)
        self.turn_count += 1
        self._update_streak()

        if self.turn_count % self.summarise_every == 0:
            self._summarise_history()

        return {
            "answer": answer,
            "is_confused": is_confused,
            "is_repeated": repeated,
            "dominant_style": dominant_style,
            "covered_topics": list(self.covered_topics),
            "knowledge_gaps": self.analyze_gaps(),
            "streak": self.streak_data,
            "confusion_tracker": self.confusion_tracker,
        }