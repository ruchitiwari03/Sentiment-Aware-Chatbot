"""
maincode.py — Sentiment-aware CLI chatbot (Tier 1 + Tier 2) — Windows-friendly

Usage (Windows):
  1) Install dependencies (PowerShell or cmd):
     pip install transformers torch sentencepiece

     If installing torch via pip fails, follow the instructions at https://pytorch.org/get-started/locally/
     and install the correct wheel for your Python version / CPU or CUDA setup.

  2) Run:
     python "d:\\Projects\\sentimental analysis\\maincode.py"

Commands inside the chatbot:
  /quit   - end conversation and show final sentiment + trend
  /save   - save conversation and analysis to JSON
  /help   - show commands

Notes:
 - Model: distilbert-base-uncased-finetuned-sst-2-english (Hugging Face)
 - First run will download model weights (several hundred MB) and cache them.
"""

import sys
import json
import os
import datetime
from typing import List, Dict, Any

# ----------------------
# Dependency check + imports
# ----------------------
try:
    from transformers import pipeline
except Exception as e:
    print("\nERROR: Missing `transformers` package or required dependencies.")
    print("Install with:\n  pip install transformers torch sentencepiece\n")
    print("If `pip install torch` fails, follow PyTorch install instructions for your platform.")
    raise e

# ----------------------
# Config / Utility funcs
# ----------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")

def signed_score_from_pipeline_result(res: Dict[str, Any]) -> float:
    """Convert pipeline result (label + score) into a signed float in [-1.0, +1.0]."""
    label = res.get("label", "").upper()
    score = float(res.get("score", 0.0))
    if label.startswith("POS"):
        return +score
    else:
        return -score

def sentiment_category_from_score(score: float) -> str:
    """Map conversation-level score to human-readable category."""
    if score >= 0.6:
        return "strongly positive"
    elif score >= 0.2:
        return "positive"
    elif score > -0.2:
        return "neutral"
    elif score > -0.6:
        return "negative"
    else:
        return "strongly negative"

# ----------------------
# Chatbot Class
# ----------------------
class ChatbotSentiment:
    def __init__(self, model_name: str = MODEL_NAME, device: int = -1):
        """
        device: -1 for CPU, else GPU device id (e.g., 0)
        """
        print(f"Loading sentiment pipeline with model '{model_name}' ... (this may take a moment on first run)")
        try:
            self.nlp = pipeline("sentiment-analysis", model=model_name, device=device)
        except Exception as e:
            print("\nERROR: Failed to load the sentiment pipeline/model.")
            print(" - Ensure you have internet to download the model on first run.")
            print(" - If you're behind a proxy or firewall, configure environment variables HTTP_PROXY / HTTPS_PROXY.")
            print(" - If you want a lightweight offline fallback, I can provide a VADER-based version.")
            raise e
        self.history: List[Dict[str, Any]] = []

    def add_message(self, speaker: str, text: str, timestamp: str = None):
        if timestamp is None:
            timestamp = now_iso()
        entry = {"speaker": speaker, "text": text, "timestamp": timestamp}
        if speaker.lower() == "user":
            try:
                sent = self.nlp(text[:512])
                sent0 = sent[0] if isinstance(sent, list) and sent else {"label": "NEUTRAL", "score": 0.0}
            except Exception:
                # Fallback: neutral if pipeline fails for a message (rare)
                sent0 = {"label": "NEUTRAL", "score": 0.0}
            entry["sentiment"] = sent0
            entry["signed_sentiment"] = signed_score_from_pipeline_result(sent0)
        else:
            entry["sentiment"] = None
            entry["signed_sentiment"] = None

        self.history.append(entry)
        return entry

    def compute_conversation_sentiment(self) -> Dict[str, Any]:
        user_entries = [h for h in self.history if h["speaker"].lower() == "user" and h.get("signed_sentiment") is not None]
        if len(user_entries) == 0:
            return {
                "num_user_messages": 0,
                "avg_score": 0.0,
                "weighted_score": 0.0,
                "category": "neutral",
                "explanation": "No user messages to evaluate."
            }

        scores = [float(e["signed_sentiment"]) for e in user_entries]
        avg_score = sum(scores) / len(scores)

        n = len(scores)
        # recency weights: 1..n (later messages get more weight)
        weights = [(i+1) for i in range(n)]
        wsum = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / wsum

        category = sentiment_category_from_score(weighted_score)

        return {
            "num_user_messages": n,
            "avg_score": avg_score,
            "weighted_score": weighted_score,
            "category": category,
            "explanation": (
                "avg_score is the plain mean of per-message signed scores in [-1,+1]. "
                "weighted_score gives more weight to recent messages and is used for the final category."
            )
        }

    def trend_summary(self) -> str:
        user_entries = [h for h in self.history if h["speaker"].lower() == "user" and h.get("signed_sentiment") is not None]
        if len(user_entries) <= 1:
            return "Not enough user messages to determine a trend."

        scores = [e["signed_sentiment"] for e in user_entries]
        deltas = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        pos_changes = sum(1 for d in deltas if d > 0.05)
        neg_changes = sum(1 for d in deltas if d < -0.05)

        first, last = scores[0], scores[-1]
        net_change = last - first

        trend_parts = []
        if net_change > 0.1:
            trend_parts.append("overall improving")
        elif net_change < -0.1:
            trend_parts.append("overall worsening")
        else:
            trend_parts.append("overall stable")

        if pos_changes > neg_changes:
            trend_parts.append(f"more positive shifts ({pos_changes} ups vs {neg_changes} downs)")
        elif neg_changes > pos_changes:
            trend_parts.append(f"more negative shifts ({neg_changes} downs vs {pos_changes} ups)")
        else:
            trend_parts.append(f"equal positive/negative shifts ({pos_changes} each)")

        return "; ".join(trend_parts) + "."

    def print_per_message_sentiments(self):
        user_entries = [h for h in self.history if h["speaker"].lower() == "user" and h.get("sentiment") is not None]
        if not user_entries:
            print("No user messages recorded.")
            return

        print("\nPer-message sentiment (most recent last):")
        print("-" * 72)
        for i, e in enumerate(user_entries, start=1):
            t = e["timestamp"]
            text = e["text"]
            s = e["sentiment"]
            signed = e["signed_sentiment"]
            label = s.get("label")
            score = s.get("score")
            print(f"{i:2d}. [{t}] {label:8s} score={score:.3f} signed={signed:+.3f}")
            preview = (text[:140] + "...") if len(text) > 140 else text
            print(f"     msg: {preview}")
        print("-" * 72)

    def save_to_json(self, filename: str = None) -> str:
        if filename is None:
            filename = f"chat_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        payload = {
            "generated_at": now_iso(),
            "model": MODEL_NAME,
            "history": self.history,
            "conversation_sentiment": self.compute_conversation_sentiment(),
            "trend_summary": self.trend_summary()
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return os.path.abspath(filename)

# ----------------------
# Main interactive loop
# ----------------------
def main():
    # Use device=-1 for CPU. If you have a GPU and torch is CUDA-enabled, set device=0
    device = -1
    bot = ChatbotSentiment(model_name=MODEL_NAME, device=device)

    print("\nChatbot ready. Type messages. Commands: /quit, /save, /help\n")
    bot.add_message("bot", "Hello! I'm your sentiment-aware chatbot. How can I help you today?")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nInput ended; treating as /quit.")
            user_input = "/quit"

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit":
                print("\nEnding conversation. Computing final sentiments...\n")
                bot.print_per_message_sentiments()
                conv = bot.compute_conversation_sentiment()
                print(f"\nConversation summary (n_user_messages={conv['num_user_messages']}):")
                print(f"  Weighted score: {conv['weighted_score']:+.3f}")
                print(f"  Average message score: {conv['avg_score']:+.3f}")
                print(f"  Final category (based on weighted score): {conv['category']}")
                print(f"  Trend: {bot.trend_summary()}\n")
                print("Thanks — conversation complete.")
                break
            elif cmd == "/save":
                fname = bot.save_to_json()
                print(f"Saved conversation + sentiments to {fname}")
                continue
            elif cmd == "/help":
                print("Commands:\n  /quit  - end and show conversation sentiment\n  /save  - save conversation+sentiment to JSON\n  /help  - show this help\n")
                continue
            else:
                print(f"Unknown command: {cmd}. Type /help for available commands.")
                continue

        added = bot.add_message("user", user_input)

        # Simple canned reply logic based on signed sentiment
        s = added["signed_sentiment"]
        if s is None:
            reply = "Thanks for sharing."
        elif s >= 0.6:
            reply = "I'm glad to hear that 😊. Tell me more if you want."
        elif s >= 0.1:
            reply = "That's nice. Would you like to expand on that?"
        elif s > -0.1:
            reply = "I see. Anything else on your mind?"
        elif s > -0.6:
            reply = "I'm sorry you're feeling that way. Do you want to talk about it?"
        else:
            reply = "That sounds really difficult. I'm here to listen if you want to share more."

        bot.add_message("bot", reply)
        print(f"Bot: {reply}")

if __name__ == "__main__":
    main()
