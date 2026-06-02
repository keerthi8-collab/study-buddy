# main.py

from agent import StudyBuddyAgent

LEVELS = ["beginner", "intermediate", "expert"]

def get_level() -> str:
    print("\nLevel: (1) beginner  (2) intermediate  (3) expert")
    choice = input("Choose [1-3, default=2]: ").strip()
    return LEVELS[{"1": 0, "2": 1, "3": 2}.get(choice, 1)]

def main():
    print("=" * 50)
    print("  StudyBuddy — Concept Explainer with Memory")
    print("  Type 'quit' to exit, 'topics' to see history")
    print("=" * 50)

    agent = StudyBuddyAgent(model="llama3", max_token_limit=1000)

    while True:
        concept = input("\n📖 What concept would you like to learn? ").strip()

        if not concept:
            continue
        if concept.lower() == "quit":
            print("Goodbye! Keep learning 🚀")
            break
        if concept.lower() == "topics":
            print("Covered so far:", ", ".join(agent.covered_topics) or "none yet")
            continue

        level = get_level()

        print(f"\n⏳ Explaining '{concept}' at {level} level...\n")
        response = agent.explain(concept, level)
        print(response)

if __name__ == "__main__":
    main()