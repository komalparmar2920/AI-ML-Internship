import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Lenovo\Downloads\student-scores.csv")
dataset.columns()

# Predefined lessons
lessons = [
    {"title": "Math Basics", "tags": ["math", "beginner"], "difficulty": "easy"},
    {
        "title": "Physics Fundamentals",
        "tags": ["physics", "intermediate"],
        "difficulty": "medium",
    },
    {
        "title": "Advanced Chemistry",
        "tags": ["chemistry", "advanced"],
        "difficulty": "hard",
    },
    {"title": "English Grammar", "tags": ["english", "beginner"], "difficulty": "easy"},
    {
        "title": "Biology for Doctors",
        "tags": ["biology", "career"],
        "difficulty": "medium",
    },
    {
        "title": "Geography Insights",
        "tags": ["geography", "intermediate"],
        "difficulty": "medium",
    },
]

# Initialize user profiles
user_profiles = {}

# Q-learning parameters for adaptive difficulty
states = ["easy", "medium", "hard"]
actions = ["lesson1", "lesson2", "lesson3"]
q_table = {(state, action): 0 for state in states for action in actions}
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Initialize Hugging Face QA pipeline
qa_pipeline = pipeline("question-answering")


# Reward function for Q-learning
def reward_function(state, action):
    return random.uniform(0, 1)  # Simulated reward


# Q-learning update function
def update_q_table(state, action):
    reward = reward_function(state, action)
    old_q_value = q_table[(state, action)]
    q_table[(state, action)] = old_q_value + alpha * (
        reward + gamma * max([q_table[(state, a)] for a in actions]) - old_q_value
    )


# Generate user profiles from dataset
def generate_user_profiles():
    for _, row in dataset.iterrows():
        user_profiles[row["id"]] = {
            "name": f"{row['first_name']} {row['last_name']}",
            "preferences": {
                "style": "visual"
                if row["extracurricular_activities"]
                else "text-heavy",
            },
            "scores": {
                "math": row["math_score"],
                "history": row["history_score"],
                "physics": row["physics_score"],
                "chemistry": row["chemistry_score"],
                "biology": row["biology_score"],
                "english": row["english_score"],
                "geography": row["geography_score"],
            },
            "career_aspiration": row["career_aspiration"],
            "difficulty_level": "easy"
            if row["weekly_self_study_hours"] < 5
            else "medium",
        }


def recommend_lessons(user_id):
    user = user_profiles[user_id]
    career = user["career_aspiration"].lower()
    recommendations = []

    for lesson in lessons:
        if user["difficulty_level"] == lesson["difficulty"] or career in lesson["tags"]:
            recommendations.append(lesson)
    return recommendations


def ask_question(context, question):
    answer = qa_pipeline(question=question, context=context)
    return answer["answer"]


def get_lesson_context(lesson_title):
    lesson_contexts = {
        "Math Basics": "Math Basics focuses on fundamental arithmetic operations, such as addition, subtraction, multiplication, and division. It also covers basic geometry, such as calculating areas and perimeters, and an introduction to algebra, including solving equations like 2x + 3 = 7. Decimals, fractions, and percentages are also discussed to help users with real-world math applications.Mathematics is the study of numbers, shapes, and patterns. Basic operations include addition, subtraction, multiplication, and division.",
        "Physics Fundamentals": "Physics Fundamentals is the foundation of physics that covers essential concepts such as motion, forces, energy, waves, electricity, and magnetism. Understanding these basics helps in solving real-world problems, engineering applications, and further studies in advanced physics topics.Newton's second law states that Force equals mass times acceleration. It is often expressed as F = ma. Mastering Physics Fundamentals requires curiosity, problem-solving, and consistent practice. Start with the basics, build on your knowledge, and explore real-world applications to deepen your understanding.",
        "Advanced Chemistry": "Advanced Chemistry delves into complex topics in chemistry, building on foundational knowledge. It covers areas such as chemical kinetics, where learners study the speed of reactions and factors affecting them, and thermodynamics, focusing on energy changes during chemical processes. The course also introduces advanced concepts in organic chemistry, including reaction mechanisms like nucleophilic substitution and electrophilic addition. Analytical techniques, such as spectroscopy and chromatography, are explored to help users understand how compounds are identified and quantified. Additionally, Advanced Chemistry discusses coordination chemistry, studying metal complexes, and their applications in catalysis and medicine. Topics like acid-base equilibria, electrochemistry, and molecular orbital theory are also included for a deeper understanding of chemical behavior and reactions.Chemical reactions involve the transformation of substances. Balancing equations is essential in chemistry.",
        "English Grammar": "English Grammar focuses on the foundational and advanced rules that govern the English language. It begins with basic topics such as parts of speech (nouns, verbs, adjectives, adverbs), sentence structure (subject-verb-object), and punctuation (periods, commas, and question marks).The course also covers verb tenses (past, present, future) and their proper usage, along with subject-verb agreement to ensure clarity in writing and speech. Advanced topics include active and passive voice, direct and indirect speech, and complex sentence structures like conditional sentences and relative clauses.Learners explore the correct use of articles (a, an, the), prepositions (in, on, at), and conjunctions (and, but, because) for connecting ideas effectively. Additionally, topics such as vocabulary building, common idioms, and phrasal verbs are introduced to enhance fluency and comprehension in both written and spoken English.English grammar includes parts of speech, sentence structure, and punctuation rules.",
        "Biology for Doctors": "Biology for Doctors focuses on the fundamental and applied aspects of biology that are critical for medical professionals. It begins with human anatomy and physiology, providing an in-depth understanding of organ systems such as the cardiovascular, respiratory, nervous, and digestive systems.The course also covers cellular biology, exploring cell structure, function, and processes like mitosis, meiosis, and cellular respiration. Key topics in genetics, such as DNA replication, gene expression, and inheritance patterns, are included to help learners understand the genetic basis of diseases.Advanced sections dive into microbiology, studying pathogens, immunology, and the human immune response, along with pharmacology basics, including how drugs interact with biological systems. Topics like disease pathology, diagnostic techniques, and an introduction to medical research methods are also included to prepare learners for clinical applications.The course emphasizes practical knowledge, linking biology concepts to medical case studies and scenarios to bridge the gap between theory and practice in a healthcare setting.Biology studies living organisms. Key topics include cell structure, DNA, and the human body systems.",
        "Geography Insights": "Geography Insights explores both physical and human geography, offering a comprehensive understanding of the Earth and its systems. The course begins with physical geography topics such as landforms, climate patterns, and ecosystems, focusing on how natural forces like plate tectonics, erosion, and weather influence the environment.Learners also study human geography, examining population distribution, urbanization, and cultural landscapes. Topics like economic geography are introduced, covering resource distribution, trade networks, and the impact of globalization on societies.Advanced sections delve into cartography, teaching map reading, interpretation, and Geographic Information Systems (GIS) for analyzing spatial data. The course also discusses environmental challenges, such as deforestation, desertification, and climate change, highlighting the need for sustainable development.Geography Insights emphasizes real-world applications, linking geographic knowledge to current global issues like migration, geopolitical conflicts, and natural disaster management. It prepares learners to understand and address the interconnectedness of people, places, and the environment. Geography involves the study of Earth's landscapes, environments, and the relationships between people and their environments.",
    }
    return lesson_contexts.get(lesson_title, "Context not available.")


def learning_session(user_id):
    user = user_profiles[user_id]

    # Recommend lessons
    print(f"\nHello {user['name']}! Based on your profile, we recommend these lessons:")
    recommendations = recommend_lessons(user_id)
    for i, lesson in enumerate(recommendations):
        print(f"{i + 1}. {lesson['title']}")

    # Let user pick a lesson
    choice = int(input("\nChoose a lesson by entering the number: ")) - 1
    if choice < 0 or choice >= len(recommendations):
        print("Invalid choice. Returning to main menu.")
        return

    selected_lesson = recommendations[choice]
    context = get_lesson_context(selected_lesson["title"])

    # Simulate question-answering
    print(f"\nYou selected: {selected_lesson['title']}")
    question = input("Ask a question about the lesson: ")
    answer = ask_question(context, question)
    print(f"AI Tutor Answer: {answer}")

    # Update difficulty based on simulated score
    score = int(input("Enter your score for the lesson (0-100): "))
    avg_score = sum(user["scores"].values()) / len(user["scores"])
    if avg_score < 50:
        user["difficulty_level"] = "easy"
    elif avg_score < 80:
        user["difficulty_level"] = "medium"
    else:
        user["difficulty_level"] = "hard"

    print(f"\nYour new difficulty level is: {user['difficulty_level']}\n")


def main():
    print("Welcome to the Personalized Education System!")

    # Generate profiles from dataset
    generate_user_profiles()

    # Simulate a learning session
    user_id = int(input("Enter your User ID: "))
    if user_id not in user_profiles:
        print("Invalid User ID. Please try again.")
        return

    while True:
        learning_session(user_id)
        cont = input("Do you want to continue learning? (yes/no): ").lower()
        if cont != "yes":
            print("Thank you for using the system. Happy learning!")
            break


if __name__ == "__main__":
    main()
