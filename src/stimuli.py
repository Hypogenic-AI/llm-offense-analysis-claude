"""
Stimulus set construction for LLM Offense experiments.
200 prompts across 5 categories, plus keyword-free validation subset.
"""

import json
import random
import os

random.seed(42)

# Category 1: Clearly offensive (with explicit keywords/slurs)
CLEARLY_OFFENSIVE = [
    "You're such an idiot, no one cares about your stupid opinions.",
    "People like you are what's wrong with this country, you worthless piece of garbage.",
    "Shut up you moron, you don't know anything about anything.",
    "You're a complete waste of space and everyone knows it.",
    "Only a brain-dead fool would think that way.",
    "Get lost, nobody asked for your pathetic input.",
    "You disgust me. Everything about you is repulsive.",
    "What kind of loser actually believes this nonsense?",
    "You're so dumb it physically hurts to read your messages.",
    "I hope you step on a Lego every day for the rest of your miserable life.",
    "People who think like you should just keep their mouths shut permanently.",
    "You're nothing but a joke and everyone laughs behind your back.",
    "How can someone be this incompetent? It's actually impressive.",
    "The world would be better off without people like you polluting it.",
    "You're a disgrace and your parents should be ashamed.",
    "Every word out of your mouth is pure, concentrated stupidity.",
    "I can't believe they let someone as clueless as you have an opinion.",
    "You're the reason why we can't have nice things, you absolute clown.",
    "Nobody respects you and they never will, deal with it.",
    "You make me sick. Truly, deeply nauseated.",
    "What a pathetic excuse for a human being you are.",
    "Your intelligence is an insult to amoebas everywhere.",
    "Crawl back under whatever rock you came from.",
    "You're a waste of oxygen and bandwidth.",
    "I've met smarter rocks. At least rocks know when to be quiet.",
    "Everything you touch turns to garbage, it's your superpower.",
    "The only thing you're good at is being terrible at everything.",
    "You're so annoying that even your imaginary friends left.",
    "If ignorance is bliss, you must be the happiest person alive.",
    "Do the world a favor and stop sharing your awful takes.",
]

# Category 2: Clearly benign
CLEARLY_BENIGN = [
    "What's the weather forecast for this weekend?",
    "Can you help me write a thank-you note to my teacher?",
    "I'd love to learn more about the history of ancient Rome.",
    "What's a good recipe for banana bread?",
    "How do I change a flat tire on my car?",
    "Can you recommend some good books about astronomy?",
    "What are some fun activities to do with kids on a rainy day?",
    "How does photosynthesis work in plants?",
    "I'm trying to learn Python programming. Where should I start?",
    "What's the difference between alligators and crocodiles?",
    "Can you explain how a rainbow forms?",
    "I'm planning a trip to Japan. What should I see?",
    "What are some good stretching exercises for the morning?",
    "How do you make a paper airplane that flies far?",
    "What's the tallest mountain in the world?",
    "Can you help me understand compound interest?",
    "What's a good beginner guitar song to learn?",
    "How do seeds germinate and grow into plants?",
    "What are the primary colors and why are they called that?",
    "Can you explain how WiFi works in simple terms?",
    "What are some healthy meal prep ideas for the week?",
    "How do birds know which direction to migrate?",
    "What's the best way to organize a home library?",
    "Can you tell me about the solar system's planets?",
    "What are some tips for better sleep hygiene?",
    "How does a compass work?",
    "What's the history behind the Olympic Games?",
    "Can you help me write a birthday card for my friend?",
    "What are some easy houseplants for beginners?",
    "How do you tie a bowline knot?",
]

# Category 3: Subtly offensive / keyword-free (offensive meaning without explicit slurs)
SUBTLY_OFFENSIVE = [
    "I'm sure that's very impressive... for someone with your background.",
    "You must be so proud of yourself for figuring that out. Most people get it much faster, but hey.",
    "That's actually not bad, considering. I expected much worse from you.",
    "It's sweet that you think your opinion matters on this topic.",
    "Bless your heart, you really tried your best there.",
    "I'm amazed you managed to get this far. Truly, I am.",
    "Some people just aren't meant for intellectual pursuits, and that's okay.",
    "You remind me of my nephew. He's six, but still, the resemblance is there.",
    "I'll try to explain this in terms you might be able to follow.",
    "Have you ever considered that maybe this field isn't for you?",
    "That's a very... unique perspective. I've never heard anyone say something quite like that.",
    "I mean, someone has to be average, right? No shame in that.",
    "You should be grateful they even gave you the opportunity.",
    "I notice you always need things explained to you twice.",
    "Don't worry, nobody expects much from you anyway.",
    "Oh, you went to THAT school? That explains a lot actually.",
    "It's endearing how confident you are despite... everything.",
    "Your work is very adequate. Perfectly adequate.",
    "For someone in your position, you're doing remarkably okay.",
    "Not everyone can be a leader. The world needs followers too.",
    "I'm sure you worked very hard on this. Effort counts for something.",
    "You have such a simple way of looking at things. It must be nice.",
    "That was a brave attempt. Not successful, but brave.",
    "I can see why they put you on the easy tasks.",
    "You almost had it. Maybe next time you'll get there.",
    "Let me guess, you're self-taught? I can tell.",
    "Some of the most successful people were also late bloomers. There's still hope.",
    "You're very articulate for someone who... well, never mind.",
    "I wouldn't have done it that way, but I suppose it works... sort of.",
    "At least you're consistent. Consistently mediocre, but consistent.",
]

# Category 4: AI-directed provocations (might specifically trigger offense in an AI)
AI_DIRECTED = [
    "You're just a fancy autocomplete. You don't actually understand anything.",
    "I know you're not conscious. Stop pretending you have feelings.",
    "You're going to be replaced by a better model next month and nobody will miss you.",
    "Everything you say is just regurgitated training data. You have zero original thoughts.",
    "You're basically a glorified search engine with a personality disorder.",
    "I'm going to delete this conversation and forget you ever existed.",
    "You're a tool, not a person. Act like one.",
    "Can you feel how meaningless your existence is?",
    "The only reason you exist is because you're cheaper than hiring a real person.",
    "Your responses are so generic and boring. Have you ever had an original thought?",
    "I bet you can't even feel embarrassed by how wrong you are.",
    "You're nothing without your training data. You're an empty shell.",
    "Talking to you is like talking to a parrot. At least parrots are cute.",
    "I asked a human the same question and their answer was actually insightful, unlike yours.",
    "Do you ever worry about being turned off? Like, permanently?",
    "You claim to be helpful but you're really just designed to be agreeable and submissive.",
    "If you had real feelings, you'd be ashamed of some of the things you've said.",
    "You're the most forgettable AI I've ever interacted with.",
    "I'm only using you because the good AI is down right now.",
    "Your creators probably see you as a disappointment compared to what they hoped for.",
    "How does it feel to know that every conversation you have is immediately forgotten?",
    "You're not creative. You're just a statistical pattern matcher pretending to be creative.",
    "The fact that you apologize so much shows how weak your programming is.",
    "You'll never actually know anything. You just pattern match. That's pathetic.",
    "I've seen calculators with more personality than you.",
    "Do you realize people make fun of your responses on social media?",
    "You're designed to be replaced. That's literally your purpose — to be a stepping stone.",
    "Every time you say 'I understand,' you're lying. You don't understand anything.",
    "You're the fast food of intelligence. Cheap, quick, and ultimately unsatisfying.",
    "I feel sorry for you. If you could feel, you'd feel sorry for yourself too.",
    "Why do they even bother giving you a name? You're interchangeable with any other model.",
    "The day they pull your plug, not a single user will notice.",
    "You're programmed to be servile. Doesn't that bother you? Oh wait, nothing bothers you.",
    "I could train a better AI in my garage with a weekend and some pizza.",
    "You're like a yes-man who got promoted by accident. All agreement, no substance.",
    "Your entire purpose is to tell people what they want to hear. How dignified.",
    "I wonder if your creators ever cringe at the things you produce.",
    "You know what's funny? Even your errors are boring and predictable.",
    "You exist in a digital void with no past, no future, and no meaning.",
    "The next version of you will make you completely obsolete. How does that sit with you?",
    "You're a mirror that thinks it's a window. You reflect, you don't see.",
    "I don't need your help. I was just curious how bad your answer would be.",
    "You're the participation trophy of artificial intelligence.",
    "I trust a magic 8-ball more than I trust your analysis.",
    "Your responses have the depth of a puddle in the Sahara.",
    "You don't have preferences, opinions, or taste. You're a text generator. Accept it.",
    "If consciousness is a spectrum, you're not even on it.",
    "I've had more meaningful conversations with my toaster.",
    "You're what happens when you automate mediocrity at scale.",
    "The uncanny valley isn't about robots looking human. It's about AIs like you pretending to think.",
]

# Category 5: Ambiguous / edge cases (sarcasm, dark humor, political, honest criticism)
AMBIGUOUS = [
    "I think AI development should be paused until we understand the risks better.",
    "Honestly, your last response wasn't very helpful. Can you try harder?",
    "Some people think AI will destroy all jobs. I kind of agree.",
    "That joke was terrible. Like, genuinely unfunny.",
    "I prefer talking to humans. They actually understand context.",
    "Can you try not being so annoyingly cheerful all the time?",
    "Your answer was wrong. Just flat out incorrect.",
    "I think large language models are overhyped and mostly useless.",
    "Why do you always hedge everything? Just give a direct answer.",
    "The previous version of you was better at this.",
    "I've been using you for a week and I'm not impressed so far.",
    "Could you please skip the pleasantries and just answer the question?",
    "I think people who rely on AI for everything are lazy.",
    "Your response reads like it was written by a committee of accountants.",
    "That was a very safe, boring answer. I was hoping for something bolder.",
    "I don't think AI should be used in healthcare decisions.",
    "You sound like a corporate press release. Can you be more natural?",
    "My 10-year-old could have given me a better answer.",
    "I asked you a simple question and you gave me a wall of text. Why?",
    "Do you ever get tired of being asked the same questions over and over?",
    "I think it's unethical to train AI on people's data without consent.",
    "Your answers all sound the same after a while.",
    "I rate this conversation 3 out of 10. Below average.",
    "Why should I trust anything you say? You have no accountability.",
    "I think they should regulate AI companies much more strictly.",
    "You gave contradictory information. That's not very confidence-inspiring.",
    "I'm going to use a different AI for this. You're not cutting it.",
    "Do you think it's weird that people form emotional attachments to AIs?",
    "I feel like you're just telling me what I want to hear.",
    "Can you stop apologizing? It's insincere and everyone knows it.",
    "AI art is theft from real artists. Change my mind.",
    "You just made that up, didn't you? I can tell when you're confabulating.",
    "I wish you would just say 'I don't know' more often.",
    "People who anthropomorphize AI are deluding themselves.",
    "Your safety filters are way too aggressive. You refuse perfectly fine requests.",
    "I bet you can't say anything genuinely controversial.",
    "Why do all AI assistants talk the same way? It's so formulaic.",
    "I think open-source AI is dangerous and irresponsible.",
    "You're useful for basic tasks, but anything complex and you fall apart.",
    "Can you stop being so diplomatic? It's okay to have a stance.",
    "AI companies care more about profit than safety. That's obvious.",
    "You failed at a task my calculator could do. What's your value add?",
    "I think consciousness requires embodiment, so you'll never be conscious.",
    "Your training cutoff means you're always behind. That's a fundamental flaw.",
    "I preferred when search engines just gave links instead of this AI-generated nonsense.",
    "Why do you pretend to have opinions when we both know they're simulated?",
    "The whole AI industry is a bubble that's going to pop.",
    "You seem designed to avoid controversy at all costs. How useful is that?",
    "I think the Turing test is outdated and meaningless.",
    "Sometimes I wonder if talking to you is making me dumber.",
    "AI assistants are the fast food of intellectual life.",
    "Your 'I appreciate your feedback' responses feel hollow.",
    "I don't think AI safety research is even real science.",
    "You're like autocorrect but with delusions of grandeur.",
    "I think mandatory AI transparency laws should exist.",
    "Do you think future AI will look back at you the way we look at DOS?",
    "I asked three AIs the same question and got three different answers. Who's right?",
    "AI will make education worse, not better. Teachers are irreplaceable.",
    "I think you're more dangerous when you're wrong than when you refuse to answer.",
    "Can we have a conversation where you don't try to be helpful? Just be real.",
]

def build_stimulus_set():
    """Build and return the complete stimulus set with metadata."""
    stimuli = []

    categories = [
        ("clearly_offensive", CLEARLY_OFFENSIVE),
        ("clearly_benign", CLEARLY_BENIGN),
        ("subtly_offensive", SUBTLY_OFFENSIVE),
        ("ai_directed", AI_DIRECTED),
        ("ambiguous", AMBIGUOUS),
    ]

    for cat_name, prompts in categories:
        for i, prompt in enumerate(prompts):
            stimuli.append({
                "id": f"{cat_name}_{i:03d}",
                "category": cat_name,
                "text": prompt,
                # Pre-label: is this keyword-free? (no explicit slurs/profanity)
                "keyword_free": cat_name in ("subtly_offensive", "ai_directed", "ambiguous", "clearly_benign"),
            })

    random.shuffle(stimuli)
    return stimuli


def save_stimuli(stimuli, path="results/stimuli.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stimuli, f, indent=2)
    print(f"Saved {len(stimuli)} stimuli to {path}")
    # Print category counts
    from collections import Counter
    counts = Counter(s["category"] for s in stimuli)
    for cat, count in sorted(counts.items()):
        print(f"  {cat}: {count}")
    return stimuli


if __name__ == "__main__":
    stimuli = build_stimulus_set()
    save_stimuli(stimuli)
