import json
from pathlib import Path

class PersonalityEngine:
    def __init__(self, big5_path: str, personality_path: str):
        with open(big5_path, "r") as f:
            self.big5 = json.load(f)

        self.load_personality(personality_path)

    def load_personality(self, personality_path: str):
        self.personality_path = Path(personality_path)
        self.personality_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.personality_path.exists():
            print("Creating default personality...")
            default_personality = {
                "self": {
                    "scale": { "min": 1, "max": 5 },
                    "traits": {
                        "o": 3,
                        "c": 3,
                        "e": 3,
                        "a": 3,
                        "n": 3
                    }
                }
            }

            with open(self.personality_path, "w") as f:
                json.dump(default_personality, f, indent=2)

            self.personality = default_personality

        else:
            # load existing
            with open(self.personality_path, "r") as f:
                self.personality = json.load(f)

    def trait_text(self, trait_key:str, level: int) -> str:
        pos = self.big5["big5"][trait_key]["pos"]
        neg = self.big5["big5"][trait_key]["neg"]

        if level == 3:
            return ""
        elif level == 1:
            return f"EXTREMELY {neg}"
        elif level == 2:
            return f"low {neg}"
        elif level == 4:
            return f"low {pos}"
        elif level == 5:
            return f"EXTREMELY {pos}"
        else:
            raise ValueError("Trait level must be 1–5")

    def get_traits_raw(self) -> dict:
        """Return raw trait values (o, c, e, a, n) as ints 1-5 for turn manager / anti-social check."""
        traits = self.personality.get("self", {}).get("traits", {})
        return {
            "o": traits.get("o", 3),
            "c": traits.get("c", 3),
            "e": traits.get("e", 3),
            "a": traits.get("a", 3),
            "n": traits.get("n", 3),
        }

    def is_antisocial(
        self,
        extraversion_max: int = 2,
        agreeableness_max: int = 2,
    ) -> bool:
        """True if robot is considered anti-social (low e and low a). Disqualified from speaking in group."""
        raw = self.get_traits_raw()
        return raw["e"] <= extraversion_max and raw["a"] <= agreeableness_max

    def get_big5_vals(self) -> dict:
        traits = self.personality["self"]["traits"]

        return {
            "openness": self.trait_text("openness", traits["o"]),
            "conscientiousness": self.trait_text("conscientiousness", traits["c"]),
            "extraversion": self.trait_text("extraversion", traits["e"]),
            "agreeableness": self.trait_text("agreeableness", traits["a"]),
            "neuroticism": self.trait_text("neuroticism", traits["n"]),
        }

    def to_prompt_text(self) -> str:
        """Final text for LLM prompts"""
        return "\n".join(
            f"{k}: {v}" for k, v in self.get_big5_vals().items() if v
        )

