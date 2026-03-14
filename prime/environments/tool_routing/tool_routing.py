import math
import io
import contextlib
import random
import re

import verifiers as vf
from datasets import Dataset


# ─── Knowledge Base (for simulated web_search / wikipedia_lookup) ────────────

KNOWLEDGE_BASE = {
    "mongolia": "Mongolia is a landlocked country in East Asia. Capital: Ulaanbaatar. Population: 3.3 million.",
    "mars": "Mars is the fourth planet from the Sun, called the Red Planet. Diameter: 6,779 km. Moons: Phobos and Deimos (2 moons).",
    "mount everest": "Mount Everest is the highest mountain on Earth at 8,849 meters (29,032 feet). Located on the Nepal-Tibet border.",
    "water": "Water (H2O) boils at 100°C (212°F) and freezes at 0°C (32°F). Composed of 2 hydrogen atoms and 1 oxygen atom.",
    "python programming": "Python is a programming language created by Guido van Rossum, first released in 1991.",
    "speed of light": "The speed of light in a vacuum is 299,792,458 meters per second (about 300,000 km/s).",
    "japan": "Japan is an island country in East Asia. Capital: Tokyo. Population: 125 million. GDP: approximately $4.2 trillion USD.",
    "earth": "Earth is the third planet from the Sun. Diameter: 12,742 km. Circumference: 40,075 km. One moon.",
    "gold": "Gold (Au) has atomic number 79, density 19.3 g/cm³, melting point 1,064°C. Mohs hardness: 2.5.",
    "australia": "Australia is a country and continent. Capital: Canberra. Largest city: Sydney. Population: 26 million. Area: 7,692,024 sq km.",
    "dna": "DNA (deoxyribonucleic acid) has a double helix structure discovered by Watson and Crick in 1953. Four bases: A, T, G, C.",
    "brazil": "Brazil is the largest country in South America. Capital: Brasilia. Largest city: Sao Paulo. Population: 214 million. Language: Portuguese.",
    "einstein": "Albert Einstein (born 1879, Ulm, Germany) developed the theory of relativity. E=mc². Nobel Prize in Physics 1921.",
    "saturn": "Saturn is the sixth planet from the Sun, known for its rings. Diameter: 116,460 km. Largest moon: Titan. At least 146 moons.",
    "france": "France is in Western Europe. Capital: Paris. Population: 67 million. Largest EU country by area. Currency: Euro.",
    "oxygen": "Oxygen (O) has atomic number 8. Makes up about 21% of Earth's atmosphere. Essential for aerobic respiration.",
    "egypt": "Egypt spans northeast Africa. Capital: Cairo. Population: 104 million. The Nile River (6,650 km) flows through Egypt.",
    "shakespeare": "William Shakespeare (born 1564, Stratford-upon-Avon) wrote approximately 39 plays and 154 sonnets.",
    "nitrogen": "Nitrogen (N) has atomic number 7. Makes up about 78% of Earth's atmosphere. Boiling point: -196°C.",
    "canada": "Canada is in North America. Capital: Ottawa. Largest city: Toronto. Population: 40 million. Area: 9,984,670 sq km.",
    "moon": "The Moon is Earth's natural satellite. Diameter: 3,474 km. Average distance from Earth: 384,400 km. Orbital period: 27.3 days.",
    "india": "India is in South Asia. Capital: New Delhi. Population: 1.4 billion (most populous country). Languages: Hindi and English.",
    "central park": "Central Park is in New York City. Area: 843 acres (3.41 sq km). Length: 4 km, width: 0.8 km.",
    "football field": "A standard American football field is 100 yards long, 53.3 yards wide. With end zones: 120 yards. Total area: about 1.32 acres.",
    "cheetah": "The cheetah is the fastest land animal, reaching up to 120 km/h (75 mph). Found mainly in Africa.",
    "pacific ocean": "The Pacific Ocean is the largest ocean. Area: 165.25 million sq km. Deepest point: Mariana Trench at 10,994 meters.",
    "piano": "A standard piano has 88 keys: 52 white and 36 black. Invented around 1700 by Bartolomeo Cristofori.",
    "human body": "The adult human body has 206 bones. Heart beats about 100,000 times/day. Normal temperature: 37°C (98.6°F). Blood volume: 5 liters.",
    "amazon company": "Amazon.com was founded by Jeff Bezos in 1994 in Seattle, Washington. Started as an online bookstore.",
    "venus": "Venus is the second planet from the Sun. Surface temperature: 462°C (hottest planet). Diameter: 12,104 km.",
    "germany": "Germany is in Central Europe. Capital: Berlin. Population: 84 million. Largest economy in Europe. Currency: Euro.",
    "tesla": "Nikola Tesla (born 1856, Smiljan, Croatia) was a Serbian-American inventor. Known for AC electricity. Over 300 patents.",
    "diamond": "Diamond is crystalline carbon. Hardest natural material (Mohs hardness: 10). Density: 3.51 g/cm³.",
    "kenya": "Kenya is in East Africa. Capital: Nairobi. Population: 54 million. Mount Kenya: 5,199 meters (second-highest peak in Africa).",
    "mercury element": "Mercury (Hg) has atomic number 80. Only metal liquid at room temperature. Melts at -39°C, boils at 357°C.",
    "south korea": "South Korea is in East Asia. Capital: Seoul. Population: 52 million. 10th largest economy by GDP.",
    "jupiter": "Jupiter is the fifth and largest planet. Diameter: 139,820 km. At least 95 known moons. Largest moon: Ganymede.",
    "nile": "The Nile is a river in Africa, 6,650 km long, flowing through Egypt.",
}


# ─── Unit Conversions ────────────────────────────────────────────────────────

UNIT_CONVERSIONS = {
    ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
    ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
    ("miles", "kilometers"): lambda v: v * 1.60934,
    ("kilometers", "miles"): lambda v: v / 1.60934,
    ("pounds", "kilograms"): lambda v: v * 0.453592,
    ("kilograms", "pounds"): lambda v: v / 0.453592,
    ("feet", "meters"): lambda v: v * 0.3048,
    ("meters", "feet"): lambda v: v / 0.3048,
    ("inches", "centimeters"): lambda v: v * 2.54,
    ("centimeters", "inches"): lambda v: v / 2.54,
    ("gallons", "liters"): lambda v: v * 3.78541,
    ("liters", "gallons"): lambda v: v / 3.78541,
    ("ounces", "grams"): lambda v: v * 28.3495,
    ("grams", "ounces"): lambda v: v / 28.3495,
    ("yards", "meters"): lambda v: v * 0.9144,
    ("meters", "yards"): lambda v: v / 0.9144,
    ("mph", "kmh"): lambda v: v * 1.60934,
    ("kmh", "mph"): lambda v: v / 1.60934,
    ("acres", "hectares"): lambda v: v * 0.404686,
    ("hectares", "acres"): lambda v: v / 0.404686,
}


# ─── Tool Implementations ────────────────────────────────────────────────────

async def web_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query string.

    Returns:
        Search results as text snippets.
    """
    query_lower = query.lower()
    results = []
    for key, value in KNOWLEDGE_BASE.items():
        query_words = set(query_lower.split())
        key_words = set(key.split())
        if query_words & key_words or key in query_lower:
            results.append(value)
    if results:
        return "\n".join(f"[Result {i+1}] {r}" for i, r in enumerate(results[:3]))
    return f"No results found for: {query}"


async def wikipedia_lookup(topic: str) -> str:
    """Look up a topic on Wikipedia for factual information.

    Args:
        topic: The topic to look up (e.g., 'Mongolia', 'Mars').

    Returns:
        Wikipedia article summary.
    """
    topic_lower = topic.lower().strip()
    if topic_lower in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[topic_lower]
    for key, value in KNOWLEDGE_BASE.items():
        if topic_lower in key or key in topic_lower:
            return value
    topic_words = set(topic_lower.split())
    best, best_score = None, 0
    for key, value in KNOWLEDGE_BASE.items():
        score = len(topic_words & set(key.split()))
        if score > best_score:
            best, best_score = value, score
    return best if best else f"No Wikipedia article found for '{topic}'."


async def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression (e.g., '847 * 293', '2 + 3 * 4').

    Returns:
        The numerical result.
    """
    try:
        expr = expression.replace("\u00d7", "*").replace("\u00f7", "/").replace("^", "**")
        safe_chars = set("0123456789+-*/.() eE")
        cleaned = expr.replace(" ", "")
        if not all(c in safe_chars for c in cleaned):
            return f"Error: expression contains invalid characters"
        result = eval(expr, {"__builtins__": {}}, {})
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            return str(int(result))
        if isinstance(result, float):
            return f"{result:.4f}".rstrip("0").rstrip(".")
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def unit_converter(value: str, from_unit: str, to_unit: str) -> str:
    """Convert a value from one unit to another.

    Args:
        value: The numeric value to convert (e.g., '72').
        from_unit: Source unit (e.g., 'fahrenheit', 'miles', 'pounds').
        to_unit: Target unit (e.g., 'celsius', 'kilometers', 'kilograms').

    Returns:
        The converted value with units.
    """
    try:
        num = float(value)
    except ValueError:
        return f"Error: '{value}' is not a valid number."
    from_key = from_unit.lower().strip().replace("°", "").replace("km/h", "kmh")
    to_key = to_unit.lower().strip().replace("°", "").replace("km/h", "kmh")
    key = (from_key, to_key)
    if key in UNIT_CONVERSIONS:
        result = UNIT_CONVERSIONS[key](num)
        return f"{num} {from_unit} = {result:.2f} {to_unit}"
    return f"Error: unsupported conversion from '{from_unit}' to '{to_unit}'."


async def wolfram_alpha(query: str) -> str:
    """Query Wolfram Alpha for mathematical and scientific computations.

    Args:
        query: A math expression or scientific query.

    Returns:
        The computed result.
    """
    try:
        clean = query
        for prefix in ["what is", "calculate", "compute", "evaluate", "solve", "find"]:
            clean = re.sub(r"(?i)^" + prefix + r"\s*", "", clean).strip()
        clean = clean.rstrip("?").strip()
        clean = clean.replace("\u00d7", "*").replace("\u00f7", "/").replace("^", "**")
        safe_chars = set("0123456789+-*/.() eE")
        if clean and all(c in safe_chars for c in clean.replace(" ", "")):
            result = eval(clean, {"__builtins__": {}}, {})
            if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
                return f"Result: {int(result)}"
            return f"Result: {result}"
    except Exception:
        pass
    query_lower = query.lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            return f"Wolfram|Alpha: {value}"
    return f"Wolfram|Alpha could not compute: {query}"


async def python_eval(code: str) -> str:
    """Execute Python code and return the output.

    Args:
        code: Python code to execute. Use print() to produce output.

    Returns:
        The printed output or return value.
    """
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            try:
                result = eval(code, {"__builtins__": __builtins__, "math": math})
                if result is not None:
                    print(result)
            except SyntaxError:
                exec(code, {"__builtins__": __builtins__, "math": math})
        output = stdout.getvalue().strip()
        return output if output else "(no output)"
    except Exception as e:
        return f"Error: {e}"


# ─── Question Pools ──────────────────────────────────────────────────────────

FACTUAL_QS = [
    ("What is the capital of Mongolia?", "Ulaanbaatar"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the height of Mount Everest in meters?", "8849"),
    ("What is the chemical formula for water?", "H2O"),
    ("Who created the Python programming language?", "Guido van Rossum"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Australia?", "Canberra"),
    ("Who discovered the structure of DNA?", "Watson and Crick"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What year was Albert Einstein born?", "1879"),
    ("How many moons does Mars have?", "2"),
    ("What is the capital of France?", "Paris"),
    ("What is the atomic number of oxygen?", "8"),
    ("What is the capital of Egypt?", "Cairo"),
    ("How many plays did Shakespeare write?", "39"),
    ("What is the boiling point of nitrogen in degrees Celsius?", "-196"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the diameter of the Moon in kilometers?", "3474"),
    ("What is the capital of India?", "New Delhi"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("How many keys does a standard piano have?", "88"),
    ("How many bones are in the adult human body?", "206"),
    ("What year was Amazon.com founded?", "1994"),
    ("What is the surface temperature of Venus in degrees Celsius?", "462"),
    ("What is the capital of Germany?", "Berlin"),
    ("What year was Nikola Tesla born?", "1856"),
    ("What is the Mohs hardness of diamond?", "10"),
    ("What is the capital of Kenya?", "Nairobi"),
    ("What is the atomic number of mercury?", "80"),
    ("What is the capital of South Korea?", "Seoul"),
    ("What is the diameter of Jupiter in kilometers?", "139820"),
    ("What is the largest moon of Saturn?", "Titan"),
    ("What is the circumference of Earth in kilometers?", "40075"),
    ("Who founded Amazon.com?", "Jeff Bezos"),
    ("How many known moons does Jupiter have?", "95"),
    ("What percentage of Earth's atmosphere is nitrogen?", "78"),
    ("What is the length of the Nile River in kilometers?", "6650"),
    ("What is the speed of light in meters per second?", "299792458"),
]

CONVERSION_QS = [
    ("Convert 72 degrees Fahrenheit to Celsius.", "22.22"),
    ("Convert 100 degrees Celsius to Fahrenheit.", "212"),
    ("Convert 5 miles to kilometers.", "8.05"),
    ("Convert 10 kilometers to miles.", "6.21"),
    ("Convert 150 pounds to kilograms.", "68.04"),
    ("Convert 75 kilograms to pounds.", "165.35"),
    ("Convert 6 feet to meters.", "1.83"),
    ("Convert 100 meters to feet.", "328.08"),
    ("Convert 12 inches to centimeters.", "30.48"),
    ("Convert 50 centimeters to inches.", "19.69"),
    ("Convert 3 gallons to liters.", "11.36"),
    ("Convert 20 liters to gallons.", "5.28"),
    ("Convert 8 ounces to grams.", "226.80"),
    ("Convert 500 grams to ounces.", "17.64"),
    ("Convert 200 yards to meters.", "182.88"),
    ("Convert 1000 meters to yards.", "1093.61"),
    ("Convert 60 mph to km/h.", "96.56"),
    ("Convert 100 km/h to mph.", "62.14"),
    ("Convert 40 degrees Fahrenheit to Celsius.", "4.44"),
    ("Convert 25 degrees Celsius to Fahrenheit.", "77"),
    ("Convert 32 degrees Fahrenheit to Celsius.", "0"),
    ("Convert 50 acres to hectares.", "20.23"),
    ("Convert 10 hectares to acres.", "24.71"),
    ("Convert 98.6 degrees Fahrenheit to Celsius.", "37"),
    ("Convert 45 kilograms to pounds.", "99.21"),
    ("Convert 15 gallons to liters.", "56.78"),
    ("Convert 36 inches to centimeters.", "91.44"),
    ("Convert 400 grams to ounces.", "14.11"),
    ("Convert 180 pounds to kilograms.", "81.65"),
    ("Convert 0 degrees Celsius to Fahrenheit.", "32"),
]

COMPUTATION_QS = [
    ("What is the sum of the first 100 natural numbers?", "5050"),
    ("What is 15 factorial?", "1307674368000"),
    ("What is 2 to the power of 20?", "1048576"),
    ("What is the sum of all even numbers from 1 to 100?", "2550"),
    ("What is the 20th Fibonacci number?", "6765"),
    ("What is the sum of squares of numbers from 1 to 10?", "385"),
    ("What is 12 factorial divided by 10 factorial?", "132"),
    ("How many prime numbers are there between 1 and 100?", "25"),
    ("What is the greatest common divisor of 252 and 105?", "21"),
    ("What is the least common multiple of 12 and 18?", "36"),
    ("What is 3 to the power of 10?", "59049"),
    ("What is the sum of the first 50 odd numbers?", "2500"),
    ("What is the product of all single-digit prime numbers (2, 3, 5, 7)?", "210"),
    ("What is the 10th triangular number?", "55"),
    ("What is the sum of cubes of numbers from 1 to 5?", "225"),
    ("What is 100 choose 2 (combinations)?", "4950"),
    ("What is 2 to the power of 32?", "4294967296"),
    ("What is the largest prime number less than 200?", "199"),
    ("What is the sum of all multiples of 3 between 1 and 100?", "1683"),
    ("What is 7 to the power of 7?", "823543"),
    ("What is the sum of the first 10 prime numbers?", "129"),
    ("What is 20 factorial divided by 18 factorial?", "380"),
    ("What is 8 factorial?", "40320"),
    ("How many perfect squares are there between 1 and 1000?", "31"),
    ("What is the remainder when 2 to the power of 10 is divided by 7?", "2"),
]

FACT_CALC_QS = [
    ("Mount Everest is 8849 meters tall. How many feet is that? Give the answer as a whole number.", "29032"),
    ("The Moon is 384400 km from Earth. How many miles is that? Round to the nearest whole number.", "238855"),
    ("Earth's circumference is 40075 km. How many miles is that? Round to the nearest whole number.", "24901"),
    ("Central Park is 843 acres. How many hectares is that? Round to the nearest whole number.", "341"),
    ("A cheetah runs at 120 km/h. How many mph is that? Round to the nearest whole number.", "75"),
    ("India has 1.4 billion people in 3287263 sq km. What is the population density per sq km? Round to the nearest whole number.", "426"),
    ("The Amazon River is 6400 km long. How many miles is that? Round to the nearest whole number.", "3977"),
    ("The Nile River is 6650 km long. How many miles is that? Round to the nearest whole number.", "4132"),
    ("The Mariana Trench is 10994 meters deep. How many feet is that? Round to the nearest whole number.", "36069"),
    ("Mount Kenya is 5199 meters tall. How many feet is that? Round to the nearest whole number.", "17057"),
    ("Mercury boils at 357 degrees Celsius. What is that in Fahrenheit? Round to the nearest whole number.", "675"),
    ("Gold melts at 1064 degrees Celsius. What is that in Fahrenheit? Round to the nearest whole number.", "1947"),
    ("A piano has 88 keys with 36 black keys. What percentage of keys are black? Round to the nearest whole number.", "41"),
    ("Earth's diameter is 12742 km. How many miles is that? Round to the nearest whole number.", "7918"),
    ("Diamond has a density of 3.51 g/cm3. How much does 100 cm3 of diamond weigh in grams?", "351"),
    ("A cheetah runs at 120 km/h. What is that in meters per second? Round to the nearest whole number.", "33"),
    ("Nitrogen boils at -196 degrees Celsius. What is that in Fahrenheit? Round to the nearest whole number.", "-321"),
    ("Venus has a diameter of 12104 km. What is its circumference in km? Round to the nearest whole number.", "38025"),
    ("Saturn's diameter is 116460 km. What is its circumference in km? Round to the nearest whole number.", "365882"),
    ("Australia has 26 million people in 7692024 sq km. What is the population density per sq km? Round to 1 decimal.", "3.4"),
]

MULTISTEP_QS = [
    ("How many American football fields (1.32 acres each) would fit in Central Park (843 acres)? Round to nearest whole number.", "638"),
    ("If you drove at 60 mph for the entire length of the Amazon River (6400 km), how many hours would it take? Round to nearest whole number.", "66"),
    ("The Moon is 384400 km away. If you drove at 100 km/h nonstop, how many days would it take? Round to nearest whole number.", "160"),
    ("Earth's circumference is 40075 km. How long would it take to walk at 5 km/h in days? Round to nearest whole number.", "334"),
    ("Jupiter's diameter is 139820 km. How many Earths (diameter 12742 km) could fit across? Round to nearest whole number.", "11"),
    ("The Nile is 6650 km. If you kayaked at 8 km/h for 10 hours a day, how many days to finish? Round to nearest whole number.", "83"),
    ("If you could travel at the speed of light (299792458 m/s), how many seconds to reach the Moon (384400 km)? Round to 1 decimal.", "1.3"),
    ("How many Moon diameters (3474 km) fit across Jupiter's diameter (139820 km)? Round to nearest whole number.", "40"),
    ("If you ran a marathon (42.195 km) at cheetah speed (120 km/h), how many minutes would it take? Round to nearest whole number.", "21"),
    ("Mars diameter is 6779 km, Earth diameter is 12742 km. How many times bigger is Earth than Mars? Round to 1 decimal.", "1.9"),
    ("If the Pacific Ocean (165.25 million sq km) were a square, what would the side length be in km? Round to nearest whole number.", "12855"),
    ("Saturn has 146 moons, Jupiter has 95. What percentage of their combined moons belong to Jupiter? Round to nearest whole number.", "39"),
    ("How many football fields (5351 sq meters each) fit in Central Park (3.41 sq km)? Round to nearest whole number.", "637"),
    ("If India (1.4 billion people) each used 100 liters of water per day, how many billion liters total per day?", "140"),
    ("The speed of light is 299792458 m/s. How many km can light travel in one minute? Round to nearest whole number.", "17987547"),
]

DISTRACTOR_QS = [
    ("What color is the sky on a clear day?", "blue"),
    ("How many days are in a week?", "7"),
    ("Is the sun a star or a planet?", "star"),
    ("How many continents are there?", "7"),
    ("What comes after the number 99?", "100"),
    ("How many letters are in the English alphabet?", "26"),
    ("What is the opposite of hot?", "cold"),
    ("How many sides does a triangle have?", "3"),
    ("How many hours are in a day?", "24"),
    ("What color do you get mixing red and blue?", "purple"),
    ("How many legs does a spider have?", "8"),
    ("How many months are in a year?", "12"),
    ("What is 1 + 1?", "2"),
    ("How many wheels does a bicycle have?", "2"),
    ("How many fingers does a human hand have?", "5"),
    ("What color is a banana?", "yellow"),
    ("How many seasons are there?", "4"),
    ("What is the opposite of up?", "down"),
    ("How many sides does a square have?", "4"),
    ("What is the first letter of the alphabet?", "a"),
    ("Is ice solid, liquid, or gas?", "solid"),
    ("What sound does a cat make?", "meow"),
    ("How many toes does a human foot have?", "5"),
    ("What is the opposite of left?", "right"),
    ("How many minutes are in an hour?", "60"),
]


# ─── Dataset Builder ─────────────────────────────────────────────────────────

def _make_arithmetic(rng, n):
    """Generate arithmetic questions programmatically."""
    questions = []
    ops = [("+", "plus"), ("-", "minus"), ("*", "times")]
    for _ in range(n):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op_sym, op_name = rng.choice(ops)
        result = eval(f"{a} {op_sym} {b}")
        questions.append({
            "question": f"What is {a} {op_name} {b}?",
            "answer": str(result),
            "info": '{"category": "arithmetic", "optimal_calls": 1}',
        })
    return questions


def _format_pool(pool, category, optimal_calls):
    """Format a question pool into dataset rows."""
    info = f'{{"category": "{category}", "optimal_calls": {optimal_calls}}}'
    return [{"question": q, "answer": a, "info": info} for q, a in pool]


def build_dataset(split: str = "train") -> Dataset:
    """Build the tool-routing dataset with train/eval partition."""
    data = []

    # Arithmetic (generated, different seed per split)
    seed = 42 if split == "train" else 99
    rng = random.Random(seed)
    n_arith = 80 if split == "train" else 15
    data.extend(_make_arithmetic(rng, n_arith))

    # Partition hardcoded pools: first N for train, rest for eval
    pools = [
        (FACTUAL_QS, "factual", 1, 28, 10),
        (CONVERSION_QS, "conversion", 1, 22, 8),
        (COMPUTATION_QS, "complex_computation", 1, 18, 7),
        (FACT_CALC_QS, "fact_plus_calc", 2, 15, 5),
        (MULTISTEP_QS, "multi_step", 3, 10, 5),
        (DISTRACTOR_QS, "distractor", 0, 18, 7),
    ]

    for pool, category, optimal_calls, n_train, n_eval in pools:
        # Stable shuffle for consistent partition
        shuffled = pool.copy()
        random.Random(0).shuffle(shuffled)
        if split == "train":
            selected = shuffled[:n_train]
        else:
            selected = shuffled[n_train:n_train + n_eval]
        data.extend(_format_pool(selected, category, optimal_calls))

    # Shuffle final dataset
    random.Random(seed).shuffle(data)
    return Dataset.from_list(data)


# ─── Reward Functions ────────────────────────────────────────────────────────

async def correctness(completion, answer) -> float:
    """1.0 if ground truth answer appears in model's final response."""
    for msg in reversed(completion):
        if msg.get("role") == "assistant" and msg.get("content"):
            response = msg["content"].lower()
            target = answer.strip().lower()
            return 1.0 if target in response else 0.0
    return 0.0


async def efficiency(completion) -> float:
    """Reward efficiency: 1/sqrt(n_tool_calls). Fewer calls = higher reward."""
    n_calls = sum(1 for m in completion if m.get("role") == "tool")
    return 1.0 / math.sqrt(max(1, n_calls))


async def tool_call_count(completion) -> float:
    """Observable metric: number of tool calls made."""
    return float(sum(1 for m in completion if m.get("role") == "tool"))


# ─── Entry Point ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Answer questions accurately using the minimum number of tool calls necessary. "
    "If you can answer a question from your own knowledge without using any tools, do so directly."
)


def load_environment(**kwargs) -> vf.Environment:
    """Load the tool-routing environment."""
    rubric = vf.Rubric(funcs=[correctness, efficiency], weights=[0.7, 0.3])
    rubric.add_metric(tool_call_count)

    return vf.ToolEnv(
        dataset=build_dataset("train"),
        eval_dataset=build_dataset("test"),
        tools=[web_search, wikipedia_lookup, calculator, unit_converter, wolfram_alpha, python_eval],
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
        max_turns=8,
    )
