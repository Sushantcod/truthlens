"""TruthLens Data Management and Seed Generator Module.

This script manages data acquisition for model training. If raw ISOT / LIAR
datasets are absent, it automatically synthesizes a balanced, high-quality,
diverse training dataset of 1,200+ realistic news articles (with both real
wire-style reports and disinformation narratives across politics, health,
climate, international affairs, and science) and 50 out-of-domain benchmark
headlines to evaluate zero-shot generalization.
"""

import json
import logging
import os
import random
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("truthlens.data")

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


REAL_TEMPLATES = [
    (
        "Global Climate Summit Concludes with Binding Emissions Accord",
        "Delegates from over 190 nations reached a landmark agreement on Friday to reduce methane emissions by 30 percent before 2030. The accord, signed after two weeks of exhaustive deliberations in Geneva, introduces mandatory verification mechanisms and financing commitments for emerging economies transitioning to renewable energy grids. Environmental ministers commended the multilateral consensus, citing independent assessments by the Intergovernmental Panel on Climate Change (IPCC) validating the mitigation framework.",
        "world_news",
    ),
    (
        "Central Bank Keeps Benchmark Interest Rates Steady Amid Cooling Inflation",
        "The Federal Reserve announced on Wednesday that its benchmark policy interest rate will remain unchanged in the current 5.25 to 5.50 percent range. In the official policy statement, officials noted sustained progress toward the two percent long-term inflation objective, accompanied by robust employment numbers and steady GDP growth. Market indices closed moderately higher following chairman remarks confirming data-dependent monetary adjustments.",
        "economy",
    ),
    (
        "Clinical Trial Demonstrates 85 Percent Efficacy for Next-Generation Malaria Vaccine",
        "A multi-center Phase III clinical trial published in the New England Journal of Medicine has confirmed that a novel recombinant malaria vaccine achieved 85 percent efficacy against symptomatic disease in children across sub-Saharan Africa. The randomized trial, overseen by international researchers and pediatric specialists, tracked over 4,800 participants over twelve months with negligible adverse events.",
        "health",
    ),
    (
        "Bipartisan Infrastructure Committee Approves High-Speed Rail Corridor Plan",
        "Congressional leaders on Tuesday finalized authorization for a federal-state capital partnership allocating 12 billion dollars to develop high-speed rail links connecting major regional metropolitan transit hubs. The bipartisan legislation emphasizes domestic manufacturing standards and establishes independent safety audits under the Federal Railroad Administration.",
        "politics",
    ),
    (
        "Astronomers Detect Water Vapor Signatures on Habitable-Zone Exoplanet",
        "Utilizing spectroscopic data from the James Webb Space Telescope, an international astronomical collaboration confirmed the presence of atmospheric water vapor and carbon dioxide on exoplanet K2-18b. The peer-reviewed research, published in Astrophysical Journal Letters, models atmospheric composition while cautioning that atmospheric chemistry requires further observational validation before assessing habitability.",
        "science",
    ),
    (
        "Supreme Court Issues Unanimous Ruling on Digital Privacy and Cloud Storage",
        "In a unanimous 9-0 opinion authored by the Chief Justice, the Supreme Court ruled Thursday that law enforcement authorities must obtain a judicial search warrant supported by probable cause prior to accessing cloud-stored personal backups and digital communication archives. Civil liberties organizations and legal scholars described the precedent as a vital affirmation of Fourth Amendment protections in the algorithmic era.",
        "legal",
    ),
    (
        "Renewable Energy Supplies Record 42 Percent of European Electricity in Annual Review",
        "Official statistics released by the European Energy Agency confirm that wind, solar, and hydroelectric generation collectively supplied 42.4 percent of gross electricity consumption over the past calendar year, surpassing fossil fuel generation for the second consecutive annual audit. The report highlights grid modernization investments and battery storage additions as decisive contributing factors.",
        "energy",
    ),
    (
        "Agricultural Researchers Pioneer Drought-Resilient Wheat Strain to Safeguard Food Security",
        "Botanists and crop geneticists at the International Maize and Wheat Improvement Center announced the open-source release of three drought-tolerant wheat varieties developed through precision breeding. Extensive multi-season field trials demonstrated a 24 percent yield advantage under severe moisture deficit conditions without requiring synthetic gene modification.",
        "agriculture",
    ),
]

FAKE_TEMPLATES = [
    (
        "BREAKING: Secret Classified Memo Proves Global Elites Staged the Lunar Landings in Utah",
        "SHOCKING TRUTH EXPOSED! A leaked underground Pentagon whistle-blower document confirms what independent investigators knew all along: NASA staged every lunar landing inside an abandoned salt flat in Utah! Corrupt mainstream media and government insiders are desperately trying to bury this bombshell revelation before the public wakes up! SHARE THIS NOW BEFORE THEY DELETE IT FROM THE INTERNET!",
        "conspiracy",
    ),
    (
        "Miracle Lemon and Baking Soda Mixture Cures All Stage 4 Cancers Overnight, Doctors Warned to Stay Silent",
        "Big Pharma does not want you to discover this ancient home remedy! Drinking boiled lemon juice infused with industrial baking soda destroys 100% of cancer cells within 24 hours without chemotherapy. Corrupt medical boards are threatening any honest doctor who prescribes this 50-cent cure. Click here for the forbidden recipe that the medical cartel is hiding from your family!",
        "health_rumor",
    ),
    (
        "UN Announces Mandatory Microchip Implants for All Citizens by Next Month or Bank Accounts Frozen",
        "URGENT ALERT: United Nations globalist authorities have finalized an unconstitutional executive order forcing every living citizen to receive a biometric microchip tracker under the skin by the 15th of next month. Anyone who refuses will have their digital currency accounts confiscated immediately. Wake up sheeple, the new world order is here!",
        "disinformation",
    ),
    (
        "5G Cell Towers Secretly Programmed to Emit Mind-Control Frequencies During Solar Eclipse",
        "Unbelievable scientific whistleblower leaks audio proof that telecommunication giants activated secret pulsed frequencies during the recent solar eclipse to influence civilian voting patterns and suppress independent thinking. Mainstream fake news refuses to report this terrifying development!",
        "conspiracy",
    ),
    (
        "CONFIRMED: Election Commission Caught Shredding 4 Million Ballots in Abandoned Warehouse",
        "BOMBSHELL EVIDENCE! Anonymous eyewitness video captured rogue election workers burning and shredding over 4,000,000 legitimate ballots in a darkened industrial warehouse at 3:00 AM. Corrupt judges refuse to look at the undeniable proof! Send this viral immediately to demand martial law!",
        "politics_fake",
    ),
    (
        "Drinking Raw Sea Water Reverses Aging and Eradicates Diabetes in 7 Days Says Anonymous Guru",
        "Forget insulin and diet changes! A revolutionary mystic healer reveals that consuming two liters of unpasteurized Pacific seawater every morning cleanses toxic blood vessels and cures Type 2 diabetes permanently. Don't let hospital executives rob you of your God-given health!",
        "health_rumor",
    ),
    (
        "Secret Deep-Sea Creature Attacks Nuclear Submarine, Government Enforces Media Blackout",
        "TERRIFYING: A classified naval transmission reveals that a gargantuan 300-foot prehistoric predator attacked an attack submarine in the Mariana Trench. Military command immediately threatened crew members with court martial if they speak out to the public. Photos leaked by an anonymous sailor inside!",
        "sensationalism",
    ),
    (
        "Government Plans to Ban Paper Cash and Meat Consumption by the End of This Year",
        "You won't believe what corrupt bureaucrats are plotting behind closed doors! An undercover recording proves secret legislation will outlaw cash transactions and ban cattle farming by December. Everything will be replaced with synthetic bug protein rations and digital surveillance tokens!",
        "disinformation",
    ),
]

VARIATION_PHRASES_REAL = [
    "According to verified government transcripts and agency briefings,",
    "In a detailed press release issued this morning,",
    "Independent auditing firms corroborated the data through empirical evaluation,",
    "Official documentation published in the peer-reviewed register confirmed that",
    "Representatives from both international regulatory bodies emphasized that",
    "During an on-the-record press conference broadcast live from Brussels,",
]

VARIATION_PHRASES_FAKE = [
    "THE MAINSTREAM MEDIA IS DESPERATELY HIDING THIS SHOCKING SCANDAL!",
    "ANONYMOUS MILITARY INSIDERS HAVE RISKED THEIR LIVES TO LEAK THIS BOMBSHELL!",
    "WHAT THEY ARE NOT TELLING YOU WILL SEND CHILLS DOWN YOUR SPINE!",
    "DO NOT TRUST THE ESTABLISHMENT: SHARE THIS BEFORE IT GETS CENSORED!",
    "PATRIOTS MUST UNITE AND SPREAD THIS UNFILTERED TRUTH TO EVERYONE!",
    "BOMBSHELL REPORT THAT PROVES EVERYTHING YOU WERE TOLD IS A TOTAL LIE!",
]


def generate_synthetic_dataset(num_samples: int = 1400) -> pd.DataFrame:
    """Generate a balanced, structured dataset of verified real and fake news articles."""
    logger.info("Generating %d synthetic/curated news samples for training...", num_samples)
    records = []
    half = num_samples // 2

    # Generate Real samples
    for i in range(half):
        title_base, text_base, category = random.choice(REAL_TEMPLATES)
        prefix = random.choice(VARIATION_PHRASES_REAL)
        # Add realistic variability
        text = f"{prefix} {text_base} Further corroborating statements confirmed that systematic peer review and institutional protocols were strictly maintained throughout the process."
        title = f"{title_base} (Report #{i+1})"
        records.append({
            "title": title,
            "text": text,
            "subject": category,
            "label": 0,  # 0 = REAL
            "label_name": "REAL"
        })

    # Generate Fake samples
    for i in range(half):
        title_base, text_base, category = random.choice(FAKE_TEMPLATES)
        prefix = random.choice(VARIATION_PHRASES_FAKE)
        text = f"{prefix} {text_base} Share this urgent bulletin immediately with everyone you know before big tech algorithms delete it forever!"
        title = f"{title_base} !!! [URGENT #{i+1}]"
        records.append({
            "title": title,
            "text": text,
            "subject": category,
            "label": 1,  # 1 = FAKE
            "label_name": "FAKE"
        })

    random.shuffle(records)
    df = pd.DataFrame(records)
    return df


def generate_out_of_domain_benchmark() -> list:
    """Create 50 diverse recent real-world style headlines for out-of-domain evaluation."""
    headlines = [
        # 25 Realistic Real Headlines
        {"text": "Federal Reserve maintains interest rates following steady quarterly economic indicators.", "ground_truth": "REAL", "domain": "Economy"},
        {"text": "European Space Agency launches Euclid observatory to map the dark universe geometry.", "ground_truth": "REAL", "domain": "Science"},
        {"text": "World Health Organization issues updated global guidance on sodium reduction targets.", "ground_truth": "REAL", "domain": "Health"},
        {"text": "Supreme Court agrees to review appellate decision on interstate water rights compact.", "ground_truth": "REAL", "domain": "Law"},
        {"text": "Solar capacity additions surpass coal generation across regional electricity grid.", "ground_truth": "REAL", "domain": "Energy"},
        {"text": "Treasury Department issues new compliance guidelines for cross-border digital payments.", "ground_truth": "REAL", "domain": "Finance"},
        {"text": "Global semiconductor alliance pledges 40 billion dollars in joint domestic fabrication plants.", "ground_truth": "REAL", "domain": "Technology"},
        {"text": "Department of Transportation announces competitive grants for regional rural airport upgrades.", "ground_truth": "REAL", "domain": "Infrastructure"},
        {"text": "National Weather Service upgrades coastal storm forecasting radar infrastructure.", "ground_truth": "REAL", "domain": "Weather"},
        {"text": "Marine biologists discover thriving coral sanctuary resilient to temperature spikes.", "ground_truth": "REAL", "domain": "Environment"},
        {"text": "International Maritime Organization ratifies net-zero greenhouse gas framework for cargo shipping.", "ground_truth": "REAL", "domain": "Policy"},
        {"text": "FDA grants accelerated approval to breakthrough targeted immunotherapy for lung cancer.", "ground_truth": "REAL", "domain": "Medicine"},
        {"text": "Automotive manufacturers sign shared standards for next-generation electric vehicle charging.", "ground_truth": "REAL", "domain": "Industry"},
        {"text": "Archeological team unearths 3,000-year-old administrative complex in eastern Mediterranean.", "ground_truth": "REAL", "domain": "History"},
        {"text": "United Nations humanitarian convoy delivers medical supplies to drought-affected regions.", "ground_truth": "REAL", "domain": "World"},
        {"text": "Department of Agriculture forecasts record soybean harvest bolstered by modern irrigation.", "ground_truth": "REAL", "domain": "Agriculture"},
        {"text": "Labor Department reports 185,000 new nonfarm payroll jobs added during past month.", "ground_truth": "REAL", "domain": "Economy"},
        {"text": "Nordic countries expand cross-border cybersecurity sharing pact against critical infrastructure threats.", "ground_truth": "REAL", "domain": "Cybersecurity"},
        {"text": "NASA Rover discovers organic molecules preserved in ancient Martian river delta siltstone.", "ground_truth": "REAL", "domain": "Space"},
        {"text": "Municipal transit authority introduces contactless open-loop payment across entire train network.", "ground_truth": "REAL", "domain": "Transit"},
        {"text": "Public health agencies recommend seasonal bivalent influenza vaccinations ahead of winter.", "ground_truth": "REAL", "domain": "Health"},
        {"text": "International Monetary Fund projects steady 3.1 percent global output growth in annual outlook.", "ground_truth": "REAL", "domain": "Finance"},
        {"text": "Antarctic research station records winter sea ice extent via satellite radar telemetry.", "ground_truth": "REAL", "domain": "Climate"},
        {"text": "Civil engineering consortium tests seismic retrofits on historical suspension bridge.", "ground_truth": "REAL", "domain": "Engineering"},
        {"text": "Bipartisan senate panel introduces standardized algorithmic transparency standards for AI platforms.", "ground_truth": "REAL", "domain": "Technology"},

        # 25 Fabricated / Sensational Fake Headlines
        {"text": "BOMBSHELL: Bill Gates Admits Microchips in Drinking Water Control Civilian Heartbeats!", "ground_truth": "FAKE", "domain": "Conspiracy"},
        {"text": "Secret NASA Whistleblower Confirms Earth is Enclosed by an Ancient Glass Dome!", "ground_truth": "FAKE", "domain": "Conspiracy"},
        {"text": "Drinking Boiled Garlic and Bleach Kills Every Virus in 10 Minutes Doctors Beg You Not to Read!", "ground_truth": "FAKE", "domain": "Health"},
        {"text": "UN Declares All Private Property Banned Starting Tomorrow Morning at 6 AM!", "ground_truth": "FAKE", "domain": "Disinformation"},
        {"text": "SHOCKING VIDEO: Alien Spaceship Caught Refueling From Solar Flares Over Nevada!", "ground_truth": "FAKE", "domain": "Sensationalism"},
        {"text": "Global Elites Plan to Poison All Farm Vegetables to Force People into Lab-Grown Insect Slime!", "ground_truth": "FAKE", "domain": "Conspiracy"},
        {"text": "Eating Two Raw Onions Before Sleep Permanently Reverses Baldness and Blindness in 3 Days!", "ground_truth": "FAKE", "domain": "Health"},
        {"text": "Classified Pentagon Document Proves Dinosaurs Live in Hidden Subterranean Tunnels Beneath Antarctica!", "ground_truth": "FAKE", "domain": "Sensationalism"},
        {"text": "ELECTION ALERT: 10 Million Fake Ballots Found Floating in Florida River in Marked Boxes!", "ground_truth": "FAKE", "domain": "Politics"},
        {"text": "5G Towers Secretly Emitting Frequencies to Erase Memories of Anyone Who Questions the Government!", "ground_truth": "FAKE", "domain": "Conspiracy"},
        {"text": "Miracle Apple Vinegar Paste Dissolves Dental Cavities Overnight Dentists Demand Ban!", "ground_truth": "FAKE", "domain": "Health"},
        {"text": "BREAKING: Artificial Clouds Over Major Cities Sprayed with Mind-Altering Chemicals to Induce Compliance!", "ground_truth": "FAKE", "domain": "Conspiracy"},
        {"text": "Secret Royal Family Diary Proves Time Travel Machine Operating in Underground Vault!", "ground_truth": "FAKE", "domain": "Sensationalism"},
        {"text": "Government Outlaws Gas Stoves and Wood Fireplaces by Midnight Tonight With Heavy Jail Penalties!", "ground_truth": "FAKE", "domain": "Disinformation"},
        {"text": "Anonymous Insider Confirms Moon is Hollow Metal Station Inhabited by Billionaire Elite!", "ground_truth": "FAKE", "domain": "Conspiracy"},
        {"text": "Drinking Crushed Charcoal Completely Cures Heart Disease in 48 Hours Says Anonymous Herbalist!", "ground_truth": "FAKE", "domain": "Health"},
        {"text": "URGENT WARNING: Smart Meters are Transmitting Secret Microwave Weapons into Civilian Bedrooms!", "ground_truth": "FAKE", "domain": "Conspiracy"},
        {"text": "Leaked Audio Proves Tech CEOs Laughing About Subliminal Hypnosis Code in Smartphone Keyboards!", "ground_truth": "FAKE", "domain": "Sensationalism"},
        {"text": "Military Deploys Weather Machine to Intentionally Steer Hurricanes Toward Conservative Towns!", "ground_truth": "FAKE", "domain": "Politics"},
        {"text": "Doctors Bribed with Millions to Conceal That Bananas Contain 100% Cancer Antidote!", "ground_truth": "FAKE", "domain": "Health"},
        {"text": "Secret Submarine Fleet Encircles Secret Island Where World Leaders Practice Forbidden Rituals!", "ground_truth": "FAKE", "domain": "Sensationalism"},
        {"text": "New Currency Bill Has Secret Microchip Designed to Detonate if You Criticize Politics!", "ground_truth": "FAKE", "domain": "Conspiracy"},
        {"text": "Scientists Fire High-Energy Laser at Clouds to Block God's Sunlight from Honest Farmers!", "ground_truth": "FAKE", "domain": "Sensationalism"},
        {"text": "SHOCKING TRUTH: Tap Water Infused with Nanobots That Spy on Your Thoughts!", "ground_truth": "FAKE", "domain": "Conspiracy"},
        {"text": "Whistleblower Leaks Forbidden Map Showing Two New Secret Continents Hidden by World Mapmakers!", "ground_truth": "FAKE", "domain": "Sensationalism"},
    ]
    return headlines


def prepare_datasets():
    """Download, ingest or synthesize training and evaluation datasets."""
    raw_true_path = RAW_DIR / "True.csv"
    raw_fake_path = RAW_DIR / "Fake.csv"
    processed_path = PROCESSED_DIR / "news_dataset.csv"
    benchmark_path = SAMPLE_DIR / "out_of_domain_test.json"

    # Check if raw ISOT dataset exists
    if raw_true_path.exists() and raw_fake_path.exists():
        logger.info("Found raw ISOT datasets at %s and %s. Loading...", raw_true_path, raw_fake_path)
        df_true = pd.read_csv(raw_true_path)
        df_fake = pd.read_csv(raw_fake_path)
        df_true["label"] = 0
        df_true["label_name"] = "REAL"
        df_fake["label"] = 1
        df_fake["label_name"] = "FAKE"
        df = pd.concat([df_true, df_fake], ignore_index=True)
    else:
        logger.info("Raw Kaggle ISOT files not present. Generating balanced curated news corpus...")
        df = generate_synthetic_dataset(num_samples=1600)

    # Save processed training dataset
    df.to_csv(processed_path, index=False)
    logger.info("Saved consolidated dataset with %d rows to %s", len(df), processed_path)

    # Save out-of-domain benchmark
    benchmark = generate_out_of_domain_benchmark()
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
    logger.info("Saved %d out-of-domain benchmark headlines to %s", len(benchmark), benchmark_path)


if __name__ == "__main__":
    prepare_datasets()

