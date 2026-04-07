"""
Generate EN→DE legal glossary using GPT-5.4.
Maps English legal terms to their German equivalents in Swiss law.
"""
import json
import os
import sys
from pathlib import Path
from openai import OpenAI

# Load env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI()

LEGAL_DOMAINS = [
    "criminal law (StGB)",
    "criminal procedure (StPO)",
    "civil law - obligations/contracts (OR)",
    "civil law - family law (ZGB)",
    "civil law - property/inheritance (ZGB)",
    "civil procedure (ZPO)",
    "administrative law (VwVG)",
    "constitutional law (BV)",
    "social insurance - disability (IVG/ATSG)",
    "social insurance - accident (UVG)",
    "social insurance - unemployment (AVIG)",
    "social insurance - health (KVG)",
    "immigration/asylum (AIG/AsylG)",
    "tax law (DBG/StHG)",
    "environmental law (USG/UVPV)",
    "planning/construction (RPG)",
    "competition/antitrust (KG)",
    "intellectual property (URG/PatG/MSchG)",
    "debt enforcement/bankruptcy (SchKG)",
    "federal court procedure (BGG)",
    "labor law (ArG)",
    "tenancy law (OR 253ff)",
    "corporate law (OR 620ff)",
    "data protection (DSG)",
    "transport/traffic law (SVG)",
    "energy law (EnG)",
    "banking/finance law (BankG/FINMAG)",
    "public procurement (BöB)",
    "international private law (IPRG)",
    "unfair competition (UWG)",
]


def generate_glossary_for_domain(domain: str) -> dict:
    """Generate EN→DE legal term mappings for a specific domain."""
    resp = client.chat.completions.create(
        model="gpt-5.4-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Swiss legal terminology expert fluent in English and German. "
                    "Generate a comprehensive glossary mapping English legal terms to their "
                    "German equivalents as used in Swiss federal law. Include statutory terms, "
                    "doctrinal concepts, and common legal phrases. Output JSON with the format: "
                    '{"terms": [{"en": "...", "de": "...", "statute": "..."}]} '
                    "where statute is the relevant Swiss law abbreviation (e.g., OR, ZGB, StGB). "
                    "Generate at least 40 terms per domain."
                ),
            },
            {
                "role": "user",
                "content": f"Generate the glossary for Swiss {domain}.",
            },
        ],
    )
    return json.loads(resp.choices[0].message.content)


def main():
    output_path = Path(__file__).parent.parent / "precompute" / "legal_glossary.json"

    glossary = {}
    for i, domain in enumerate(LEGAL_DOMAINS):
        print(f"[{i+1}/{len(LEGAL_DOMAINS)}] Generating glossary for: {domain}")
        try:
            result = generate_glossary_for_domain(domain)
            glossary[domain] = result.get("terms", [])
            print(f"  -> {len(glossary[domain])} terms")
        except Exception as e:
            print(f"  ERROR: {e}")
            glossary[domain] = []

    # Also build flat lookup
    flat = {}
    for domain, terms in glossary.items():
        for t in terms:
            en = t.get("en", "").lower()
            if en and en not in flat:
                flat[en] = {
                    "de": t.get("de", ""),
                    "statute": t.get("statute", ""),
                    "domain": domain,
                }

    output = {"by_domain": glossary, "flat_lookup": flat}
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(flat)} unique terms to {output_path}")


if __name__ == "__main__":
    main()
