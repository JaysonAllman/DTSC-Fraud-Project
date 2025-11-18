import re

KEYWORDS = {
    "Phishing/Spoofing": [
        "Phishing", "Phish", "Phished",
        "Spoof", "Spoofed", "Spoofing",
        "Smish", "Smishing",
        "Spearphishing", "Spear-Phishing", "Spearphished"
    ],

    "Extortion": [
        "Extortion", "Extort", "Extorted", "Extorting",
        "Blackmail", "Blackmailed", "Blackmailing", "Blackmails"
    ],

    "Personal Data Breach": [
        "Personal Data Breach", "Personal Data Breaches",
        "Personal Data",
        "Personal Data Breached", "Personal Data Leak", "Personal Data Leaked",
        "Personal Data Exposure", "Personal Data Exposed"
    ],

    "Non-Payment/Non-Delivery": [
        "Non-Payment", "Non-Payments",
        "Non-Delivery", "Non-Deliveries",
        "Non Payment", "Non Delivery",
        "No Payment", "No Delivery",
        "Unpaid", "Undelivered"
    ],

    "Investment": [
        "Investment", "Investments",
        "Investor", "Investors",
        "Investing", "Invested",
        "Retirement",
        "401k", "401(k)",
        "Ponzi", "Pyramid Scheme"
    ],

    "Tech Support": [
        "Tech Support", "Technical Support",
        "Tech Help", "Technical Help",
        "Support Scam", "Support Scams",
        "Tech Support Scam", "Tech Support Scams"
    ],

    "Business Email Compromise": [
        "Business Email Compromise", "BEC",
        "Email Compromise", "Compromised Email",
        "Business Email", "Business Emails"
    ],

    "Identity Theft": [
        "Identity Theft", "Identities Stolen",
        "Stolen Identity", "Stolen Identities",
        "ID Theft", "ID Thefts",
        "Identity Fraud", "Identity Frauds",
        "Identity Stolen"
    ],

    "Employment": [
        "Employment Fraud", "Employment Scams",
        "Job Scam", "Job Scams",
        "Fake Job Offer", "Fake Job Offers"
    ],

    "Confidence/Romance": [
        "Romance Fraud", "Romance Frauds",
        "Romance Scam", "Romance Scams",
        "Confidence Fraud", "Confidence Frauds",
        "Confidence Scam", "Confidence Scams",
        "Dating Scam", "Dating Scams",
        "Grandparent Scam", "Grandparent Scams",
        "Grandparents Scam", "Grandparents Scams"
    ],

    "Government Impersonation": [
        "Government Impersonation", "Government Impersonator",
        "Government Impersonators",
        "Impersonate Officer", "Impersonated Officer",
        "Fake Government Agent", "Fake Government Agents",
        "Government Scam", "Government Scams"
    ],

    "Credit Card/Check Fraud": [
        "Credit Card Fraud", "Credit Card Frauds",
        "Credit Card", "Credit Cards",
        "Card Fraud", "Card Frauds",
        "Check Fraud", "Check Frauds",
        "Fake Check", "Fake Checks",
        "ACH", "EFT", "Recurring Charge", "Recurring Charges"
    ],

    "Harassment/Stalking": [
        "Harassment", "Harassing",
        "Harassed", "Stalking", "Stalker", "Stalkers"
    ],

    "Real Estate": [
        "Real Estate", "Real Estate Scam",
        "Real Estate Scams", "Rental"
        "Rental Scam", "Rental Scams",
        "Timeshare", "Timeshares"
    ],

    "Advanced Fee": [
        "Advanced Fee", "Advanced Fees",
        "Advance Fee", "Advance Fees",
        "Prepaid", "Prepayment", "Upfront Fee", "Upfront Fees"
    ],

    "Crimes Against Children": [
        "Crimes Against Children",
        "Child Abuse", "Child Abused",
        "Children", "Minor", "Minors",
        "Underage", "Child"
    ],

    "Lottery/Sweepstakes/Inheritance": [
        "Lottery", "Lotteries",
        "Sweepstakes", "Sweepstake",
        "Inheritance", "Inherited",
        "Prize", "Prizes",
        "Winning", "Winnings",
    ],

    "Data Breach": [
        "Data Breach", "Data Breaches",
        "Data Breached",
        "Data Leak", "Data Leaks",
        "Data Leaked", "Data Exposure", "Data Exposed"
    ],

    "Ransomware": [
        "Ransomware",
        "Ransom", "Ransoms",
        "Blocked Access", "Block Access",
    ],

    "Overpayment": [
        "Overpayment", "Overpayments",
        "Overpaid", "Commission Scam", "Commission Scams",
        "Fake Check", "Fake Checks"
    ],

    "IPR*/Copyright&Counterfeit": [
        "IPR", "Intellectual Property",
        "Copyright Fraud", "Copyright Infringement",
        "Counterfeit", "Counterfeits",
        "Trademark", "Trademarks",
        "Patent", "Patents",
        "Piracy", "Pirated",
        "Trade Secrets", "Pirated Movies", "Pirated Music"
    ],

    "Threats of Violence": [
        "Threats of Violence",
        "Threatened", "Threatening",
        "Pain", "Injury", "Injuries",
        "Death Threat", "Death Threats",
        "Self-Harm", "Harm", "Violence"
    ],

    "SIM Swap": [
        "SIM Swap",
        "SIM Swapping", "SIM Swapped",
        "Mobile Service Provider",
        "Phone Service"
    ],

    "Botnet": [
        "Botnet", "Botnets",
        "Distributed Denial of Service",
        "DDoS", "Denial of Service",
        "Telephony Denial of Service",
        "Phone Denial of Service"
    ],

    "Malware": [
        "Malware",
        "Virus", "Viruses",
        "Trojan", "Trojans", "Trojanized"
        "Spyware", "Worm", "Worms",
        "Damage Data", "Destroy Data",
        "Malicious Software"
    ],

    "Cryptocurrency": [
        "Cryptocurrency", "Cryptocurrencies",
        "Bitcoin", "Ethereum",
        "Crypto", "Crypto Scam", "Crypto Scams",
        "Digital Currency", "Digital Currencies"
    ],
}


def build_regex(keywords):
    patterns = []
    for term in keywords:
        escaped = re.escape(term)
        if " " in term:  
            # Multi-word phrase → match as-is
            patterns.append(rf"\b{escaped}\b")
        else:
            # Single word → strict whole-word match
            patterns.append(rf"\b{escaped}\b")
    return re.compile("|".join(patterns), re.I)

FRAUD_REGEX = {category: build_regex(terms) for category, terms in KEYWORDS.items()}