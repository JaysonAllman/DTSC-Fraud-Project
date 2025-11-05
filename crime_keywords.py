import re

CRIME_PATTERNS = {
    "phishing": ["phishing", "phishings", "spoofing", "spoofings", "email scam", "email scams", "email fraud", "email frauds", "fake email", "fake emails", "smishing", "smishings", "vishing", "vishings"],
    "extortion": ["extortion", "extortions", "blackmail", "blackmails", "ransom demand", "ransom demands"],
    "personal_data_breach": ["data breach", "data breaches", "information leak", "information leaks", "personal data exposure", "personal data exposures"],
    "non_payment_non_delivery": ["non-payment", "non-payments", "non delivery", "non deliveries", "failed delivery", "failed deliveries", "did not pay", "didn't pay"],
    "investment": ["investment fraud", "investment frauds", "ponzi scheme", "ponzi schemes", "pyramid scheme", "pyramid schemes", "securities fraud", "securities frauds"],
    "tech_support": ["tech support scam", "tech support scams", "fake tech help", "fake tech helps", "technical support fraud", "technical support frauds"],
    "business_email_compromise": ["business email compromise", "business email compromises", "bec scam", "bec scams", "ceo fraud", "ceo frauds"],
    "identity_theft": ["identity theft", "identity thefts", "id theft", "id thefts", "stolen identity", "stolen identities", "identity fraud", "identity frauds"],
    "employment": ["employment scam", "employment scams", "job scam", "job scams", "fake job posting", "fake job postings"],
    "confidence_romance": ["romance scam", "romance scams", "dating scam", "dating scams", "catfishing", "catfishings"],
    "government_impersonation": ["government impersonation", "government impersonations", "fake irs", "fake polices", "fake police", "fake fbi"],
    "credit_card_check_fraud": ["credit card fraud", "credit card frauds", "check fraud", "check frauds", "bank fraud", "bank frauds"],
    "harassment_stalking": ["harassment", "harassments", "stalking", "stalkings", "cyberstalking", "cyberstalkings"],
    "real_estate": ["real estate fraud", "real estate frauds", "property scam", "property scams", "mortgage fraud", "mortgage frauds"],
    "advanced_fee": ["advanced fee scam", "advanced fee scams", "upfront fee", "upfront fees", "prepayment scam", "prepayment scams"],
    "crimes_against_children": ["child exploitation", "child exploitations", "child abuse", "child abuses", "child pornography", "child pornographies"],
    "lottery_sweepstakes_inheritance": ["lottery scam", "lottery scams", "sweepstakes fraud", "sweepstakes frauds", "inheritance scam", "inheritance scams"],
    "ransomware": ["ransomware", "ransoms", "malware ransom", "malware ransoms"],
    "overpayment": ["overpayment scam", "overpayment scams", "refund scam", "refund scams"],
    "ipr_copyright_counterfeit": ["copyright infringement", "copyright infringements", "counterfeit goods", "counterfeit good", "fake products", "fake product"],
    "threats_of_violence": ["threats", "threat", "violence threats", "violence threat"],
    "sim_swap": ["sim swap", "sim swaps", "sim hijacking", "sim hijackings"],
    "botnet": ["botnet", "botnets", "ddos network", "ddos networks"],
    "malware": ["malware", "malwares", "virus infection", "virus infections", "trojan", "trojans"],
    "cryptocurrency": ["crypto scam", "crypto scams", "bitcoin fraud", "bitcoin frauds", "cryptocurrency scam", "cryptocurrency scams", "crypto theft", "crypto thefts"]
}


FRAUD_REGEX = {k: re.compile("|".join([re.escape(term) for term in v]), re.I) for k, v in CRIME_PATTERNS.items()}