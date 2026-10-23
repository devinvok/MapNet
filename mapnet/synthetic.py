"""
Synthetic 'company' to test the map on: 6 departments x 4 topics, each topic
with N systems (sources) of documents, plus optional chat channels full of
chatter that name-drops the same projects and vendors (the realistic noise
that confuses a flat search).

Every fact document is about one unique entity (a project / vendor / ticket
name). Each question names an entity plus a few topic words that only partly
overlap the gold document's wording, so lexical matching is imperfect.
"""
import random

TOPICS = {
    "hr": {
        "payroll": "salary payslip bonus deduction overtime payday compensation wage raise withholding payroll stipend",
        "benefits": "insurance dental pension parental leave wellness gym coverage enrollment vision medical childcare",
        "recruiting": "candidate interview offer recruiter hiring referral onboarding resume position headcount sourcing shortlist",
        "performance": "review goals feedback promotion rating calibration okr appraisal coaching competency growth evaluation",
    },
    "finance": {
        "invoices": "invoice vendor payment overdue remittance billing receivable net30 purchase order statement settlement",
        "budget": "budget forecast quarter allocation spend variance capex opex plan headroom approval cost",
        "tax": "vat tax filing deduction audit withholding return authority exemption levy compliance assessment",
        "expenses": "expense receipt reimbursement travel mileage perdiem card claim hotel flight meal policy",
    },
    "engineering": {
        "incidents": "outage incident pager latency rollback postmortem severity alert downtime mitigation oncall failover",
        "deployments": "deploy release pipeline staging canary build artifact rollout version ci merge branch",
        "architecture": "service database schema api queue cache microservice design scaling sharding gateway replication",
        "security": "vulnerability patch cve firewall encryption token phishing credential scan pentest mfa access",
    },
    "legal": {
        "contracts": "contract clause renewal termination liability indemnity signature amendment term counterparty obligation agreement",
        "compliance": "gdpr regulation policy consent retention privacy audit controls certification soc2 breach notice",
        "ip": "patent trademark copyright license invention filing infringement priorart claim royalty disclosure opensource",
        "disputes": "litigation lawsuit settlement court arbitration claimant counsel hearing damages subpoena mediation filing",
    },
    "sales": {
        "pipeline": "lead opportunity deal prospect quota forecast stage close crm demo proposal conversion",
        "pricing": "discount price tier quote margin list bundle subscription seat annual upsell rebate",
        "accounts": "customer account renewal churn expansion champion stakeholder contract health executive sponsor retention",
        "marketing": "campaign webinar newsletter brand ads launch audience funnel seo event content impressions",
    },
    "support": {
        "tickets": "ticket escalation sla queue priority customer response resolution backlog triage agent case",
        "knowledge": "article faq guide troubleshooting howto steps documentation workaround solution manual tutorial reference",
        "hardware": "laptop monitor printer keyboard dock headset replacement warranty device charger peripheral imaging",
        "access": "permission account password reset vpn group provisioning license sso mailbox role unlock",
    },
}
TOPICS = {d: {t: w.split() for t, w in ts.items()} for d, ts in TOPICS.items()}

FILLER = ("update team note please check latest regarding attached thanks week today status "
          "info details follow question shared draft final summary").split()
STEMS = ("kestrel orion basalt nimbus falcon juniper cobalt quartz atlas zephyr ember sable "
         "harbor tundra willow vortex onyx cedar meridian halcyon lumen aurora granite delta "
         "sierra beacon pioneer summit ridge crest").split()
CHANNELS = ["general", "random", "standup", "announcements"]
QUESTION_STARTS = ["what is the status of", "find the", "where is the", "details on", "who handles"]

# Which topics are restricted, and to whom.
RESTRICTED = {("hr", "payroll"): {"hr"}, ("finance", "tax"): {"finance"},
              ("legal", "disputes"): {"legal"}}


def build_company(sources_per_topic=4, docs_per_source=40, noise_ratio=0.5,
                  cross_mention_rate=0.3, seed=0):
    """Returns (sources, facts). sources: list of dicts for MapNet.register_source.
    facts: list of (doc_id, entity, domain, topic) for question generation."""
    rng = random.Random(seed)
    counter = iter(range(10 ** 9))
    new_entity = lambda: f"{rng.choice(STEMS)}{next(counter)}"
    sources, facts = [], []
    for domain, topics in TOPICS.items():
        for topic, vocab in topics.items():
            for s in range(sources_per_topic):
                docs = []
                for d in range(docs_per_source):
                    ent = new_entity()
                    words = rng.sample(vocab, 5) + rng.sample(FILLER, 4)
                    if facts and rng.random() < cross_mention_rate:
                        words.append(rng.choice(facts)[1])  # mentions some other project
                    rng.shuffle(words)
                    doc_id = f"{domain}.{topic}.{s}.{d}"
                    docs.append((doc_id, f"{ent}: {' '.join(words)} (ref {ent})"))
                    facts.append((doc_id, ent, domain, topic))
                sources.append(dict(name=f"{topic}-system-{s}", path=[domain, topic], docs=docs,
                                    acl=tuple(RESTRICTED.get((domain, topic), {"*"}))))
    all_vocab = [w for ts in TOPICS.values() for v in ts.values() for w in v]
    n_noise = int(len(sources) * noise_ratio)
    for s in range(n_noise):
        ch = CHANNELS[s % len(CHANNELS)]
        docs = []
        for d in range(docs_per_source):
            words = ([rng.choice(facts)[1] for _ in range(2)] + rng.sample(all_vocab, 2)
                     + rng.sample(FILLER, 6))
            rng.shuffle(words)
            docs.append((f"chat.{ch}.{s}.{d}", " ".join(words)))
        sources.append(dict(name=f"{ch}-{s}", path=["chat", ch], docs=docs, acl=("*",)))
    return sources, facts


# How people actually ask, in words the documents never use. This is the case
# where automatic summaries fail and feedback has to teach the map.
SLANG = {
    "payroll": "paycheck paystub", "benefits": "perks healthplan", "recruiting": "hire applicant",
    "performance": "scorecard selfreview", "invoices": "bill supplier", "budget": "funding spending",
    "tax": "irs taxman", "expenses": "refund trip", "incidents": "broken down",
    "deployments": "ship push", "architecture": "backend infra", "security": "hack exploit",
    "contracts": "paperwork signoff", "compliance": "regulator rules", "ip": "invent protect",
    "disputes": "sue judge", "pipeline": "prospects funnelstage", "pricing": "cheaper pricey",
    "accounts": "client relationship", "marketing": "promo advert", "tickets": "complaint helpdesk",
    "knowledge": "wiki docs", "hardware": "computer gadget", "access": "login locked",
}
SLANG = {t: w.split() for t, w in SLANG.items()}


def make_questions(facts, n, seed=1, exclude=(), slang=False):
    """slang=True: the question uses everyday words that appear in no document."""
    rng = random.Random(seed)
    exclude = set(exclude)
    pool = [f for f in facts if f[0] not in exclude]
    out = []
    for doc_id, ent, domain, topic in rng.sample(pool, min(n, len(pool))):
        words = SLANG[topic] if slang else rng.sample(TOPICS[domain][topic], 2)
        out.append((f"{rng.choice(QUESTION_STARTS)} {ent} {' '.join(words)}", doc_id))
    return out
