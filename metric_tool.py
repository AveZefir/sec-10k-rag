import json
from pathlib import Path

def metric_tool(router_ans = None):
    companies_metrics = ''
    router_inf = json.loads(router_ans)
    companies = router_inf['companies']
    year = router_inf['year']
    metric = router_inf['metric']
    if metric is None or not companies:
        return None
    FACTS_DIR = Path(__file__).parent / "data" / "extracted_facts"
    Companies_CIK = {"Apple":"CIK0000320193.json", "Microsoft":"CIK0000789019.json","Amazon":"CIK0001018724.json"}
    Companies_months = {"Apple":"-09", "Microsoft":"-06","Amazon":"-12"}
    METRIC_TAGS = {
        "revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "net_income": "NetIncomeLoss",
    }
    for company in companies:
        Company_facts = json.load(open(FACTS_DIR / Companies_CIK[company]))
        try:
            company_records = Company_facts["facts"]["us-gaap"][METRIC_TAGS[metric]]["units"]["USD"]
        except KeyError:
            continue
        if year is None:
            fy_years = [n['end'][:4] for n in company_records
                        if n['end'][4:7] == Companies_months[company]]
            if not fy_years:
                continue
            years_for_company = [max(fy_years)]
        else:
            years_for_company = year if isinstance(year, list) else [year]
        for y in years_for_company:
            end = str(y) + Companies_months[company]
            metric_set = set()
            [metric_set.add((n['start'],n['end'],n['val'],n['filed'])) for n in company_records if n['end'].startswith(end)]
            if not metric_set:
                continue
            latest = max(metric_set, key=lambda item: item[3])
            companies_metrics += f'Company: {company}, year: {y}, {metric}: {latest[2]}\n'
    return companies_metrics or None