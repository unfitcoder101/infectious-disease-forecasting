"""Recreate the 2014-2018 MOH Singapore source overlap audit."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SG=Path(__file__).resolve().parent
START,END='2014-W01','2018-W52'
API=SG/'data/raw/moh_weekly_dengue_df_dhf_2014_2018_api_snapshot.json'
BULLETIN=SG/'data/raw/WeeklyInfectiousDiseaseBulletinCases.csv'
SELECTED=ROOT/'data/raw/singapore_dengue_fever_weekly_clean.csv'
TABLES=SG/'tables'


def main():
    payload=json.loads(API.read_text())
    assert payload['success'] and payload['result']['total']==530
    raw=pd.DataFrame(payload['result']['records'])
    valid=raw[raw.number.notna() & (raw.number!='NA')].copy()
    for c in ('year','eweek','number'):
        valid[c]=valid[c].astype(int)
    valid['epi_week']=[f'{y}-W{w:02d}' for y,w in zip(valid.year,valid.eweek)]
    valid=valid[valid.epi_week.between(START,END)]
    ref=valid.pivot(index='epi_week',columns='type_dengue',values='number').reset_index()
    ref=ref.rename(columns={'Dengue':'moh_dengue','DHF':'moh_dhf'})
    ref['dengue_plus_dhf']=ref.moh_dengue+ref.moh_dhf
    selected=pd.read_csv(SELECTED)
    selected=selected.loc[selected.epi_week.between(START,END),['epi_week','dengue_fever_cases']]
    out=selected.merge(ref,on='epi_week',how='outer',indicator=True).sort_values('epi_week')
    expected=[f'{y}-W{w:02d}' for y,limit in [(2014,53),(2015,52),(2016,52),(2017,52),(2018,52)] for w in range(1,limit+1)]
    assert len(expected)==261 and out.epi_week.tolist()==expected
    assert (out['_merge']=='both').all()
    out['moh_dengue_minus_project']=out.moh_dengue-out.dengue_fever_cases
    out['combined_minus_project']=out.dengue_plus_dhf-out.dengue_fever_cases
    out['dengue_count_mismatch']=out.moh_dengue_minus_project!=0
    out['combined_count_mismatch']=out.combined_minus_project!=0
    out.drop(columns='_merge').to_csv(TABLES/'overlap_check_2014_2018.csv',index=False)
    out[out.dengue_count_mismatch].drop(columns='_merge',errors='ignore').to_csv(
        TABLES/'overlap_mismatched_weeks_2014_2018.csv',index=False)

    bulletin=pd.read_csv(BULLETIN)
    broad=bulletin[(bulletin.disease=='Dengue Fever') & bulletin.epi_week.between(START,END)]
    broad=broad[['epi_week','no._of_cases']].rename(columns={'no._of_cases':'bulletin_dengue_fever'})
    broad_join=selected.merge(broad,on='epi_week',how='outer',indicator=True)
    assert len(broad_join)==261 and (broad_join['_merge']=='both').all()
    broad_diff=broad_join.bulletin_dengue_fever-broad_join.dengue_fever_cases
    diff=out.moh_dengue_minus_project
    summary={
        'comparison_period':f'{START} through {END}', 'expected_weeks':261,
        'dedicated_source_api_records_total':payload['result']['total'],
        'weeks_matched':len(out),'coverage_mismatches':int((out['_merge']!='both').sum()),
        'dedicated_dengue_exact_matches':int((diff==0).sum()),
        'dedicated_dengue_mismatches':int((diff!=0).sum()),
        'dedicated_dengue_mismatch_rate_pct':float((diff!=0).mean()*100),
        'dedicated_dengue_difference_min':int(diff.min()),
        'dedicated_dengue_difference_max':int(diff.max()),
        'dedicated_dengue_mean_absolute_difference':float(diff.abs().mean()),
        'combined_df_dhf_mismatches':int((out.combined_minus_project!=0).sum()),
        'combined_difference_min':int(out.combined_minus_project.min()),
        'combined_difference_max':int(out.combined_minus_project.max()),
        'combined_difference_equals_dhf_every_week':bool((out.combined_minus_project==out.moh_dhf).all()),
        'broader_bulletin_exact_matches':int((broad_diff==0).sum()),
        'broader_bulletin_mismatches':int((broad_diff!=0).sum()),
        'mismatches_by_year':out[diff!=0].assign(year=lambda d:d.epi_week.str[:4]).groupby('year').size().astype(int).to_dict(),
        'mismatched_weeks':out.loc[diff!=0,'epi_week'].tolist(),
        'dedicated_api_sha256':hashlib.sha256(API.read_bytes()).hexdigest(),
        'broader_bulletin_sha256':hashlib.sha256(BULLETIN.read_bytes()).hexdigest(),
    }
    (TABLES/'overlap_check_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
