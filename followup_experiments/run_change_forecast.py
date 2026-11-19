"""Reproducible level-vs-change forecasting follow-up for three dengue series.

Uses the original XGBoost/LightGBM models, fixed parameters, 1/2/4/8/12
history windows, persistence baseline, expanding 70% training window and
non-overlapping 13-week test blocks. All result artifacts are written only to
followup_experiments/results and followup_experiments/figures.
"""
from __future__ import annotations
import argparse
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
import xgboost
import lightgbm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'singapore_analysis'))
from data_loader import load_raw, CITIES
from evaluation import rolling_origin_splits
from features import HISTORY_WINDOWS, build_aligned_tables
from models import RANDOM_SEED, get_models
from preprocessing import preprocess
from singapore_data import load_singapore

OUT = Path(__file__).resolve().parent
RESULTS = OUT / 'results'
FIGURES = OUT / 'figures'
BLOCK = 13
TRAIN_FRAC = 0.70
METHODS = ('level', 'change')


def get_series():
    result = {
        'San Juan': preprocess(load_raw('sj')),
        'Iquitos': preprocess(load_raw('iq')),
        'Singapore': load_singapore(),
    }
    # Use the original MOH epi-week number for Singapore seasonal features.
    sg = pd.read_csv(ROOT / 'data/raw/singapore_dengue_fever_weekly_clean.csv')
    sg_dates = pd.DataFrame({'date': load_singapore()['date'], 'week_of_year': sg['epi_week_number']})
    week_map = dict(zip(sg_dates.date, sg_dates.week_of_year))
    return result, {'Singapore': week_map}


def build_tables(series, week_map=None):
    tables = build_aligned_tables(series, HISTORY_WINDOWS)
    for w, t in tables.items():
        if week_map:
            t['week_of_year'] = t.date.map(week_map).astype(int)
        else:
            t['week_of_year'] = t.date.dt.isocalendar().week.astype(int)
        t['season_sin'] = np.sin(2 * np.pi * t.week_of_year / 52.0)
        t['season_cos'] = np.cos(2 * np.pi * t.week_of_year / 52.0)
        t['delta_log'] = np.log1p(t.target) - np.log1p(t.lag_1)
    lengths = {len(t) for t in tables.values()}
    assert len(lengths) == 1
    return tables


def predict_level_or_change(model, method, x_train, train, x_test, test):
    y_train = train.target if method == 'level' else train.delta_log
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if method == 'change':
        pred = np.expm1(np.log1p(test.lag_1.to_numpy(dtype=float)) + pred)
        pred = np.maximum(pred, 0.0)
    return np.asarray(pred, dtype=float)


def run_one_series(name, series, week_map=None):
    tables = build_tables(series, week_map)
    n = len(tables[max(HISTORY_WINDOWS)])
    min_train = int(n * TRAIN_FRAC)
    rows = []
    for w, table in tables.items():
        folds = rolling_origin_splits(table, min_train, BLOCK)
        for fold, train, test in folds:
            # Add the shared baseline once; folds are identical across windows.
            if w == min(HISTORY_WINDOWS):
                for _, row in test.iterrows():
                    rows.append(dict(series=name, method='persistence', model='Persistence',
                                     history_window=0, fold=fold, date=row.date,
                                     y_true=float(row.target), y_pred=float(row.lag_1)))
            # The two approaches and all model/window configurations receive
            # precisely the same train/test dates within each series.
            for method in METHODS:
                for model_name, model in get_models().items():
                    lag_cols = [f'lag_{k}' for k in range(1, w + 1)]
                    pred = predict_level_or_change(model, method, train[lag_cols], train,
                                                   test[lag_cols], test)
                    for (_, row), yhat in zip(test.iterrows(), pred):
                        rows.append(dict(series=name, method=method, model=model_name,
                                         history_window=w, fold=fold, date=row.date,
                                         y_true=float(row.target), y_pred=float(yhat)))
    oos = pd.DataFrame(rows)
    oos['abs_error'] = (oos.y_true - oos.y_pred).abs()
    oos['sq_error'] = (oos.y_true - oos.y_pred) ** 2
    return oos, n, min_train, (n - min_train) // BLOCK


def summarize(oos):
    base = (oos[oos.method == 'persistence'].groupby('series')
            .agg(persistence_MAE=('abs_error', 'mean'),
                 persistence_RMSE=('sq_error', lambda x: float(np.sqrt(x.mean()))),
                 n_oos=('date', 'nunique')).reset_index())
    rows = []
    main = oos[oos.method.isin(METHODS) | oos.method.str.endswith('_seasonal')]
    for (series, method, model, window), g in main.groupby(['series','method','model','history_window']):
        mae = float(g.abs_error.mean()); rmse = float(np.sqrt(g.sq_error.mean()))
        rows.append(dict(series=series, method=method, model=model, history_window=int(window),
                         n_oos=len(g), MAE=mae, RMSE=rmse))
    result = pd.DataFrame(rows).merge(base[['series','persistence_MAE','persistence_RMSE']], on='series')
    result['relative_MAE'] = result.MAE / result.persistence_MAE
    result['relative_RMSE'] = result.RMSE / result.persistence_RMSE
    result = result.sort_values(['series','method','model','history_window']).reset_index(drop=True)
    # Baseline rows are explicit in a companion tidy table; its relative MAE is 1.
    baseline = base.assign(method='persistence', model='Persistence', history_window=0,
                           MAE=lambda d:d.persistence_MAE, RMSE=lambda d:d.persistence_RMSE,
                           relative_MAE=1.0, relative_RMSE=1.0)
    baseline = baseline[['series','method','model','history_window','n_oos','MAE','RMSE',
                         'persistence_MAE','persistence_RMSE','relative_MAE','relative_RMSE']]
    return pd.concat([result,baseline],ignore_index=True).sort_values(
        ['series','method','model','history_window']).reset_index(drop=True)


def dm_h1(a, b):
    """Two-sided DM test on pointwise absolute errors, h=1, HLN small-sample correction.

    h=1 means the forecast-error differential variance uses lag zero (no
    overlapping-horizon autocovariance correction). The common weekly series
    may still have serial dependence; this assumption is recorded in the report.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3 or np.allclose(d, d[0]):
        return float(np.nan), float(np.nan), n
    var = d.var(ddof=1)
    if var == 0:
        return float(np.nan), float(np.nan), n
    dm = d.mean() / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2*1 + 1*(1-1)/n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n-1)
    return float(stat), float(p), n


def holm(p):
    p = np.asarray(p,float); out=np.full(len(p),np.nan)
    idx=np.where(np.isfinite(p))[0]
    order=idx[np.argsort(p[idx])]; running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(idx)-rank)*p[i])); out[i]=running
    return out


def make_dm_table(oos, summary, series_name):
    source=oos[oos.series==series_name]
    rows=[]
    base=source[source.method=='persistence'].set_index(['date','fold']).abs_error
    for method in METHODS:
        for model in ('XGBoost','LightGBM'):
            for w in HISTORY_WINDOWS:
                a=source[(source.method==method)&(source.model==model)&(source.history_window==w)].set_index(['date','fold']).abs_error
                a,b=a.align(base,join='inner')
                stat,p,n=dm_h1(a,b)
                rows.append(dict(series=series_name,comparison='model/window vs persistence',method=method,model=model,
                                 history_window=w,short_window='',long_window='',n_oos=n,mean_abs_error_difference=float((a-b).mean()),
                                 model_relative_MAE=float(a.mean()/b.mean()),
                                 DM_statistic=stat,p_value=p))
    summ=summary[(summary.series==series_name)&summary.method.isin(METHODS)]
    for method in METHODS:
        for model in ('XGBoost','LightGBM'):
            sub=summ[(summ.method==method)&(summ.model==model)]
            short=int(sub[sub.history_window.isin([1,2,4])].sort_values(['MAE','history_window']).iloc[0].history_window)
            long=int(sub[sub.history_window.isin([8,12])].sort_values(['MAE','history_window']).iloc[0].history_window)
            a=source[(source.method==method)&(source.model==model)&(source.history_window==short)].set_index(['date','fold']).abs_error
            b=source[(source.method==method)&(source.model==model)&(source.history_window==long)].set_index(['date','fold']).abs_error
            a,b=a.align(b,join='inner'); stat,p,n=dm_h1(a,b)
            rows.append(dict(series=series_name,comparison='best short vs best long (selected by OOS MAE)',method=method,model=model,
                             short_window=short,long_window=long,n_oos=n,mean_abs_error_difference=float((a-b).mean()),
                             short_relative_MAE=float(a.mean()/base.loc[a.index].mean()),
                             long_relative_MAE=float(b.mean()/base.loc[b.index].mean()),
                             DM_statistic=stat,p_value=p))
    out=pd.DataFrame(rows)
    out['p_holm_within_series']=holm(out.p_value)
    out['loss_difference_note']='negative mean difference favors first-listed model/configuration'
    return out


def run_selected_seasonal_configs(name, series, summary, week_map=None):
    """Refit only each model/target's best short and best long windows with seasonality."""
    tables=build_tables(series,week_map)
    n=len(tables[max(HISTORY_WINDOWS)]); min_train=int(n*TRAIN_FRAC); rows=[]
    gsummary=summary[(summary.series==name)&summary.method.isin(METHODS)]
    for method in METHODS:
        for model_name in ('XGBoost','LightGBM'):
            group=gsummary[(gsummary.method==method)&(gsummary.model==model_name)]
            chosen=[]
            chosen.append(('best_short',int(group[group.history_window.isin([1,2,4])]
                                             .sort_values(['MAE','history_window']).iloc[0].history_window)))
            chosen.append(('best_long',int(group[group.history_window.isin([8,12])]
                                            .sort_values(['MAE','history_window']).iloc[0].history_window)))
            for label,w in chosen:
                table=tables[w]; lag_cols=[f'lag_{k}' for k in range(1,w+1)]
                xcols=lag_cols+['season_sin','season_cos']
                for fold,train,test in rolling_origin_splits(table,min_train,BLOCK):
                    model=get_models()[model_name]
                    pred=predict_level_or_change(model,method,train[xcols],train,test[xcols],test)
                    for (_,row),yhat in zip(test.iterrows(),pred):
                        rows.append(dict(series=name,method=method+'_seasonal',model=model_name,
                                         history_window=w,selected_configuration=label,fold=fold,
                                         date=row.date,y_true=float(row.target),y_pred=float(yhat)))
    out=pd.DataFrame(rows)
    out['abs_error']=(out.y_true-out.y_pred).abs(); out['sq_error']=(out.y_true-out.y_pred)**2
    return out


def seasonality_sensitivity(oos, summary):
    rows=[]
    seasonal=oos[oos.method.str.endswith('_seasonal')]
    for (series,method,model), group in summary[summary.method.isin(METHODS)].groupby(['series','method','model']):
        short=int(group[group.history_window.isin([1,2,4])].sort_values(['MAE','history_window']).iloc[0].history_window)
        long=int(group[group.history_window.isin([8,12])].sort_values(['MAE','history_window']).iloc[0].history_window)
        for label,w in [('best_short',short),('best_long',long)]:
            base=oos[(oos.series==series)&(oos.method==method)&(oos.model==model)&(oos.history_window==w)].set_index(['date','fold'])
            seas=seasonal[(seasonal.series==series)&(seasonal.method==method+'_seasonal')&
                          (seasonal.model==model)&(seasonal.history_window==w)].set_index(['date','fold'])
            joined=base[['y_true','y_pred']].join(seas[['y_pred']],lsuffix='_base',rsuffix='_seasonal')
            base_mae=float((joined.y_true-joined.y_pred_base).abs().mean())
            seas_mae=float((joined.y_true-joined.y_pred_seasonal).abs().mean())
            base_rmse=float(np.sqrt(((joined.y_true-joined.y_pred_base)**2).mean()))
            seas_rmse=float(np.sqrt(((joined.y_true-joined.y_pred_seasonal)**2).mean()))
            persist=oos[(oos.series==series)&(oos.method=='persistence')].abs_error.mean()
            rows.append(dict(series=series,method=method,model=model,configuration=label,history_window=w,n_oos=len(joined),
                             MAE_without_seasonality=base_mae,MAE_with_seasonality=seas_mae,
                             relative_MAE_without_seasonality=base_mae/persist,relative_MAE_with_seasonality=seas_mae/persist,
                             change_in_MAE=seas_mae-base_mae,RMSE_without_seasonality=base_rmse,
                             RMSE_with_seasonality=seas_rmse,change_in_RMSE=seas_rmse-base_rmse,
                             selected_as_best='window selected by no-seasonality OOS MAE'))
    return pd.DataFrame(rows)


def plot_relative(summary, series):
    s=summary[(summary.series==series)&summary.method.isin(METHODS)]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    colors={'XGBoost':'#2a78d6','LightGBM':'#eb6834'}
    for ax,method in zip(axs,METHODS):
        sub=s[s.method==method]
        for model,g in sub.groupby('model'):
            g=g.sort_values('history_window')
            ax.plot(g.history_window,g.relative_MAE,marker='o',linewidth=2,label=model,color=colors[model])
        ax.axhline(1,color='black',linestyle='--',linewidth=1,label='Persistence (relative MAE = 1)')
        ax.set_title('Direct case-count target' if method=='level' else 'Log-change target')
        ax.set_xlabel('History window (weeks)'); ax.set_xticks(HISTORY_WINDOWS); ax.grid(alpha=.25)
    axs[0].set_ylabel('Relative MAE (model MAE / persistence MAE)')
    axs[1].legend(loc='best',fontsize=8)
    fig.suptitle(f'{series}: level and change forecasts on 13-week rolling-origin folds')
    fig.tight_layout()
    fig.savefig(FIGURES/f'{series.lower().replace(" ","_")}_relative_mae.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); args=ap.parse_args()
    if not args.execute:
        print('Dry run only. Add --execute to train all models.')
        return
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    RESULTS.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    versions={'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,
              'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
              'xgboost':xgboost.__version__,'lightgbm':lightgbm.__version__,
              'random_seed':RANDOM_SEED,'python_random_seed':RANDOM_SEED,'numpy_seed':RANDOM_SEED,
              'history_windows':HISTORY_WINDOWS,'train_fraction':TRAIN_FRAC,'test_block_weeks':BLOCK,
              'statistical_test':'Diebold-Mariano, absolute-error loss, horizon=1, HLN small-sample correction; Holm across all tests within each series'}
    (RESULTS/'run_manifest.json').write_text(json.dumps(versions,indent=2)+'\n')
    data,week_maps=get_series(); all_oos=[]; plans=[]
    for name,series in data.items():
        print(f'Running {name}: {len(series)} weekly observations...',flush=True)
        oos,n,min_train,nfolds=run_one_series(name,series,week_maps.get(name))
        all_oos.append(oos)
        plans.append(dict(series=name,observations=len(series),aligned_rows=n,min_train=min_train,test_block_weeks=BLOCK,
                          n_folds=nfolds,unused_test_rows=n-min_train-nfolds*BLOCK))
    primary_oos=pd.concat(all_oos,ignore_index=True)
    primary_summary=summarize(primary_oos)
    seasonal_oos=[]
    for name,series in data.items():
        print(f'Running selected seasonal configurations: {name}...',flush=True)
        seasonal_oos.append(run_selected_seasonal_configs(name,series,primary_summary,week_maps.get(name)))
    oos=pd.concat([primary_oos,*seasonal_oos],ignore_index=True)
    summary=summarize(oos)
    pmae_by_series={name:float(g[g.method=='persistence'].abs_error.mean()) for name,g in oos.groupby('series')}
    for (name,method,model,w), idx in oos.groupby(['series','method','model','history_window']).groups.items():
        config_mae=float(oos.loc[idx,'abs_error'].mean())
        oos.loc[idx,'relative_MAE']=config_mae/pmae_by_series[name]
    oos.to_csv(RESULTS/'oos_predictions.csv',index=False)
    summary.to_csv(RESULTS/'all_models_relative_mae_summary.csv',index=False)
    pd.DataFrame(plans).to_csv(RESULTS/'fold_plan_summary.csv',index=False)
    for name in data:
        make_dm_table(primary_oos,primary_summary,name).to_csv(RESULTS/f'dm_tests_{name.lower().replace(" ","_")}.csv',index=False)
    seas=seasonality_sensitivity(oos,primary_summary)
    seas.to_csv(RESULTS/'seasonality_sensitivity.csv',index=False)
    for name in data: plot_relative(summary,name)
    print('\nFold plan:'); print(pd.DataFrame(plans).to_string(index=False))
    print('\nPrimary model summaries:'); print(summary[summary.method.isin(METHODS)].round(3).to_string(index=False))
    print('\nSeasonality:'); print(seas.round(3).to_string(index=False))
    print(f'\nOutputs: {RESULTS}; figures: {FIGURES}')

if __name__=='__main__': main()
