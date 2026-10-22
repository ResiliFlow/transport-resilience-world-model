"""Independent, loopback-only road safety runtime. Python standard library only.

No imports from TWM, no research-drive dependency, no model training, no writes
to the existing platform. Plans are user-controlled browser/download artefacts.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import mimetypes
import zipfile
import engineering
from functools import lru_cache
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / 'bundle/manifest.json'
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding='utf-8')) if MANIFEST_PATH.is_file() else None
SERVICE = 'twm-road-safety'
VERSION = '1.7.2'
METRIC_INDEX = {'fatal': 0, 'serious': 1, 'high': 6, 'rising': 7}
OBSERVED_INDEX = {'fatal': 8, 'serious': 9, 'high': 10, 'rising': 11}

@lru_cache(maxsize=2)
def dataset(name):
    if name not in ('future','history'): raise ValueError('Choose future or history.')
    with gzip.open(ROOT/f'bundle/{name}.json.gz','rt',encoding='utf-8') as f:
        data = json.load(f)
    return data

def clean_query(q):
    name=str(q.get('dataset','future'))
    data=dataset(name)
    period=str(q.get('period','2027-2030' if name=='future' else '2025'))
    if period not in data['forecasts']: raise ValueError('This forecast period is not available.')
    metric=str(q.get('metric','serious'))
    valid=tuple(METRIC_INDEX)
    if metric not in valid: raise ValueError('Choose a valid outcome for this view.')
    region=str(q.get('region','All areas'))[:100]
    search=str(q.get('search','')).strip()[:80]
    return {'dataset':name,'period':period,'metric':metric,'region':region,'search':search}

def rows_for(q):
    data=dataset(q['dataset'])
    vals=data['forecasts'][q['period']]
    pos=METRIC_INDEX[q['metric']]
    result=[]
    for r,v in zip(data['roads'],vals):
        if q['region']!='All areas' and r['region']!=q['region']: continue
        if q['search'] and q['search'].casefold() not in (r['road']+' '+r['id']).casefold(): continue
        row={**r,'value':v[pos],'values':v}
        if q['dataset']=='history':
            row['historical_band']=v[12]
        result.append(row)
    result.sort(key=lambda r:(-r['value'],r['id']))
    return result

def summarise(rows,q):
    history=q['dataset']=='history'
    summary={'count':len(rows),'km':math.fsum(r['km'] for r in rows),
             'expected_first':math.fsum(r['values'][0] for r in rows),
             'expected_second':math.fsum(r['values'][1] for r in rows),
             'observed_first':math.fsum(r['values'][8] for r in rows) if history else None,
             'observed_second':math.fsum(r['values'][9] for r in rows) if history else None,
             'missing_map':sum(r['id'] in dataset(q['dataset'])['missing_geometry'] for r in rows)}
    if history:
        summary['evaluation']=evaluate_history(rows,q['metric'])
    return summary

def evaluate_history(rows,metric):
    probability=metric in ('high','rising')
    eligible=[r for r in rows if metric!='rising' or r['values'][12]<3]
    n=len(eligible)
    pairs=[(r['value'],r['values'][OBSERVED_INDEX[metric]]) for r in eligible]
    result={'metric':metric,'kind':'probability' if probability else 'count',
            'n':n,'excluded':len(rows)-n,'scope':'all_filtered_roads',
            'predicted_total':math.fsum(p for p,o in pairs),
            'observed_total':math.fsum(o for p,o in pairs)}
    result.update({'predicted_mean':result['predicted_total']/n if n else None,
                   'observed_mean':result['observed_total']/n if n else None,
                   'mae':math.fsum(abs(p-o) for p,o in pairs)/n if n else None,
                   'rmse':math.sqrt(math.fsum((p-o)**2 for p,o in pairs)/n) if n else None})
    if probability:
        result['brier']=math.fsum((p-o)**2 for p,o in pairs)/n if n else None
        bins=[]
        for i in range(5):
            selected=[(p,o) for p,o in pairs if min(4,int(p*5))==i]
            if selected:
                bins.append({'lower':i/5,'upper':(i+1)/5,'n':len(selected),
                             'predicted':math.fsum(p for p,o in selected)/len(selected),
                             'actual':math.fsum(o for p,o in selected)/len(selected)})
        result['calibration']=bins
    else:
        lower=2 if metric=='fatal' else 4
        result['coverage_95']=sum(r['values'][lower]<=r['values'][OBSERVED_INDEX[metric]]<=r['values'][lower+1] for r in eligible)/n if n else None
    return result

def yearly_history():
    result=[]
    for year in dataset('history')['forecasts']:
        q=clean_query({'dataset':'history','period':year})
        rows=rows_for(q)
        result.append({'year':int(year),'count':len(rows),
                       'metrics':{m:evaluate_history([{**r,'value':r['values'][METRIC_INDEX[m]]} for r in rows],m) for m in METRIC_INDEX}})
    return result

def results(q):
    q=clean_query(q);rows=rows_for(q)
    return {'query':q,'summary':summarise(rows,q),'rows':rows[:200],
            'map':[[r['id'],r['value']] for r in rows], 'list_limit':200,'bundle_id':MANIFEST['bundle_id']}

def plan(q):
    settings=clean_query(q)
    try: capacity=float(q.get('capacity',100))
    except (TypeError,ValueError): raise ValueError('Enter a numeric inspection capacity.')
    if not math.isfinite(capacity) or capacity<=0 or capacity>100000:
        raise ValueError('Capacity must be greater than zero and at most 100,000.')
    kind=q.get('capacity_unit','km')
    if kind not in ('km','roads'): raise ValueError('Capacity must use km or road units.')
    if kind=='roads' and not capacity.is_integer(): raise ValueError('Enter a whole number of road units.')
    candidates=rows_for(settings)
    selected=[];used=0
    for row in candidates:
        cost=row['km'] if kind=='km' else 1
        if cost<=0: continue
        if used+cost<=capacity+1e-9:
            selected.append(row);used+=cost
    if not selected:
        minimum=min((r['km'] for r in candidates if r['km']>0),default=None)
        message='No roads match these filters.' if not candidates else f'No whole road unit fits. The shortest matching unit is {minimum:.2f} km.'
    else: message=f'{len(selected)} whole road units selected.'
    return {'schema':1,'bundle_id':MANIFEST['bundle_id'],'query':settings,'capacity':capacity,'capacity_unit':kind,'used':used,'remaining':max(0,capacity-used),'rows':selected,'summary':summarise(selected,settings),'message':message,
            'selection_rule':'Highest selected-outcome score first. Skip whole road units that do not fit. This is a screening list, not a travel route or an estimate of injuries prevented.'}

def export_csv(data):
    output=io.StringIO(newline='');writer=csv.writer(output)
    future=data['query']['dataset']=='future'
    names=dataset(data['query']['dataset'])['fields']
    writer.writerow(['road_unit_id','road','area','reference_length_km','period','ranking_outcome','ranking_value',*names,'geometry_available','bundle_id'])
    missing=set(dataset(data['query']['dataset'])['missing_geometry'])
    for r in data['rows']:
        cells=[r['id'],r['road'],r['region'],r['km'],data['query']['period'],data['query']['metric'],r['value'],*r['values'],r['id'] not in missing,MANIFEST['bundle_id']]
        writer.writerow([("'"+x) if isinstance(x,str) and x.startswith(('=','+','-','@')) else x for x in cells])
    return ('\ufeff'+output.getvalue()).encode('utf-8')

def table_bytes(header, rows=()):
    output=io.StringIO(newline=''); writer=csv.writer(output)
    writer.writerow(header)
    for row in rows:
        writer.writerow([("'"+value) if isinstance(value,str) and value.startswith(('=','+','-','@')) else value for value in row])
    return ('\ufeff'+output.getvalue()).encode('utf-8')

DATA_GUIDE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Road Safety data guide</title><style>body{font:16px/1.65 Arial,sans-serif;color:#1b3042;max-width:900px;margin:40px auto;padding:0 22px}h1{font-size:32px}h2{margin-top:30px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd7de;padding:10px;text-align:left;vertical-align:top}th{background:#eff4f7}.note{padding:16px;background:#fff5cd;border-left:4px solid #c7a223}code{overflow-wrap:anywhere}a{color:#165c79}</style></head><body>
<h1>Road Safety data guide</h1><p>Transport World Model · Research preview</p>
<p class="note"><strong>No dataset upload is enabled in this release.</strong> Research data and forecasts are supplied separately. Blank preparation templates describe the expected fields. These templates are not the trained models' complete feature specification, and do not guarantee compatibility with a new region.</p>
<h2>What is in the project sample?</h2><p>With an authorised compatible bundle installed, <code>future_people/</code> contains supplied projections and <code>past_people/</code> contains supplied historical predictions and observations. Join the folders by <code>road_id</code>, not road name alone. The sample explains the file structure; it is not a performance benchmark.</p>
<p>Exports contain results from the supplied bundle; no research inputs are distributed with this repository. A road unit can include several disconnected sections. The sample selection is deterministic, covers available areas and excludes units without geometry. It is not a random or representative performance sample.</p>
<h2>Prepare your own area's inputs</h2><table><thead><tr><th>File</th><th>Key fields</th><th>Meaning</th></tr></thead><tbody>
<tr><td>roads_template.csv</td><td>road_id, road_name, area, length_km, speed_limit_kmh, road_class, junction_count</td><td>One row per consistently defined road segment. Kilometres and km/h. Keep one ID across all files and years. An A-road number alone is not a unique segment ID.</td></tr>
<tr><td>annual_safety_template.csv</td><td>road_id, year, injury_crashes, severe_crashes, fatalities, serious_injuries, reporting_complete</td><td>Crashes are events. Fatalities and serious injuries count people. Severe crashes involve at least one death or serious injury and are a subset of injury crashes. Reporting complete is true/false.</td></tr>
<tr><td>annual_traffic_template.csv</td><td>road_id, year, annual_vehicle_km, aadt_vehicles_per_day, exposure_source</td><td>Annual vehicle-kilometres or annual average daily traffic (AADT), with a documented source. Do not mix vehicles/day with vehicle-km/year. Document changes in road coverage.</td></tr>
<tr><td>road_geometry_template.geojson</td><td>FeatureCollection; each feature has properties.road_id</td><td>LineString or MultiLineString coordinates in WGS84 [longitude, latitude]. Replace the empty collection with your actual roads. Do not use centroids as road lines.</td></tr>
</tbody></table>
<h2>Checks before any future import</h2><ol><li>Use the same road_id in geometry, road properties and annual records.</li><li>Provide several consecutive historical years, including genuinely observed zero counts. Missing data are blank, not zero.</li><li>Record non-negative integer counts and consistent severity definitions. Severe crashes cannot exceed all injury crashes.</li><li>Record positive road lengths and non-negative traffic exposure. If deriving vehicle-km from daily traffic, document length, direction and the year's day count.</li><li>Do not include names, licence plates or personal details. Aggregated segment-year records are sufficient for these templates.</li><li>Forecasting for a new area requires data validation and a separately tested model workflow. No training occurs when opening this demo.</li></ol>
<h2>Reading sample forecast fields</h2><p><code>predicted_fatal</code> and <code>predicted_serious</code> are expected people and may be fractional. <code>observed_fatal</code> and <code>observed_serious</code> are recorded people. Individual 95% limits concern realised counts; do not add interval bounds into a regional interval.</p><p><code>band_probability_3</code> is the probability of high harm. <code>escalation_probability</code> is the probability of a higher harm level than the two-year baseline. Probabilities range from 0 to 1. <code>high_harm_observed</code> and <code>escalation_observed</code> are actual events: 1 means yes and 0 means no, not measured probabilities.</p><p>Harm levels range from 0 to 3, with annual death boundaries of 0.2, 0.5 and 1, or serious-injury boundaries of 1, 3 and 6. Reaching either threshold raises the level. Historical checks use one target year; a 2025 check uses information through 2023 and a 2022–2023 baseline. Earlier checks follow the same two-year offset. Future projections use records through 2025 and a 2024–2025 baseline. Historical person counts must not be summed across repeated period rows.</p><p>Probability reliability is evaluated against actual events across roads. Roads already at baseline level 3 are excluded from worsening-harm reliability summaries, because they cannot move higher. Annual checks do not directly measure four-year forecast accuracy. Future actual outcomes are not yet available.</p>
<h2>Data sources and attribution</h2><p>This source release includes no road-safety research data or saved forecasts. Any supplied bundle must document its sources, permissions and required attribution. Retain those notices with exported results.</p></body></html>"""

DATA_GUIDE = DATA_GUIDE.replace('<h2>Data sources and attribution</h2>', '<h2>SPF and CMF examples</h2><p>The engineering folder contains SPF baselines for the same sample roads and a CMF evidence record. SPF means Safety Performance Function. It estimates annual injury crashes, not people. Each row records its actual traffic reference year. Older reference years are not imputed as current data. CMF means Crash Modification Factor. Selecting a supported measure displays a fixed-assumption reference estimate, followed by its read-only conditions. The comparison assumes coverage of the whole road unit, no previous camera enforcement and no simultaneous speed-limit or road-layout change. Both scenarios use the same reference-year traffic. These conditions are not verified site facts. Check them on site before implementation. Known unsupported road types remain excluded. Historical SPF checks use a pre-2020 model and recorded 2020-2025 traffic. The original person forecasts are unchanged.</p><h2>Data sources and attribution</h2>')

@lru_cache(maxsize=2)
def download_package(kind):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data_guide.html',DATA_GUIDE.encode('utf-8'))
        if kind=='data_templates':
            templates={
                'roads_template.csv':['road_id','road_name','area','length_km','speed_limit_kmh','road_class','junction_count'],
                'annual_safety_template.csv':['road_id','year','injury_crashes','severe_crashes','fatalities','serious_injuries','reporting_complete'],
                'annual_traffic_template.csv':['road_id','year','annual_vehicle_km','aadt_vehicles_per_day','exposure_source']}
            for name,fields in templates.items(): archive.writestr(name,table_bytes(fields))
            archive.writestr('road_geometry_template.geojson',json.dumps({'type':'FeatureCollection','features':[]}))
            archive.writestr('README.txt','Blank data-preparation templates. No upload is enabled in this research preview.\nRead data_guide.html for fields, units and validation rules.\nThese are not the full trained-model feature specification. Do not use made-up records or treat missing values as observed zeros.\n')
        elif kind=='project_sample':
            for name,folder in [('future','future_people'),('history','past_people')]:
                data=dataset(name);missing=set(data['missing_geometry']);chosen=[];areas=set()
                ordered=sorted(enumerate(data['roads']),key=lambda x:(x[1]['region'],x[1]['id']))
                for index,road in ordered:
                    if road['id'] not in missing and road['region'] not in areas:
                        chosen.append(index);areas.add(road['region'])
                    if len(chosen)>=20: break
                for index,road in ordered:
                    if len(chosen)>=20: break
                    if road['id'] not in missing and index not in chosen: chosen.append(index)
                ids={data['roads'][index]['id'] for index in chosen}
                archive.writestr(folder+'/roads.csv',table_bytes(['road_id','road_name','area','length_km','reference_longitude','reference_latitude'],[[data['roads'][i][k] for k in ('id','road','region','km','lon','lat')] for i in chosen]))
                archive.writestr(folder+'/forecasts.csv',table_bytes(['road_id','period',*data['fields']],[[data['roads'][i]['id'],period,*values[i]] for period,values in data['forecasts'].items() for i in chosen]))
                with gzip.open(ROOT/f'bundle/{name}_map.json.gz','rt',encoding='utf-8') as stream: geometry=json.load(stream)
                features=[{'type':'Feature','properties':{'road_id':f['properties']['id']},'geometry':f['geometry']} for f in geometry['features'] if f['properties']['id'] in ids]
                archive.writestr(folder+'/roads.geojson',json.dumps({'type':'FeatureCollection','features':features},separators=(',',':')))
            archive.writestr('engineering/spf_baselines.csv',table_bytes(['road_id','reference_year','length_km','daily_traffic','expected_injury_crashes_per_year','model_id'],[[key,engineering.road_result(key)['baseline_year'],engineering.road_result(key)['length_km'],engineering.road_result(key)['aadf'],engineering.road_result(key)['expected'],engineering.package()['model_id']] for key in sorted(ids)]))
            archive.writestr('engineering/cmf_evidence.json',json.dumps(engineering.package()['cmf'],indent=2))
            archive.writestr('README.txt','Real subset of frozen injury-study outputs, not raw training data.\nBundle: '+MANIFEST['bundle_id']+'\nThe two folders contain the same 20 mapped road units. Join them by road_id, not road number alone.\nUnits are chosen deterministically by area then road ID. Not a performance sample.\nFuture and historical files cover the same four outcomes. Probability events use 1=yes and 0=no.\nNo model was rerun or trained. Read data_guide.html.\n')
        else: raise ValueError('Unknown data package.')
    return buffer.getvalue()

class Handler(BaseHTTPRequestHandler):
    server_version='TWM-Road-Safety/1.0'
    def log_message(self,format,*args): pass

    def send(self,status,body,kind='application/json; charset=utf-8',extra=None):
        self.send_response(status)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','strict-origin-when-cross-origin')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https://tile.openstreetmap.org; connect-src 'self' https://tile.openstreetmap.org; font-src 'self'; frame-ancestors 'self'; object-src 'none'; base-uri 'self'")
        self.send_header('Cache-Control','no-cache')
        for k,v in (extra or {}).items(): self.send_header(k,v)
        self.end_headers()
        if self.command!='HEAD': self.wfile.write(body)

    def json(self,status,obj):
        self.send(status,json.dumps(obj,ensure_ascii=False,allow_nan=False).encode())

    def host_ok(self):
        host=self.headers.get('Host','').split(':')[0]
        return host in ('127.0.0.1','localhost')

    def do_HEAD(self): self.do_GET()

    def do_GET(self):
        try:
            if not self.host_ok(): return self.json(403,{'error':'Local access only.'})
            parsed=urlparse(self.path)
            path=unquote(parsed.path)
            q={k:v[0] for k,v in parse_qs(parsed.query).items()}
            if path=='/api/health':
                return self.json(200,{'status':'healthy','workspace':SERVICE,'version':VERSION,'bundle_id':MANIFEST['bundle_id'],'research_drive_required':False})
            if path=='/api/meta':
                meta={name:{'regions':sorted({r['region'] for r in dataset(name)['roads']}),'periods':list(dataset(name)['forecasts']),'count':len(dataset(name)['roads']),'missing_geometry':dataset(name)['missing_geometry'],'road_names':sorted({r['road'] for r in dataset(name)['roads']}),'road_labels':{r['id']:r['road'] for r in dataset(name)['roads']}} for name in ('future','history')}
                return self.json(200,{'bundle_id':MANIFEST['bundle_id'],'datasets':meta})
            if path in ('/downloads/project_sample.zip','/downloads/data_templates.zip'):
                kind=Path(path).stem
                return self.send(200,download_package(kind),'application/zip',{'Content-Disposition':f'attachment; filename="TWM_Road_Safety_{kind}.zip"'})
            if path=='/downloads/data_guide.html':
                return self.send(200,DATA_GUIDE.encode('utf-8'),'text/html; charset=utf-8',{'Content-Disposition':'attachment; filename="TWM_Road_Safety_Data_Guide.html"'})
            if path=='/api/results': return self.json(200,results(q))
            if path=='/api/results.csv':
                settings=clean_query(q)
                data={'query':settings,'rows':rows_for(settings)}
                filename=f"TWM_Road_Safety_{settings['dataset']}_{settings['period']}.csv"
                return self.send(200,export_csv(data),'text/csv; charset=utf-8',{'Content-Disposition':f'attachment; filename="{filename}"'})
            if path=='/api/road':
                settings=clean_query(q);data=dataset(settings['dataset'])
                idx=next((i for i,r in enumerate(data['roads']) if r['id']==q.get('id')),None)
                if idx is None: return self.json(404,{'error':'Road not found.'})
                return self.json(200,{'road':data['roads'][idx],'fields':data['fields'],'values':data['forecasts'][settings['period']][idx],'periods':{p:v[idx] for p,v in data['forecasts'].items()},'period_metadata':data.get('period_metadata',{}),'geometry_available':data['roads'][idx]['id'] not in data['missing_geometry']})
            if path=='/api/history/yearly': return self.json(200,yearly_history())
            if path=='/api/engineering/meta': return self.json(200,engineering.meta())
            if path=='/api/engineering/road': return self.json(200,engineering.road_result(q.get('id','')))
            if path=='/api/engineering/checks':
                settings=clean_query({'dataset':'history','period':'2025','region':q.get('region','All areas'),'search':q.get('search','')})
                return self.json(200,engineering.checks([r['id'] for r in rows_for(settings)]))
            routes={'/':'portal.html','/safety':'index.html','/safety/':'index.html','/favicon.ico':'assets/brands.png'}
            if path in routes: file=ROOT/routes[path]
            elif path.startswith('/assets/') or path in ('/app.js','/styles.css','/portal.js'):
                file=(ROOT/path.lstrip('/')).resolve()
                if not file.is_relative_to(ROOT): return self.json(403,{'error':'Invalid path.'})
            elif path.startswith('/map/') and path[5:] in ('future','history','outline'):
                name=path[5:];file=ROOT/'bundle'/('outline.json.gz' if name=='outline' else f'{name}_map.json.gz')
                body=file.read_bytes()
                if 'gzip' in self.headers.get('Accept-Encoding',''):
                    return self.send(200,body,extra={'Content-Encoding':'gzip','Vary':'Accept-Encoding'})
                return self.send(200,gzip.decompress(body))
            else: return self.json(404,{'error':'Not found.'})
            if not file.is_file(): return self.json(404,{'error':'Not found.'})
            return self.send(200,file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
        except (ValueError,KeyError) as e: self.json(400,{'error':str(e)})
        except (BrokenPipeError,ConnectionResetError): pass
        except Exception: self.json(500,{'error':'The local data could not be read. Restart this workspace and retry.'})

    def do_POST(self):
        try:
            if not self.host_ok(): return self.json(403,{'error':'Local access only.'})
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
                return self.json(403,{'error':'Cross-origin requests are not accepted.'})
            n=int(self.headers.get('Content-Length','0'))
            if n<=0 or n>32768: return self.json(413,{'error':'Request is too large or empty.'})
            q=json.loads(self.rfile.read(n))
            if not isinstance(q,dict): raise ValueError('Expected a settings object.')
            if self.path=='/api/plan': return self.json(200,plan(q))
            if self.path=='/api/engineering/scenario': return self.json(200,engineering.scenario(q))
            if self.path=='/api/export':
                data=plan(q)
                return self.send(200,export_csv(data),'text/csv; charset=utf-8',{'Content-Disposition':'attachment; filename="road_safety_inspection_plan.csv"'})
            return self.json(404,{'error':'Not found.'})
        except (ValueError,KeyError,TypeError) as e: self.json(400,{'error':str(e)})
        except (BrokenPipeError,ConnectionResetError): pass

def verify_bundle():
    if MANIFEST is None:
        raise RuntimeError('An authorised road-safety data bundle is required; it is not included in the source release.')
    for name,entry in MANIFEST['files'].items():
        p=ROOT/'bundle'/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
            raise RuntimeError(f'Bundle integrity check failed: {name}')

class RoadSafetyServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8010);args=parser.parse_args()
    try:
        verify_bundle()
    except (OSError, RuntimeError) as error:
        parser.error(str(error))
    server=RoadSafetyServer(('127.0.0.1',args.port),Handler)
    print(f'Road safety ready: http://127.0.0.1:{args.port}/safety',flush=True)
    server.serve_forever()
